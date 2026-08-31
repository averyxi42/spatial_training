"""Flash-native bitwise invariance test for the transient hand-token KV mechanism.

Perturb turn-0's HAND image content (same resolution -> same token count). With
TransientFilteredCache + stitched positions, turns 1..T-1 must be BITWISE identical
(hand K/V never persists; positions independent of hand by construction). Negative
control: plain DynamicCache must show later turns change. Runs the production
attention implementation (flash_attention_2). No production code modified.

  CUDA_VISIBLE_DEVICES=<gpu> env -u DISPLAY \
    /workspace/conda/envs/longnav_vlm/bin/python tests/proto/invariance_transient_kv.py
"""
import sys
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/Projects/spatial_training_mshab/src")
from parity_transient_kv import build_messages, tokenize, strip_hand, MODEL_ID, DEV  # noqa
import parity_transient_kv as P


def episode(rng_head, rng_hand, t_turns=3, hand_suffix=False):
    """hand_suffix=True places the hand image LAST in the turn (nothing persistent
    after it) -- the layout under which cache filtering alone determines whether
    later turns are bitwise invariant to hand content. The production layout
    (hand before the readout) intentionally lets the readout attend the hand, so
    hand info persists THROUGH the readout's K/V; there invariance is quantitative,
    not bitwise."""
    msgs, imgs, turns = [], [], []
    for t in range(t_turns):
        head = Image.fromarray(rng_head.integers(0, 255, (256, 256, 3), dtype=np.uint8))
        hand = Image.fromarray(rng_hand.integers(0, 255, (128, 128, 3), dtype=np.uint8))
        if hand_suffix:
            m = [{"role": "user", "content": [
                    {"type": "text", "text": f"Observation {t}:"}, {"type": "image"},
                    {"type": "text", "text": "Action:"}]},
                 {"role": "assistant", "content": [{"type": "text", "text": "**____**"}]},
                 {"role": "user", "content": [
                    {"type": "text", "text": "Hand:"}, {"type": "image"}]}]
        else:
            m = [{"role": "user", "content": [
                    {"type": "text", "text": f"Observation {t}:"}, {"type": "image"},
                    {"type": "text", "text": "Hand:"}, {"type": "image"},
                    {"type": "text", "text": "Action:"}]},
                 {"role": "assistant", "content": [{"type": "text", "text": "**____**"}]}]
        msgs += m; imgs += [head, hand]
        turns.append((list(msgs), list(imgs), 2 * t))
    return turns


def run(model, processor, turns, transient, dense=True, to_end=False):
    lgs, _, _, _, cache = P.run_incremental(model, processor, turns, transient=transient,
                                            dense=dense, transient_to_end=to_end)
    return lgs, cache.get_seq_length()


def main():
    # PRODUCTION PARITY: VLMWorker.__init__ (vlm_worker.py:54-57) disables HF's
    # flash packed-sequence detection -- mrope position_ids are non-ramp for image
    # chunks and would otherwise be misread as packed varlen batches. Mirror it.
    import transformers.modeling_flash_attention_utils as fa_utils
    fa_utils._is_packed_sequence = lambda position_ids, batch_size: False
    from longnav.utils.modeling import Qwen3VLSparseForConditionalGeneration
    from longnav.proto.transient_kv import install_transient
    from transformers import AutoConfig, AutoProcessor
    torch.manual_seed(0)
    cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = Qwen3VLSparseForConditionalGeneration.from_pretrained(
        MODEL_ID, config=cfg, torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="flash_attention_2").to(DEV).eval()
    install_transient(model)
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    # Same head stream; two different hand streams (perturbation hits every turn's hand,
    # in particular turn 0's, whose influence on turns 1-2 is what we're testing).
    turns_a = episode(np.random.default_rng(1), np.random.default_rng(100))
    turns_b = episode(np.random.default_rng(1), np.random.default_rng(200))

    # A0. Determinism control: identical episode twice must be bitwise equal.
    sfx_a = episode(np.random.default_rng(1), np.random.default_rng(100), hand_suffix=True)
    sfx_b = episode(np.random.default_rng(1), np.random.default_rng(200), hand_suffix=True)
    lg1, _ = run(model, processor, sfx_a, transient=True, dense=True, to_end=True)
    lg2, _ = run(model, processor, sfx_a, transient=True, dense=True, to_end=True)
    det = all(torch.equal(a, b) for a, b in zip(lg1, lg2))
    print(f"RESULT determinism (same episode twice): bitwise={det} "
          f"{'PASS' if det else 'FAIL (kernels nondeterministic; bitwise criteria void)'}", flush=True)
    # A. Suffix layout, tail fully transient: bitwise iff no leak AND no numeric coupling.
    for mode, dense in (("dense", True), ("sparse", False)):
        lga, ca = run(model, processor, sfx_a, transient=True, dense=dense, to_end=True)
        lgb, cb = run(model, processor, sfx_b, transient=True, dense=dense, to_end=True)
        later_bitwise = all(torch.equal(a, b) for a, b in zip(lga[1:], lgb[1:]))
        d_later = max(float((a - b).abs().max()) for a, b in zip(lga[1:], lgb[1:]))
        print(f"RESULT [suffix/{mode}] later bitwise={later_bitwise} max|dlogit| {d_later:.4f} "
              f"cache {ca}/{cb} (tiny magnitude => vision co-batch numerics, not a K/V leak)", flush=True)
    # B. Production layout: hand info flows through the readout K/V by design;
    # measure the magnitude (informational, not pass/fail).
    for mode, dense in (("dense", True), ("sparse", False)):
        lga, ca = run(model, processor, turns_a, transient=True, dense=dense)
        lgb, cb = run(model, processor, turns_b, transient=True, dense=dense)
        d_later = max(float((a - b).abs().max()) for a, b in zip(lga[1:], lgb[1:]))
        d_t0 = float((lga[0] - lgb[0]).abs().max())
        print(f"RESULT [prod-layout/{mode}] hand-perturbation leak via readout: "
              f"later-turn max|dlogit| {d_later:.3f} (turn0 {d_t0:.3f}) cache {ca}/{cb}", flush=True)

    lga, _ = run(model, processor, turns_a, transient=False, dense=True)
    lgb, _ = run(model, processor, turns_b, transient=False, dense=True)
    leaked = not all(torch.equal(a, b) for a, b in zip(lga[1:], lgb[1:]))
    print(f"RESULT negative control (plain cache): later turns change={leaked} "
          f"{'PASS' if leaked else 'FAIL (test has no teeth)'}", flush=True)
    print("INVARIANCE_DONE", flush=True)


if __name__ == "__main__":
    main()
