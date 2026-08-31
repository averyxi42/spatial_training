"""PROTOTYPE parity test for the transient hand-token KV mechanism (GPU, standalone).

    CUDA_VISIBLE_DEVICES=<gpu> env -u DISPLAY \
      /workspace/conda/envs/longnav_vlm/bin/python tests/proto/parity_transient_kv.py

Arms (real sparse checkpoint, real processor, sdpa so the reference can take a 4D mask):
  noise floor : incremental plain DynamicCache, natural positions   vs  full forward.
  transient   : incremental TransientFilteredCache + stitched RoPE  vs  full forward
                with the equivalent 4D mask + identical positions.
The transient arm passes if its per-turn readout-logit deviation is comparable to the
noise floor (chunked-vs-full bf16 attention accumulation differs regardless of us).
Then a sparse-mode smoke: mechanism runs under the sparsifying path, cache stays
persistent-only, similarity DB excludes hand embeds. No production code modified.
"""
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/Projects/spatial_training_mshab/src")

MODEL_ID = "Phyllis1/qwen3_sft_sft_sparse_03drop_single_action_20260103_210803_ckpt10800"
T_TURNS = 3
DEV = "cuda"


def build_messages(t, rng):
    """Turn t of a mobile-manip-style conversation: head image + small hand image."""
    head = Image.fromarray(rng.integers(0, 255, (256, 256, 3), dtype=np.uint8))
    hand = Image.fromarray(rng.integers(0, 255, (128, 128, 3), dtype=np.uint8))
    msgs = [
        {"role": "user", "content": [
            {"type": "text", "text": f"Observation {t}:"},
            {"type": "image"},
            {"type": "text", "text": "Hand:"},
            {"type": "image"},
            {"type": "text", "text": "Action:"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "**____**"}]},
    ]
    return msgs, [head, hand]


def tokenize(processor, messages, images):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return processor(text=text, images=images, videos=None, padding=False, return_tensors="pt")


def strip_hand(processor, turn_inputs, span, hand_grid_idx):
    """The same chunk with the hand image's tokens and grid removed (for positions)."""
    s, e = span
    ids = torch.cat([turn_inputs["input_ids"][:, :s], turn_inputs["input_ids"][:, e:]], dim=1)
    grids = turn_inputs["image_grid_thw"]
    keep = [i for i in range(grids.shape[0]) if i != hand_grid_idx]
    return {"input_ids": ids, "image_grid_thw": grids[keep], "attention_mask": torch.ones_like(ids)}


def run_incremental(model, processor, turns, transient: bool, dense: bool, transient_to_end: bool = False):
    from transformers import DynamicCache
    from longnav.proto.transient_kv import (TransientFilteredCache, image_token_spans,
                                            stitched_positions)
    lm = model.model.language_model
    image_token_id = model.config.image_token_id
    cache = TransientFilteredCache() if transient else DynamicCache()
    offset = 0
    prev_len = 0
    readout_logits, all_ids, all_pos, all_trans = [], [], [], []
    pre_hand_logits = []
    past_db = None
    for (msgs_all, imgs_all, n_imgs_prev) in turns:
        full = tokenize(processor, msgs_all, imgs_all)
        assert prev_len == 0 or torch.equal(full["input_ids"][:, :prev_len], all_ids_cat[:, :prev_len]), \
            "chat template not prefix-stable"
        chunk = {"input_ids": full["input_ids"][:, prev_len:]}
        chunk["attention_mask"] = torch.ones_like(chunk["input_ids"])
        # this turn's images = the last two grids / pixel slices
        grids = full["image_grid_thw"][n_imgs_prev:]
        n_pix_prev = int((full["image_grid_thw"][:n_imgs_prev].prod(-1)).sum()) if n_imgs_prev else 0
        chunk["pixel_values"] = full["pixel_values"][n_pix_prev:]
        chunk["image_grid_thw"] = grids
        spans = image_token_spans(chunk["input_ids"], image_token_id)
        assert len(spans) == 2, spans
        hand_span = spans[1]
        tmask = torch.zeros(chunk["input_ids"].shape[1], dtype=torch.bool)
        if transient_to_end:
            tmask[hand_span[0]:] = True     # hand image + trailing template tokens
        else:
            tmask[hand_span[0]:hand_span[1]] = True
        if transient:
            pos, offset = stitched_positions(model.model, chunk,
                                             strip_hand(processor, chunk, hand_span, 1),
                                             tmask, offset)
            lm._pending_transient = tmask
        else:
            pos, delta = model.model.get_rope_index(
                input_ids=chunk["input_ids"], image_grid_thw=chunk["image_grid_thw"],
                video_grid_thw=None, attention_mask=chunk["attention_mask"])
            pos = pos + offset
            offset += chunk["input_ids"].shape[1] + int(delta.item())
        fw = {k: v.to(DEV) for k, v in chunk.items()}
        fw["attention_mask"] = None
        fw["position_ids"] = pos.to(DEV)
        if dense:
            fw["seq_keep_mask"] = True
            fw["vis_keep_mask"] = True
        else:
            fw["past_image_embeds"] = past_db
            fw["save_image_db"] = True
        with torch.inference_mode():
            out = model.forward(**fw, past_key_values=cache, use_cache=True,
                                logits_to_keep=0 if transient_to_end else 1)
        cache = out["past_key_values"]
        if transient_to_end:
            # readout position = last token BEFORE the transient tail (the tail's
            # trailing template tokens attend the hand by construction and are
            # expected to differ; the readout must not). In sparse mode the logits
            # are over the POST-sparsification sequence: map the index through the
            # keep mask (pre-hand keeps depend only on the head image + text, so the
            # mapped index is identical across perturbation runs).
            km = out.get("seq_keep_mask", None)
            idx = int(km[:hand_span[0]].sum()) - 1 if km is not None else hand_span[0] - 1
            pre_hand_logits.append(out["logits"][0, idx].float().cpu())
        if not dense and getattr(lm, "kept_visual_embeds", None):
            new_db = lm.kept_visual_embeds[0]
            past_db = [new_db if past_db is None else torch.cat([past_db[0], new_db], 0)]
        readout_logits.append(out["logits"][0, -1].float().cpu())
        all_ids.append(chunk["input_ids"]); all_pos.append(pos); all_trans.append(tmask)
        all_ids_cat = full["input_ids"]
        prev_len = full["input_ids"].shape[1]
    if transient_to_end:
        return pre_hand_logits, torch.cat(all_ids, 1), torch.cat(all_pos, -1), torch.cat(all_trans), cache
    return readout_logits, torch.cat(all_ids, 1), torch.cat(all_pos, -1), torch.cat(all_trans), cache


def run_full(model, processor, turns, ids, pos, trans, masked: bool, dense: bool):
    """One no-cache forward over the identical token stream; 4D mask blinds future
    tokens to past transient blocks when masked=True."""
    msgs_all, imgs_all, _ = turns[-1]
    full = tokenize(processor, msgs_all, imgs_all)
    assert torch.equal(full["input_ids"], ids), "token stream mismatch"
    S = ids.shape[1]
    turn_of = torch.zeros(S, dtype=torch.long)
    # reconstruct turn boundaries from per-turn lengths
    lens = []
    prev = 0
    for (m, im, _) in turns:
        L = tokenize(processor, m, im)["input_ids"].shape[1]
        lens.append(L - prev); prev = L
    c = 0
    for i, L in enumerate(lens):
        turn_of[c:c + L] = i; c += L
    fw = {"input_ids": full["input_ids"].to(DEV), "pixel_values": full["pixel_values"].to(DEV),
          "image_grid_thw": full["image_grid_thw"].to(DEV)}
    if pos is not None:
        fw["position_ids"] = pos.to(DEV)
    if masked:
        causal = torch.tril(torch.ones(S, S, dtype=torch.bool))
        if trans is not None:
            blind = trans[None, :] & (turn_of[:, None] > turn_of[None, :])
        else:
            blind = torch.zeros(S, S, dtype=torch.bool)
        allow = causal & ~blind
        bias = torch.zeros(1, 1, S, S, dtype=torch.bfloat16)
        bias.masked_fill_(~allow[None, None], torch.finfo(torch.bfloat16).min)
        fw["attention_mask"] = bias.to(DEV)
    if dense:
        fw["seq_keep_mask"] = True
        fw["vis_keep_mask"] = True
    with torch.inference_mode():
        out = model.forward(**fw, use_cache=False, logits_to_keep=0)
    ends = torch.cumsum(torch.tensor(lens), 0) - 1
    return [out["logits"][0, e].float().cpu() for e in ends]


def main():
    from longnav.utils.modeling import Qwen3VLSparseForConditionalGeneration
    from longnav.proto.transient_kv import install_transient
    from transformers import AutoConfig, AutoProcessor
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = Qwen3VLSparseForConditionalGeneration.from_pretrained(
        MODEL_ID, config=cfg, torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="sdpa").to(DEV).eval()
    install_transient(model)
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    msgs, imgs = [], []
    turns = []
    for t in range(T_TURNS):
        m, im = build_messages(t, rng)
        msgs = msgs + m
        imgs = imgs + im
        turns.append((list(msgs), list(imgs), 2 * t))

    print("== DENSE mode ==", flush=True)
    lg_plain, ids_p, pos_p, tr_p, cache_p = run_incremental(model, processor, turns, transient=False, dense=True)
    # pure-causal 4D mask: same sdpa additive-mask kernel path as the cached side
    # (is_causal fast path vs additive path differ ~5.9 bf16: dump/probe_prefill_ab.py)
    lg_fullp = run_full(model, processor, turns, ids_p, pos_p, None, masked=True, dense=True)
    diffs = [float((a - b).abs().max()) for a, b in zip(lg_plain, lg_fullp)]
    agree = [int(a.argmax() == b.argmax()) for a, b in zip(lg_plain, lg_fullp)]
    floor = max(diffs)
    print(f"RESULT noise floor per-turn max|dlogit| {['%.3f' % d for d in diffs]} argmax-agree {agree}", flush=True)
    lg_tr, ids_t, pos_t, tr_t, cache_t = run_incremental(model, processor, turns, transient=True, dense=True)
    lg_fullm = run_full(model, processor, turns, ids_t, pos_t, tr_t, masked=True, dense=True)
    tdiffs = [float((a - b).abs().max()) for a, b in zip(lg_tr, lg_fullm)]
    print(f"RESULT transient per-turn max|dlogit| {['%.3f' % d for d in tdiffs]}", flush=True)
    dev = max(tdiffs)
    hand_total = int(tr_t.sum())
    print(f"RESULT transient arm: max|dlogit| {dev:.4f}  cache {cache_t.get_seq_length()} "
          f"(= {ids_t.shape[1]} tokens - {hand_total} hand)  "
          f"{'PASS' if dev <= max(3 * floor, 0.05) else 'FAIL'}", flush=True)
    assert cache_t.get_seq_length() == ids_t.shape[1] - hand_total, "cache leaked hand tokens"

    print("== SPARSE mode smoke ==", flush=True)
    lg_s, ids_s, pos_s, tr_s, cache_s = run_incremental(model, processor, turns, transient=True, dense=False)
    print(f"RESULT sparse smoke: ran {len(lg_s)} turns, cache {cache_s.get_seq_length()} "
          f"<= dense persistent {ids_s.shape[1] - int(tr_s.sum())}", flush=True)
    print("PARITY_DONE", flush=True)


if __name__ == "__main__":
    main()
