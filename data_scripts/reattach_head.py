"""Put a head trained on frozen context back on the trunk the context came from.

Without this, head-only training answers an interesting question and produces nothing
runnable. With it the loop closes: harvest the good trunk -> train the head in minutes ->
reattach -> evaluate as an ordinary policy.

    python data_scripts/reattach_head.py \
        --head-run dump/head_only/run_cnf_a/final \
        --store    dump/head_only/ctx_v9_ck11400 \
        --out      dump/head_only/reattached_cnf_a

The output is byte-compatible with the ordinary loading path -- `adapter/` (copied
verbatim from the source checkpoint: the trunk is unchanged, that is the point),
`turn_vector_head.pt`, `turn_vector_head_config.json`, `flow_config.json` and the
processor files -- so anything that loads a `train_flow_sft.py` checkpoint loads this one
with no special case.

Three things this refuses to do quietly
---------------------------------------
**Reattach to the wrong trunk.** The store records a content hash of the source
checkpoint's adapter and head weights; it is checked, and a mismatch names both sides and
stops. A head trained on trunk A bolted onto trunk B produces a policy that loads, runs,
and is meaningless -- there is no observable that would go wrong, which is exactly why
the check has to be here.

**Write a config that disagrees with the weights.** The head config is rebuilt from the
head that was actually trained -- flow layers/hidden/depth, context dim, chunk shape,
noise floor -- not copied from the source checkpoint, whose head may be a completely
different type. A config/weight mismatch surfaces as garbage actions rather than an
error.

**Assert instead of verify.** After assembly, the model's own forward is run on real
turns from the harvest corpus and the pooled trunk state it computes is compared against
what the harvest recorded for those same rows. That single check proves the trunk is
unchanged, that the head is wired into the right slot, and that the harvest was faithful;
if it fails, every conclusion drawn from the frozen-context runs is void. So it is a hard
gate, not a warning, and `--no-verify` exists only for a machine with no corpus on it.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--head-run", required=True,
                    help="a train_head_only.py checkpoint (e.g. .../final)")
    ap.add_argument("--store", default=None, nargs="+",
                    help="the context store(s); default: the one(s) named in the head "
                         "run's head_only_run.json")
    ap.add_argument("--source-ckpt", default=None,
                    help="override the trunk; default: the store's source checkpoint. "
                         "The hash check still applies")
    ap.add_argument("--out", required=True)
    ap.add_argument("--verify-rows", type=int, default=64,
                    help="turns compared against the harvest. 0 disables (see --no-verify)")
    ap.add_argument("--verify-tol", type=float, default=2e-2,
                    help="max abs difference allowed on the pooled trunk state. The "
                         "default is loose because the trunk runs in bf16, whose epsilon "
                         "is ~8e-3 on values of order 1; it is still ~50x tighter than "
                         "any real wiring error")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the equivalence gate. Only for a machine with no corpus")
    ap.add_argument("--device", default="cuda")
    return ap.parse_args()


def main():
    args = parse_args()

    from transformers import AutoProcessor

    from longnav.utils.flow_head import FLOW_CONFIG_FILE, TurnFlowPolicy
    from longnav.utils.head_only import (
        ContextStore, FrozenContextFlowPolicy, hash_checkpoint,
    )
    from longnav.utils.vector_sft import (
        ADAPTER_SUBDIR, HEAD_CONFIG_FILE, HEAD_WEIGHTS_FILE, LossConfig, ModelConfig,
        migrate_model_config,
    )

    head_run = Path(args.head_run)
    run_meta_path = _find_run_meta(head_run)
    run_meta = json.loads(run_meta_path.read_text()) if run_meta_path else {}
    store_path = args.store or run_meta.get("args", {}).get("store")
    if isinstance(store_path, str):
        store_path = [store_path]
    if not store_path:
        raise SystemExit("--store is required (the head run carries no head_only_run.json)")
    store = ContextStore.open_many(store_path)
    prov = store.provenance
    source = Path(args.source_ckpt or prov.source_checkpoint)

    print(f"head run : {head_run}")
    print(f"store    : {' + '.join(str(s) for s in store_path)}")
    print(f"trunk    : {source}  (step {prov.source_step})")

    # ---- gate 1: is this the trunk the context came from? ---------------------------
    prov.assert_matches_checkpoint(source)
    print(f"  hash ok : {prov.source_hash[:16]}... matches")

    head = FrozenContextFlowPolicy.from_pretrained(head_run)
    if head.identity_head:
        raise SystemExit(
            "this head was trained with --context-key context, i.e. on the source head's "
            "own output with the projection frozen. Reattaching it would need the source "
            "head as well as this flow, and the result would not be a single "
            "TurnVectorHead -- retrain with --context-key pooled to get a reattachable "
            "head."
        )
    if head.pooled_dim != prov.pooled_dim:
        raise SystemExit(f"head expects pooled_dim {head.pooled_dim}, store has "
                         f"{prov.pooled_dim}")

    # ---- build the policy: source trunk config, this head's shape --------------------
    src_meta = json.loads((source / HEAD_CONFIG_FILE).read_text())
    model_cfg = ModelConfig(**migrate_model_config(src_meta["model"]))
    processor = AutoProcessor.from_pretrained(source)
    model = TurnFlowPolicy.build(
        model_cfg,
        # The loss the reattached checkpoint declares is the one it was trained under,
        # not the source's -- the source may be a flow-matching or regression head.
        LossConfig(kind="flow_nll", normalize_targets=False),
        lora=None,
        target_shape=(head.context_dim,),
        processor=processor,
        dtype=torch.bfloat16,
        chunk_shape=head.chunk_shape,
        flow_kwargs=_flow_kwargs(head),
        decode=head.decoder.decode,
        decode_k=head.decoder.k,
    )
    model.train_content_len = src_meta.get("train_content_len")

    from peft import PeftModel

    if not (source / ADAPTER_SUBDIR).exists():
        raise SystemExit(f"{source} has no {ADAPTER_SUBDIR}/ -- there is no trunk to reattach to")
    model.backbone = PeftModel.from_pretrained(model.backbone, str(source / ADAPTER_SUBDIR))
    model.attach_modality_hooks()

    # Trunk-side weights come from the SOURCE (the modality encoders are part of the
    # frozen trunk and were never trained head-only); head-side weights come from the
    # head run. Getting this backwards would silently re-randomise the pose encoder.
    src_blob = torch.load(source / HEAD_WEIGHTS_FILE, map_location="cpu", weights_only=False)
    model.modality_embedder.load_state_blob(src_blob.get("modality"))
    model.head.load_state_dict(head.head.state_dict(), strict=True)
    model.normalizer.load_state_dict(head.normalizer.state_dict(), strict=True)
    print(f"  assembled: trunk from {source.name}, head+flow from {head_run.name}, "
          f"modality encoders {[s.token for s in model.modality_embedder.specs] or 'none'}")

    model.to(args.device).eval()

    # ---- gate 2: does it compute the context that was harvested? ---------------------
    if not args.no_verify and args.verify_rows > 0:
        _verify_equivalence(model, processor, store, prov, args)

    # ---- write ----------------------------------------------------------------------
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # `save_pretrained` writes the adapter from the live PeftModel, which is the source's
    # weights loaded above -- but copying the directory is what the requirement asks for
    # and is stronger: it preserves the source's own adapter_config.json byte for byte,
    # so nothing about the trunk can drift through a round trip.
    if (out / ADAPTER_SUBDIR).exists():
        shutil.rmtree(out / ADAPTER_SUBDIR)
    shutil.copytree(source / ADAPTER_SUBDIR, out / ADAPTER_SUBDIR)
    blob = {"head": model.head.state_dict(), "normalizer": model.normalizer.state_dict()}
    modality = model.modality_embedder.state_blob()
    if modality is not None:
        blob["modality"] = modality
    torch.save(blob, out / HEAD_WEIGHTS_FILE)
    # The config is rebuilt to describe the head that exists, via the same writer the
    # trainer uses -- so the flow's architecture in flow_config.json cannot disagree with
    # the tensors in turn_vector_head.pt.
    meta = json.loads((source / HEAD_CONFIG_FILE).read_text())
    meta["loss"] = {"kind": "flow_nll", "huber_beta": 1.0, "normalize_targets": False,
                    "normalizer_fit_rows": 512}
    meta["target_shape"] = [head.context_dim]
    meta["model"] = {**meta["model"],
                     "head_hidden_dims": _head_hidden_dims(model.head)}
    for stale in ("fm_config", "fm_decoder_kwargs", "fm_context_dim", "fm_n_ticks",
                  "flow_head_version"):
        meta.pop(stale, None)      # the source head's shape, which this is not
    (out / HEAD_CONFIG_FILE).write_text(json.dumps(meta, indent=2))
    (out / FLOW_CONFIG_FILE).write_text(json.dumps({
        "flow": model.flow.config(),
        "chunk_shape": list(model.chunk_shape),
        "decode": model.decoder.decode,
        "decode_k": model.decoder.k,
    }, indent=2))
    processor.save_pretrained(out)
    (out / "reattach.json").write_text(json.dumps({
        "head_run": str(head_run.resolve()),
        "store": [str(Path(s).resolve()) for s in store_path],
        "source_checkpoint": str(source.resolve()),
        "source_hash": prov.source_hash,
        "reattached_hash": None,
        "verified": not (args.no_verify or args.verify_rows == 0),
    }, indent=2))
    # Written after everything else, so it hashes the finished directory.
    blob = json.loads((out / "reattach.json").read_text())
    blob["reattached_hash"] = hash_checkpoint(out)
    (out / "reattach.json").write_text(json.dumps(blob, indent=2))

    print(f"\n-> {out}")
    print(f"   {sorted(p.name for p in out.iterdir())}")
    _report_loadability(out)


def _verify_equivalence(model, processor, store, prov, args):
    """Recompute the harvest on real turns and require it to match. Hard gate.

    Compares the **pooled** trunk state, not the context: the head has been replaced by
    construction, so its output is expected to differ, whereas the pooled state is
    exactly the quantity the store froze. Pooled equality is also the stronger statement
    -- any head is a deterministic function of it -- so if pooled matches, the harvest
    was faithful and the trunk is unchanged.
    """
    from datasets import load_from_disk

    from longnav.utils.vector_sft import DataConfig, TurnVectorCollator

    ds = load_from_disk(prov.corpus)[prov.split]
    shard0 = store.shards[0]
    ep_ids, ep_rows = shard0["episodes"], shard0["episode_rows"]
    data_cfg = DataConfig(target_column=prov.target_column,
                          max_turns_per_sample=prov.max_turns_per_sample)
    collator = TurnVectorCollator(processor, data_cfg, train=False, seed=0,
                                  modality_specs=model.model_cfg.modality_specs)

    checked, worst, row0 = 0, 0.0, 0
    for ep, n_rows in zip(ep_ids, ep_rows):
        if checked >= args.verify_rows:
            break
        batch = collator([ds[int(ep)]])
        batch.pop("targets"), batch.pop("num_turns", None)
        batch = {k: (v.to(args.device) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        with torch.no_grad():
            got = model.encode_turns(**batch).pooled.float().cpu().numpy()
        want = store.get("pooled", np.arange(row0, row0 + n_rows))
        take = min(len(got), len(want), args.verify_rows - checked)
        worst = max(worst, float(np.abs(got[:take] - want[:take]).max()))
        checked += take
        row0 += n_rows

    scale = float(np.abs(want).mean())
    print(f"  equivalence: {checked} turn(s), max |recomputed - harvested| = {worst:.3e} "
          f"(pooled |mean| {scale:.3f}, tol {args.verify_tol:g})")
    if worst > args.verify_tol:
        raise SystemExit(
            f"EQUIVALENCE GATE FAILED: the reassembled trunk computes a pooled state "
            f"differing from the harvest by up to {worst:.3e} (> {args.verify_tol:g}).\n"
            "One of these is true and all of them are fatal: the adapter is not the one "
            "the context was harvested with, the modality encoders are not loaded, the "
            "pooling/affix configuration differs, or the harvest was written from a "
            "different code path. Every conclusion drawn from this store's head-only "
            "runs would be void, so nothing is written."
        )


def _report_loadability(out: Path):
    """Answer the open question: can the ObjectNav harness run a reattached CNF?

    `flow_head`'s docstring claims `VectorRolloutPolicy.step` works untouched because
    `FlowActionDecoder` occupies the `normalizer` slot. The claim was never tested, and
    the answer decides whether `objectnav_eval` needs a fourth backend. Two paths are
    tried, because they fail for different reasons and only one of them is a real
    obstacle:

      1. `VectorRolloutPolicy.from_checkpoint` -- the generic path. It goes through
         `TurnVectorRegressor.from_pretrained`, which puts a `TargetNormalizer` in the
         normalizer slot and then loads the flow's tensors into it strictly.
      2. `VectorRolloutPolicy(TurnFlowPolicy.from_pretrained(...), ...)` -- the same
         rollout policy, constructed from a model that knows it is a flow.

    If 2 works and 1 does not, the docstring's claim is right about `step` and wrong
    about loading, and a backend is a few lines of construction rather than a new
    rollout implementation.
    """
    from longnav.utils.flow_head import TurnFlowPolicy
    from longnav.utils.vector_rollout import RolloutConfig, VectorRolloutPolicy

    cfg = RolloutConfig(device="cpu", merge_lora=False)
    try:
        VectorRolloutPolicy.from_checkpoint(out, cfg)
        print("   VectorRolloutPolicy.from_checkpoint    : LOADS")
    except Exception as exc:      # noqa: BLE001 -- the point is to report, not to handle
        head = str(exc).split("\n")[0][:160]
        print(f"   VectorRolloutPolicy.from_checkpoint    : FAILS ({type(exc).__name__}: {head})")

    try:
        from transformers import AutoProcessor

        proc = AutoProcessor.from_pretrained(out)
        model = TurnFlowPolicy.from_pretrained(out, proc, dtype=torch.float32, device="cpu")
        policy = VectorRolloutPolicy(model, proc, cfg)
        chunk = policy.model.normalizer.denormalize(
            torch.zeros(1, model.target_shape[0]))
        print(f"   VectorRolloutPolicy(TurnFlowPolicy...) : LOADS and decodes "
              f"{tuple(chunk.shape)} from the normalizer slot")
    except Exception as exc:      # noqa: BLE001
        head = str(exc).split("\n")[0][:160]
        print(f"   VectorRolloutPolicy(TurnFlowPolicy...) : FAILS ({type(exc).__name__}: {head})")


def _flow_kwargs(head):
    cfg = dict(head.flow.config())
    for k in ("context_dim", "chunk_len", "n_channels"):
        cfg.pop(k, None)
    return cfg


def _head_hidden_dims(head):
    import torch.nn as nn

    return [m.out_features for m in head.mlp if isinstance(m, nn.Linear)][:-1]


def _find_run_meta(head_run: Path):
    for cand in (head_run / "head_only_run.json", head_run.parent / "head_only_run.json"):
        if cand.exists():
            return cand
    return None


if __name__ == "__main__":
    main()
