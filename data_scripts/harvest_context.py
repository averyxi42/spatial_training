"""Freeze a trunk: write `(pooled, context, target_chunk, pose)` for every observation.

The trunk of a trained policy is a function from (images, pose, conversation) to one
vector per turn. Running it is ~4 s/step of VLM; the vector it produces is 8 KB. So if
the question is about the **head**, computing that vector once and putting it on disk
turns a day-long experiment into a two-minute one. That is what this writes.

    python data_scripts/harvest_context.py \
        --ckpt dump/pose_injection/run_v9_flow_planar_pose_2p5hz/checkpoint-11400 \
        --dataset /Projects/data/v2_25hz_obs2.5hz/formatted_pose \
        --split train --out dump/head_only/ctx_v9_ck11400 --episodes 300

Two columns, not one, because they cut the model at two different places:

    pooled  (N, pooled_dim)  the frozen trunk state, `head.pooled_context(...)`. Training
                             on this trains the projection AND the flow -- the real
                             trainer's trainable set minus LoRA.
    context (N, context_dim) the source head's own output, `head.project(pooled)`.
                             Training on this trains only the flow, which is the
                             `dump/cnf_sigma_ablation` setting, and is also the value the
                             reattachment equivalence check compares against.

What this script is careful about, and why
------------------------------------------
**Resumability.** 39k episodes is not a thing to redo on a crash. Progress is per shard:
arrays are written under a temporary name and renamed, and the manifest entry -- which is
what makes an episode "done" -- is appended only afterwards. Re-running skips episodes
already in the manifest.

**Provenance.** A frozen-context dataset whose source trunk is unknown is worthless,
because the trunk is the independent variable of every experiment run on it. The manifest
records the checkpoint, its step, a content hash of its adapter + head weights, the
corpus, the split, the modality spec and the chunk shape. `reattach_head.py` refuses to
put a head back on a trunk whose hash does not match.

**The pose modality.** It has three separate wiring points and two of them fail silently:

    1. `modality_specs` on the `ModelConfig`  -- here it comes from the checkpoint's own
       `turn_vector_head_config.json`, so it cannot disagree with what was trained.
    2. `ModalityBatch.pop_from` + `pending()` around the backbone call -- inside
       `TurnVectorRegressor.encode_turns`, which is the same code the trainer runs.
    3. `modality_specs=` on the `TurnVectorCollator` -- taken off the model, not a flag.

All three are asserted before the first episode. And because a wiring check only proves
the plumbing is connected, `--verify-pose` additionally proves the water is flowing: it
re-runs one batch with the pose values perturbed and requires the context to move. A
harvest with the pose silently absent looks completely normal and is wrong.
"""

import argparse
import json
import os
import sys
import time
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
    ap.add_argument("--ckpt", required=True, help="source policy checkpoint (the trunk)")
    ap.add_argument("--dataset", required=True, help="a `load_from_disk` DatasetDict")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", required=True, help="context store directory to create/extend")
    ap.add_argument("--episodes", type=int, default=None,
                    help="cap on episodes harvested this invocation (default: all)")
    ap.add_argument("--episode-start", type=int, default=0)
    ap.add_argument("--episode-stop", type=int, default=None,
                    help="half-open row range of the split to harvest. Harvesting is one "
                         "forward per episode and embarrassingly parallel, but a single "
                         "manifest written by several processes would race -- so each "
                         "worker takes a disjoint range into its own --out and "
                         "ContextStore.open_many joins them")
    ap.add_argument("--max-turns", type=int, default=400,
                    help="turn window per conversation, matching the run's --max-turns. "
                         "Eval-mode windowing (always from turn 0), so a resumed harvest "
                         "sees the same turns as the first attempt")
    ap.add_argument("--target-column", default="action_chunks")
    ap.add_argument("--messages-column", default="messages")
    ap.add_argument("--images-column", default="images")
    ap.add_argument("--episode-id-column", default=None,
                    help="column holding a stable episode id; default: the row index, "
                         "which is stable for a fixed dataset and split")
    ap.add_argument("--shard-episodes", type=int, default=64,
                    help="episodes per shard; the unit of crash loss and of resume")
    ap.add_argument("--obs-hz", type=float, default=None,
                    help="recorded in the provenance only -- the corpus decides it, this "
                         "just makes the artifact self-describing")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    ap.add_argument("--verify-pose", dest="verify_pose", action="store_true", default=True)
    ap.add_argument("--no-verify-pose", dest="verify_pose", action="store_false",
                    help="skip the perturbation check (it costs one extra forward)")
    ap.add_argument("--notes", default="")
    return ap.parse_args()


def main():
    args = parse_args()

    from datasets import load_from_disk
    from transformers import AutoProcessor

    from longnav.utils.head_only import (
        ContextStore, HarvestProvenance, hash_checkpoint, load_trunk,
    )
    from longnav.utils.vector_sft import DataConfig, TurnVectorCollator

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    ds = load_from_disk(args.dataset)[args.split]
    processor = AutoProcessor.from_pretrained(args.ckpt)
    model = load_trunk(args.ckpt, processor, dtype=dtype, device=args.device)
    print(f"trunk: {args.ckpt}")
    print(f"  target_shape {model.target_shape}  pool {model.model_cfg.pool_mode}  "
          f"head_hidden_dims {model.model_cfg.head_hidden_dims}")
    print(f"  skipped the head-specific normalizer slot "
          f"({len(model.skipped_normalizer_keys)} tensors) -- the context is "
          f"head-independent by construction")

    # ---- the three modality wiring points, asserted before anything is written -------
    specs = model.model_cfg.modality_specs
    declared = [s.token for s in model.modality_embedder.specs]
    print(f"  modality specs on the ModelConfig: {declared or 'none'}")
    if specs and not model.modality_embedder:
        raise SystemExit(
            "the checkpoint declares modality specs but the embedder is empty -- wiring "
            "point 1 (modality_specs on the ModelConfig) is dead"
        )
    if model.modality_embedder and not model.modality_embedder.keys:
        raise SystemExit("the modality embedder has no keys; nothing could be injected")

    data_cfg = DataConfig(
        target_column=args.target_column,
        messages_column=args.messages_column,
        images_column=args.images_column,
        max_turns_per_sample=args.max_turns or None,
    )
    # Wiring point 3: taken off the model, never re-parsed from a flag, so the collator
    # cannot emit a marker the model does not expect or vice versa.
    collator = TurnVectorCollator(processor, data_cfg, train=False, seed=0,
                                  modality_specs=specs)
    if len(collator.modality_specs) != len(specs):
        raise SystemExit("the collator did not take the model's modality specs")

    chunk_shape = tuple(np.asarray(ds[0][args.target_column]).shape[1:])
    if len(chunk_shape) != 2:
        raise SystemExit(f"expected a (n_turns, T, C) target column, got chunk {chunk_shape}")
    print(f"corpus: {args.dataset}[{args.split}]  {len(ds)} rows  chunk {chunk_shape}")

    store = ContextStore(args.out)
    prov = HarvestProvenance(
        source_checkpoint=str(Path(args.ckpt).resolve()),
        source_step=_step_of(args.ckpt),
        source_hash=hash_checkpoint(args.ckpt),
        source_target_shape=list(model.target_shape),
        corpus=str(Path(args.dataset).resolve()),
        split=args.split,
        pooled_dim=int(model.head.pooled_dim),
        context_dim=int(model.target_shape[0]),
        chunk_shape=list(chunk_shape),
        modality_specs=[_spec_json(s) for s in specs],
        pool_mode=model.model_cfg.pool_mode,
        head_hidden_dims=list(model.model_cfg.head_hidden_dims),
        max_turns_per_sample=args.max_turns or None,
        target_column=args.target_column,
        obs_hz=args.obs_hz,
        notes=args.notes,
    )
    store.init(prov)
    done = store.done_episodes()
    print(f"store: {args.out}  ({len(done)} episode(s) already harvested)")

    ep_ids = ([str(v) for v in ds[args.episode_id_column]]
              if args.episode_id_column else [str(i) for i in range(len(ds))])
    lo, hi = args.episode_start, (len(ds) if args.episode_stop is None else args.episode_stop)
    todo = [i for i in range(lo, min(hi, len(ds))) if ep_ids[i] not in done]
    if args.episodes is not None:
        todo = todo[: args.episodes]
    if not todo:
        print("nothing to do")
        return
    print(f"harvesting {len(todo)} episode(s)")

    pose_key = next((s.key for s in specs), None)
    buf = {k: [] for k in ("pooled", "context", "targets")}
    if pose_key is not None:
        buf["pose"] = []
    shard_eps, shard_rows = [], []
    t0, n_rows, verified = time.time(), 0, not (args.verify_pose and pose_key)

    def flush():
        nonlocal shard_eps, shard_rows
        if not shard_eps:
            return
        arrays = {k: np.concatenate(v).astype(np.float32) for k, v in buf.items() if v}
        store.write_shard(arrays, shard_eps, shard_rows)
        for v in buf.values():
            v.clear()
        shard_eps, shard_rows = [], []

    for n, i in enumerate(todo):
        batch = collator([ds[int(i)]])
        targets = batch.pop("targets")
        batch.pop("num_turns", None)
        batch = {k: (v.to(args.device) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        with torch.no_grad():
            enc = model.encode_turns(**batch)
            pooled = enc.pooled.float()
            context = model.head.project(enc.pooled).float()
        if pooled.shape[0] != targets.shape[0]:
            raise RuntimeError(
                f"episode {ep_ids[i]}: {pooled.shape[0]} turn(s) but "
                f"{targets.shape[0]} target(s) -- the alignment gate the trainer applies"
            )

        if not verified:
            _verify_pose_is_live(model, batch, pose_key, context)
            verified = True

        buf["pooled"].append(pooled.cpu().numpy())
        buf["context"].append(context.cpu().numpy())
        buf["targets"].append(targets.numpy())
        if pose_key is not None:
            buf["pose"].append(
                batch[f"modality_{pose_key}_values"].float().cpu().numpy())
        shard_eps.append(ep_ids[i])
        shard_rows.append(int(pooled.shape[0]))
        n_rows += int(pooled.shape[0])

        if len(shard_eps) >= args.shard_episodes:
            flush()
            el = time.time() - t0
            print(f"  {n + 1}/{len(todo)} episodes, {n_rows} rows, "
                  f"{el:.0f}s ({n_rows / max(el, 1e-9):.1f} rows/s)", flush=True)
    flush()

    el = time.time() - t0
    size = sum(p.stat().st_size for p in Path(args.out).glob("*.npy"))
    print(f"\n-> {args.out}")
    print(f"   {n_rows} rows over {len(todo)} episode(s) in {el:.0f}s "
          f"({n_rows / max(el, 1e-9):.1f} rows/s, {el / max(1, len(todo)):.2f} s/episode)")
    print(f"   {size / 2**30:.3f} GiB on disk "
          f"({size / max(1, n_rows):.0f} bytes/row)")
    print(f"   provenance sha256 {prov.source_hash[:16]}...")


def _verify_pose_is_live(model, batch, pose_key, context):
    """Prove the injected pose reaches the context, rather than that it was passed in.

    Re-runs the same batch with the pose values shifted by a metre and requires the
    context to move by more than float noise. A wiring check cannot catch an encoder
    whose output is discarded downstream, or a marker that the chat template mangled into
    a position the embedder never sees; this can. It costs one forward, once per harvest.
    """
    key = f"modality_{pose_key}_values"
    perturbed = dict(batch)
    perturbed[key] = batch[key] + 1.0
    with torch.no_grad():
        alt = model.head.project(model.encode_turns(**perturbed).pooled).float()
    delta = float((alt - context).abs().max())
    scale = float(context.abs().max())
    if delta <= 1e-4 * max(scale, 1e-6):
        raise SystemExit(
            f"the {pose_key!r} modality is wired but INERT: shifting every pose by 1.0 "
            f"moved the context by {delta:.3e} against a context scale of {scale:.3e}. "
            "The harvest would record a context conditioned on an input the model was "
            "trained with and is not receiving. Refusing to write it."
        )
    print(f"  pose is live: a +1.0 shift moves the context by {delta:.4g} "
          f"(context scale {scale:.4g})")


def _spec_json(spec):
    from dataclasses import asdict, is_dataclass

    return asdict(spec) if is_dataclass(spec) else json.loads(json.dumps(spec, default=str))


def _step_of(ckpt):
    name = Path(ckpt).name
    return int(name.split("-")[-1]) if name.startswith("checkpoint-") else None


if __name__ == "__main__":
    main()
