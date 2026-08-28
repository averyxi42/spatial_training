#!/usr/bin/env python3
"""Does `c` still steer the decode in the FULL model? The gate before any RL is built.

`CODE_RL_PLAN.md` section 9: RL over `c` optimises a lever that must actually be connected.
The 94.0%-within-one-step obedience figure on record is from the **c-only prototype** -- a
head with no backbone and no competing flow-matching objective. This measures the same thing
on a real checkpoint, where `r(h)` is live and has been growing (`ctx_res_over_code` rose
0.075 -> 0.561 over 1200 steps). If obedience has collapsed, the RL plan is void.

Reported BY MISS DISTANCE, never as strict equality (`CODE_CONDITIONED_POLICY.md` section 6):
FSQ indices are ordinal, so an adjacent code is a neighbouring trajectory mode rather than a
different behaviour, and counting a boundary landing as a total failure reads far worse than
the truth. Headline is the within-one-step rate; >= 2 steps is the real failure rate.

Three questions, separated because they have different consequences:

  TEACHER  obey at c* = g(A*), the dataset's own code. Isolates the DECODER's obedience
           from the policy head's accuracy, and is the number comparable to the prototype's.
  POLICY   obey at c ~ p(c|h), what RL would actually condition on. Can differ from TEACHER
           if the head prefers codes the decoder renders badly.
  Z0 SPREAD across m draws at fixed (h, c): do they land in the SAME cell? This is the
           "is z_0's influence sub-cell" question (section 3.2) measured directly -- if z_0
           moves the sample across cells, chain noise randomises the mode RL just chose.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "src"), str(_ROOT / "data_scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


@torch.no_grad()
def collect_h(model, ds, rows, collator, device):
    """(h, chunks, valid) per turn. One backbone pass per row -- the expensive part."""
    from longnav.utils.modality_embed import ModalityBatch
    from longnav.utils.turn_vectors import extract_turn_vectors

    H, A, V = [], [], []
    for n, r in enumerate(rows):
        row = ds[int(r)]
        chunks = np.asarray(row["action_chunks"], dtype=np.float32)
        if chunks.ndim != 3:
            continue
        batch = collator([row])
        skip = {"targets", "labels", "num_turns", "stop_targets"}
        inputs = {k: (v.to(device) if torch.is_tensor(v) else v)
                  for k, v in batch.items() if k not in skip}
        modality = {k: inputs.pop(k) for k in list(inputs) if k.startswith("modality_")}
        mb = ModalityBatch.pop_from({**modality}, known_keys=model.modality_embedder.keys)
        with model.modality_embedder.pending(mb):
            # logits_to_keep=1: without it the LM head materialises full-vocab logits at
            # every position, which is gigabytes on a long turn sequence and is not the
            # readout we want.
            out = model.backbone(use_cache=False, logits_to_keep=1, **inputs)
        h, _ = extract_turn_vectors(
            out, inputs["input_ids"], model.head, prefix_ids=model.prefix_ids,
            postfix_ids=model.postfix_ids, shift_left=model.model_cfg.shift_left,
            strict=True)
        t = min(len(h), len(chunks))
        a = torch.from_numpy(chunks[:t])
        H.append(h[:t].float().cpu())
        A.append(a)
        # Probe rows carry all-NaN targets; encode_chunk would poison the FSQ bound with
        # NaN and silently return grid index 0, so they are masked out here.
        V.append(~torch.isnan(a).any(dim=2).any(dim=1))
        if n % 20 == 0:
            print(f"  {n}/{len(rows)} rows, {sum(len(x) for x in H)} turns", flush=True)
    return torch.cat(H), torch.cat(A), torch.cat(V)


def lattice_miss(a, b, levels):
    """L-inf distance on the FSQ lattice between two FLAT code indices.

    NOT `|a - b|`. FSQ is `d` dims of `L_i` levels with basis `[1, L_0, L_0*L_1, ...]`, so
    the flat index is mixed-radix: at levels [8, 5] the index is `d0 + 8*d1`. Then

      * a difference of **8** is ONE step in dim 1 -- an adjacent cell, which |a-b| would
        report as a gross 8-step miss (this showed up as a hard spike at exactly 8 in the
        first run's histogram, which is what exposed the bug);
      * a difference of **1** across a row boundary (7 -> 8) is a wrap: dim 0 moves 7 steps
        and dim 1 moves 1, which |a-b| would report as a perfect near miss.

    Ordinality is per-dimension, so the neighbour relation is on the lattice. L-inf (any of
    the surrounding cells counts as one step) rather than L1, since a diagonal neighbour is
    no further away in trajectory space than an edge one.
    """
    d = np.zeros(len(a), dtype=np.int64)
    ra, rb = a.copy(), b.copy()
    for L in levels:
        d = np.maximum(d, np.abs((ra % L) - (rb % L)))
        ra, rb = ra // L, rb // L
    return d


def report(name, miss_xy, miss_th, out):
    """Miss-distance summary for one condition. Within-one-step is the headline."""
    n = len(miss_xy)
    both0 = ((miss_xy == 0) & (miss_th == 0)).mean()
    both1 = ((miss_xy <= 1) & (miss_th <= 1)).mean()
    d = {
        "n": int(n),
        "strict_xy": float((miss_xy == 0).mean()),
        "strict_theta": float((miss_th == 0).mean()),
        "strict_joint": float(both0),
        "within1_xy": float((miss_xy <= 1).mean()),
        "within1_theta": float((miss_th <= 1).mean()),
        "within1_joint": float(both1),
        "gross_xy": float((miss_xy >= 2).mean()),
        "gross_theta": float((miss_th >= 2).mean()),
        "mean_miss_xy": float(miss_xy.mean()),
        "mean_miss_theta": float(miss_th.mean()),
        "hist_xy": {str(k): int(v) for k, v in sorted(Counter(miss_xy.tolist()).items())[:8]},
        "hist_theta": {str(k): int(v) for k, v in sorted(Counter(miss_th.tolist()).items())[:8]},
    }
    out[name] = d
    print(f"\n--- {name}  (n = {n})")
    print(f"  strict    xy {d['strict_xy']:.3f}  theta {d['strict_theta']:.3f}  "
          f"joint {d['strict_joint']:.3f}")
    print(f"  WITHIN 1  xy {d['within1_xy']:.3f}  theta {d['within1_theta']:.3f}  "
          f"joint {d['within1_joint']:.3f}   <- headline")
    print(f"  gross>=2  xy {d['gross_xy']:.3f}  theta {d['gross_theta']:.3f}"
          f"           <- real failure rate")
    print(f"  mean miss xy {d['mean_miss_xy']:.2f}  theta {d['mean_miss_theta']:.2f} steps")
    print(f"  miss hist xy    {d['hist_xy']}")
    print(f"  miss hist theta {d['hist_theta']}")
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--rows", type=int, default=200)
    ap.add_argument("--max-turns", type=int, default=200)
    ap.add_argument("--samples", type=int, default=4,
                    help="z_0 draws per (h, c); >1 is what measures the z_0 spread")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mem-fraction", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from datasets import load_from_disk
    from transformers import AutoProcessor

    from longnav.utils.bin_codec import compose_chunk
    from longnav.utils.flow_matching_head import TurnFlowActionRegressor
    from longnav.utils.vector_sft import DataConfig, TurnVectorCollator

    dev = args.device
    if dev.startswith("cuda") and args.mem_fraction > 0:
        torch.cuda.set_per_process_memory_fraction(args.mem_fraction)

    meta = json.loads((Path(args.checkpoint) / "turn_vector_head_config.json").read_text())
    if not meta.get("fm_code"):
        raise SystemExit(f"{args.checkpoint} has no fm_code: not a code-conditioned "
                         "checkpoint, and obedience is undefined for it")
    processor = AutoProcessor.from_pretrained(meta["model"]["model_id"])
    model = TurnFlowActionRegressor.from_pretrained(args.checkpoint, processor, device=dev)
    model.eval()
    codec = model.normalizer
    tok = codec.tokenizer
    if tok is None:
        raise SystemExit("checkpoint carries fm_code but no tokenizer; cannot encode "
                         "generations back to codes")

    ds = load_from_disk(args.dataset)
    ds = ds[args.split] if hasattr(ds, "keys") and args.split in ds else ds
    rng = np.random.default_rng(args.seed)
    rows = rng.choice(len(ds), size=min(args.rows, len(ds)), replace=False)
    collator = TurnVectorCollator(
        processor=processor,
        data=DataConfig(target_column="action_chunks",
                        max_turns_per_sample=args.max_turns),
        train=False, modality_specs=tuple(model.model_cfg.modality_specs))

    print(f"caching h from {args.checkpoint} ...", flush=True)
    H, A, V = collect_h(model, ds, rows, collator, dev)
    H, A = H[V], A[V]                               # drop probe rows once, here
    print(f"{len(H)} valid turns, context dim {H.shape[1]}", flush=True)

    # Teacher codes for every turn, precomputed. Doing this inside the batch loop with a
    # closure counter would silently depend on obey() iterating in order.
    with torch.no_grad():
        TCX, TCT = [], []
        for i in range(0, len(A), args.batch_size):
            cx, ct = tok.encode_chunk(A[i:i + args.batch_size].to(dev))
            TCX.append(cx.cpu()); TCT.append(ct.cpu())
        TCX, TCT = torch.cat(TCX), torch.cat(TCT)
    del model
    torch.cuda.empty_cache()

    res = {"checkpoint": args.checkpoint, "n_turns": int(len(H)),
           "samples_per_pair": int(args.samples),
           "prototype_reference": {"strict_xy": 0.840, "strict_theta": 0.795,
                                   "strict_joint": 0.672, "within1_joint": 0.940,
                                   "note": "c-only prototype: no backbone, no flow objective"}}

    gen = torch.Generator(device=dev).manual_seed(args.seed)
    # The lattice shape, read off the tokenizer rather than assumed -- the miss metric is
    # meaningless without it (see lattice_miss).
    lv_xy = list(tok.model.xy.fsq.levels)
    lv_th = list(tok.model.theta.fsq.levels)
    res["levels"] = {"xy": lv_xy, "theta": lv_th}
    print(f"FSQ lattice: xy {lv_xy}, theta {lv_th}", flush=True)

    @torch.no_grad()
    def obey(codes_fn, tag):
        """codes_fn(h_batch, slice) -> (c_xy, c_theta) to condition on -> miss distances."""
        mx, mt, agree = [], [], []
        for i in range(0, len(H), args.batch_size):
            h = H[i:i + args.batch_size].to(dev)
            cx, ct = codes_fn(h, slice(i, i + len(h)))
            per_sample = []
            for _ in range(args.samples):
                ctx = codec.code_mixer(h, cx, ct)
                z = torch.randn(len(h), codec.decoder.n_ticks, codec.decoder.n_dims,
                                device=dev, generator=gen)
                chunk = compose_chunk(codec.generate(ctx, noise=z))
                gx, gt = tok.encode_chunk(chunk)
                per_sample.append((gx, gt))
                mx.append(lattice_miss(gx.cpu().numpy(), cx.cpu().numpy(), lv_xy))
                mt.append(lattice_miss(gt.cpu().numpy(), ct.cpu().numpy(), lv_th))
            if args.samples > 1:
                # Do the m draws land in the SAME cell as each other? Agreement with
                # draw 0, which is the sub-cell question without reference to c.
                g0x, g0t = per_sample[0]
                for gx, gt in per_sample[1:]:
                    agree.append(((gx == g0x) & (gt == g0t)).cpu().numpy())
        a = np.concatenate(agree) if agree else None
        return (np.concatenate(mx), np.concatenate(mt), a)

    # TEACHER -- the decoder's own obedience, comparable to the prototype number.
    def teacher_codes(h, sl):
        return TCX[sl].to(h.device), TCT[sl].to(h.device)
    mx, mt, ag = obey(teacher_codes, "teacher")
    report("teacher", mx, mt, res)
    if ag is not None:
        res["teacher"]["z0_same_cell"] = float(ag.mean())
        print(f"  z0 spread: {ag.mean():.3f} of extra draws land in the SAME cell as "
              f"draw 0 at fixed (h, c)")

    # POLICY -- the codes RL would actually condition on.
    def policy_codes(h, sl):
        lg = codec.code_head.logits(h.float())
        idx = lg.argmax(dim=-1)
        return idx // codec.code_head.n_theta, idx % codec.code_head.n_theta
    mx, mt, ag = obey(policy_codes, "policy")
    report("policy_argmax", mx, mt, res)
    if ag is not None:
        res["policy_argmax"]["z0_same_cell"] = float(ag.mean())
        print(f"  z0 spread: {ag.mean():.3f} of extra draws land in the SAME cell as "
              f"draw 0 at fixed (h, c)")

    print("\n" + "=" * 70)
    print("Compare TEACHER against the c-only prototype: strict joint 0.672, "
          "within-1 joint 0.940.")
    print("CAVEAT ON THAT COMPARISON: this script measures within-1 as L-inf on the FSQ "
          "lattice. If the prototype's 0.940 was computed as |flat index difference| <= 1 "
          "-- which the surviving diagnostic scripts do not compute at all, they only do "
          "strict equality -- then it counted dim-1 neighbours (a difference of 8) as "
          "gross misses and row wraps as near misses, and the two numbers are NOT "
          "comparable. Only the STRICT columns compare unambiguously.")
    print("A large drop means r(h) has taken authority from c and the RL lever is "
          "weakly connected (CODE_RL_PLAN.md section 9).")
    print("Caveat: the prototype was measured on its own corpus split with no backbone; "
          "this is a different sample and the two are not a controlled comparison.")
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
