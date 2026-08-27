#!/usr/bin/env python3
"""What did the dual FSQ tokenizer actually capture?

Reconstruction error is the training objective and answers neither question that
decides the design (`docs/CODE_CONDITIONED_POLICY.md` section 4). This answers two
cheaper ones, both without the backbone:

1. **Does pruning prune?** Occupancy over the joint `|xy| x |theta|` product on the
   FULL corpus, and the coverage curve as a frequency floor rises. With ~2M chunks
   against 1600 cells, "drop what never occurs" may filter nothing -- in which case
   pruning is a deliberate deletion of rare modes, not a free structural win.

2. **Do scans get their own codes?** A scan is an excursion-and-return in cumulative
   heading: large swept range, small net change. Score it as
   `sweep - |net|` and ask whether high-scoring chunks CONCENTRATE in a few theta
   codes or smear across all of them. Concentration is the property the whole design
   needs; smearing means the code cannot express the mode and cannot steer it.

This is a proxy for the conditional gate, not the gate itself -- that one needs `h`
and asks whether the code removes residual variation in `A` given the state.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

sys.path.insert(0, str(_ROOT / "data_scripts"))
from train_chunk_tokenizer import DualTokenizer, load_chunks, to_cumulative  # noqa: E402


def coverage_curve(counts, floors):
    """Fraction of CHUNKS retained if cells with fewer than `f` occurrences are cut."""
    total = counts.sum()
    return [{"floor": int(f),
             "cells_kept": int((counts >= f).sum()),
             "chunk_coverage": float(counts[counts >= f].sum() / total)}
            for f in floors]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--row-stride", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mem-fraction", type=float, default=0.06)
    ap.add_argument("--scan-quantile", type=float, default=0.98,
                    help="chunks above this quantile of (sweep - |net|) are 'scans'")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.device.startswith("cuda") and args.mem_fraction > 0:
        torch.cuda.set_per_process_memory_fraction(args.mem_fraction)

    ck = torch.load(args.tokenizer, map_location="cpu", weights_only=False)
    model = DualTokenizer(ck["xy_levels"], ck["theta_levels"],
                          d_model=ck["d_model"], n_layers=ck["n_layers"])
    model.load_state_dict(ck["model"])
    model.eval().to(args.device)
    n_xy, n_th = model.xy.fsq.vocab, model.theta.fsq.vocab

    raw = load_chunks(args.dataset, args.split, args.row_stride, 0)
    chunks, was_cum, _, _ = to_cumulative(raw)
    print(f"{len(chunks)} chunks, stored_cumulative={was_cum}", flush=True)

    theta = chunks[..., 2]
    net = np.abs(theta[:, -1])
    sweep = theta.max(axis=1) - theta.min(axis=1)
    scanness = sweep - net                       # >0 iff the heading came back
    thr = float(np.quantile(scanness, args.scan_quantile))
    is_scan = scanness >= thr
    print(f"scanness threshold q{args.scan_quantile} = {thr:.4f} rad; "
          f"{is_scan.sum()} scan-like chunks ({100*is_scan.mean():.2f}%)")

    norm = chunks.copy()
    norm[..., :2] /= ck["xy_scale"]
    norm[..., 2] /= ck["theta_scale"]

    xi, ti, err_xy, err_th = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(norm), args.batch_size):
            b = torch.from_numpy(norm[i:i + args.batch_size]).to(args.device)
            xy_hat, th_hat, x_idx, t_idx = model(b)
            xi.append(x_idx.cpu().numpy()); ti.append(t_idx.cpu().numpy())
            err_xy.append(((xy_hat - b[..., :2]) ** 2).mean(dim=(1, 2)).cpu().numpy())
            err_th.append(((th_hat - b[..., 2:3]) ** 2).mean(dim=(1, 2)).cpu().numpy())
    xi = np.concatenate(xi); ti = np.concatenate(ti)
    err_xy = np.sqrt(np.concatenate(err_xy)) * ck["xy_scale"]
    err_th = np.sqrt(np.concatenate(err_th)) * ck["theta_scale"]

    joint = xi * n_th + ti
    jcounts = np.bincount(joint, minlength=n_xy * n_th)
    tcounts = np.bincount(ti, minlength=n_th)
    scan_tcounts = np.bincount(ti[is_scan], minlength=n_th)

    # Concentration: how few theta codes hold most scans, and how much scans are
    # ENRICHED in their top codes relative to the corpus. Enrichment is the real
    # signal -- a code holding many scans only matters if it is not simply a large code.
    order = np.argsort(-scan_tcounts)
    scan_share = scan_tcounts[order] / max(scan_tcounts.sum(), 1)
    base_share = tcounts[order] / tcounts.sum()
    cum = np.cumsum(scan_share)
    k80 = int(np.searchsorted(cum, 0.80) + 1)
    enrich = float(scan_share[:k80].sum() / max(base_share[:k80].sum(), 1e-9))

    res = {
        "n_chunks": int(len(chunks)),
        "vocab": {"xy": n_xy, "theta": n_th, "joint": n_xy * n_th},
        "recon": {
            "rmse_xy_m": float(err_xy.mean()),
            "rmse_theta_rad": float(err_th.mean()),
            "rmse_theta_rad_scans": float(err_th[is_scan].mean()),
            "rmse_theta_rad_nonscans": float(err_th[~is_scan].mean()),
        },
        "occupancy": {
            "joint_cells_occupied": int((jcounts > 0).sum()),
            "joint_cells_total": int(n_xy * n_th),
            "coverage_curve": coverage_curve(
                jcounts, [1, 10, 100, 500, 1000, 2000, 5000, 10000]),
        },
        "scans": {
            "threshold_rad": thr,
            "n_scans": int(is_scan.sum()),
            "theta_codes_holding_80pct": k80,
            "enrichment_of_those_codes": enrich,
            "top_codes": [{"code": int(order[i]),
                           "scan_share": float(scan_share[i]),
                           "corpus_share": float(base_share[i])}
                          for i in range(min(6, n_th))],
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
