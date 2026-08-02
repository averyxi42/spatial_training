"""Fit the action codebook used by the autoregressive policy head.

Production entry point. Replaces the ad-hoc fitters that lived under `dump/`.

    <vln python> data_scripts/fit_action_codebook.py \
        --diffs dump/cnf_head/autoencoder/cache/diffs_train.npz \
        --k 1024 --corpus v2_25hz \
        --out dump/autoregressive_head/codebook_1024_v2_lloyd.json

What this produces is a **lookup table of K prototype actions** -- see
`docs/ACTION_CODEBOOK.md` for what the output file means and why it exists.

Two properties are enforced here rather than left to the caller, because both have
silently corrupted a training run before:

1.  **Full-batch Lloyd, never MiniBatchKMeans.** MiniBatch produced prototypes
    identical to 11 decimal places -- 1024 nominal codes, 773 actually
    distinguishable -- which is an unlearnable label for 42% of ticks and a hard
    ~1.22 nats/token loss floor. Lloyd shows zero degeneracy at every K tried.

2.  **A separation gate that refuses to save.** Every reconstruction metric is blind
    to duplicate prototypes, because duplicates decode identically: RMSE, decisive
    error and stop-keep all read fine while the head is being asked to predict a
    distinction that does not exist. The gate is the only thing that catches it, so
    it runs before the file is written and `--force` is the only way past it.

Nothing about the stop action is imposed. Earlier versions hard-coded index 0 to
literal (0,0,0); that was removed after it was measured to *cause* distortion --
see `docs/ACTION_CODEBOOK.md`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from longnav.utils.action_codebook import ActionCodebook  # noqa: E402


def separation_report(centroids: np.ndarray, tols=(1e-4, 1e-3)) -> dict:
    """How many prototypes can a consumer actually tell apart?

    `n_distinguishable < K` means the head is being asked to predict a distinction
    that does not exist in the action space, and the shortfall is a cross-entropy
    floor no amount of training removes.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    out = {"K": int(centroids.shape[0]),
           "min_separation_m": float(pdist(centroids).min())}
    link = linkage(centroids, "single")
    for tol in tols:
        n = int(len(np.unique(fcluster(link, tol, criterion="distance"))))
        out[f"n_distinguishable_at_{tol:g}"] = n
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset",
                     help="HF dataset dir (e.g. data/v2_25hz/formatted). Differentials "
                          "are computed from its action_chunks. This is the normal path.")
    src.add_argument("--diffs",
                     help="alternatively, an npz of precomputed differentials")
    p.add_argument("--split", default="train")
    p.add_argument("--fit-episodes", type=int, default=None,
                   help="subsample this many episodes when reading --dataset")
    p.add_argument("--k", type=int, default=1024, help="number of prototypes")
    p.add_argument("--out", required=True, help="output .json path")
    p.add_argument("--corpus", default="unknown",
                   help="corpus tag recorded in meta, e.g. v2_25hz. A codebook fitted on "
                        "one corpus is NOT valid for another -- the tick rate and the "
                        "controller dynamics both change the action distribution.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fit-rows", type=int, default=2_000_000,
                   help="subsample this many ticks for the fit (0 = all)")
    p.add_argument("--iters", type=int, default=60)
    p.add_argument("--device", default="cpu")
    p.add_argument("--min-separation", type=float, default=1e-5,
                   help="refuse to save if the closest pair of prototypes is nearer than "
                        "this (metres). The check that catches a degenerate fit.")
    p.add_argument("--force", action="store_true",
                   help="save even if the separation gate fails. Do not use for a "
                        "training run.")
    args = p.parse_args()

    if args.dataset:
        from longnav.utils.action_diffs import load_diffs_from_dataset
        diffs = load_diffs_from_dataset(args.dataset, args.split,
                                        args.fit_episodes, args.seed)
        print(f"loaded {len(diffs):,} ticks from {args.dataset}[{args.split}]")
    else:
        z = np.load(args.diffs)
        diffs = z["diffs"] if "diffs" in z else z[z.files[0]]
        diffs = diffs.reshape(-1, 3).astype(np.float64)
        print(f"loaded {len(diffs):,} ticks from {args.diffs}")

    cb = ActionCodebook.fit(diffs, args.k, seed=args.seed, iters=args.iters,
                            device=args.device,
                            fit_rows=args.fit_rows or None)
    cb.meta["corpus"] = args.corpus

    sep = separation_report(np.asarray(cb.centroids))
    print(json.dumps(sep, indent=1))

    ok = sep["min_separation_m"] >= args.min_separation
    n_distinct = sep.get("n_distinguishable_at_0.0001", args.k)
    if n_distinct < args.k:
        ok = False
        print(f"!! only {n_distinct}/{args.k} prototypes distinguishable at 0.1 mm")

    if not ok and not args.force:
        raise SystemExit(
            f"REFUSING TO SAVE: min separation {sep['min_separation_m']:.3g} m is below "
            f"{args.min_separation:g}, or codes collapse.\nA degenerate codebook trains "
            f"a head to predict labels decided by floating-point noise, and every "
            f"reconstruction metric will look fine. Re-fit (different seed, or lower K) "
            f"rather than passing --force.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cb.save(out)
    (out.with_suffix(".separation.json")).write_text(json.dumps(sep, indent=1))
    print(f"-> {out}\n-> {out.with_suffix('.separation.json')}")


if __name__ == "__main__":
    main()
