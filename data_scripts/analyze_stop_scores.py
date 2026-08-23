#!/usr/bin/env python
"""Score a stop head at the operating point it will actually be deployed at.

Reads the npz dumps written by `--probe-save-scores` and reports what the aggregate
training metrics cannot:

* **TPR at a fixed, deployable FPR.** For ObjectNav the asymmetry is total: a false
  positive ENDS the episode in failure, while a false negative costs nothing because the
  agent keeps driving and re-enters the goal radius (measured on HM3D val rollouts of
  ck12000: median 63 steps outside before first arrival, median 144 steps inside after).
  Simulating first-fire-ends-the-episode over those 394 real episodes gives, against an
  oracle ceiling of 0.723:

      per-step FPR   0.20    0.05    0.01    0.003   0.001   0.0003
      success        0.004   0.08    0.35    0.56    0.66    0.70

  and success is nearly INVARIANT to TPR across 0.1-0.9. So the head must be judged by
  TPR at FPR ~1e-3, not by precision/recall at the 0.5 threshold, and certainly not by
  Youden's J -- J is dominated by TPR, which is the term that does not matter.
* **AUC/AP**, which are threshold-free and base-rate-explicit.
* **The ordered/shuffled split.** Shuffled rows have no elapsed-turn cue, so their AUC is
  the clock-free reading of the head. Pooling the halves hides it.

Usage:
    python data_scripts/analyze_stop_scores.py <run_dir> [--prefix eval_onpolicy_]
"""
import argparse, glob, os, re
import numpy as np


def load(run_dir, prefix=None):
    files = sorted(glob.glob(os.path.join(run_dir, "probe_stop_scores", "*.npz")))
    if not files:
        raise SystemExit(f"no probe_stop_scores/*.npz under {run_dir}; was the run "
                         "launched with --probe-save-scores?")
    by_step = {}
    for f in files:
        base = os.path.basename(f)
        if prefix and not base.startswith(prefix):
            continue
        m = re.search(r"step(\d+)_rank(\d+)\.npz$", base)
        if not m:
            continue
        step = int(m.group(1))
        d = np.load(f)
        by_step.setdefault(step, []).append({k: d[k] for k in d.files})
    return by_step


def roc_stats(logit, label):
    """AUC, AP, and TPR at a ladder of deployable FPRs. No sklearn dependency."""
    ok = np.isfinite(logit) & np.isfinite(label)
    logit, label = logit[ok], label[ok].astype(bool)
    n_pos, n_neg = int(label.sum()), int((~label).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(-logit, kind="mergesort")
    y = label[order]
    tp = np.cumsum(y)
    fp = np.cumsum(~y)
    tpr, fpr = tp / n_pos, fp / n_neg
    auc = float(np.trapz(tpr, fpr))
    prec = tp / np.maximum(tp + fp, 1)
    ap = float(np.sum(np.diff(np.concatenate([[0.0], tpr])) * prec))
    out = {"n": int(ok.sum()), "base_rate": n_pos / (n_pos + n_neg),
           "auc": auc, "ap": ap}
    for target in (0.01, 0.003, 0.001, 0.0003):
        i = np.searchsorted(fpr, target, side="right") - 1
        # the logit threshold that buys this FPR, and the recall it leaves
        out[f"tpr@fpr{target}"] = float(tpr[i]) if i >= 0 else 0.0
        out[f"thr@fpr{target}"] = float(logit[order][i]) if i >= 0 else float("inf")
    return out


def main():
    ap_ = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("run_dir")
    ap_.add_argument("--prefix", default=None,
                     help="component filter, e.g. eval_onpolicy_")
    args = ap_.parse_args()

    by_step = load(args.run_dir, args.prefix)
    for step in sorted(by_step):
        parts = by_step[step]
        lg = np.concatenate([p["logits"] for p in parts])
        lb = np.concatenate([p["labels"] for p in parts])
        sh = np.concatenate([p["shuffled"] for p in parts])
        print(f"\n=== step {step}  ({len(parts)} ranks, {lg.size} turns) ===")
        for name, mask in (("pooled", np.ones_like(sh)),
                           ("ordered", ~sh), ("shuffled (clock-free)", sh)):
            if not mask.any():
                continue
            s = roc_stats(lg[mask], lb[mask])
            if s is None:
                print(f"  {name:22s} one class only -- undefined")
                continue
            print(f"  {name:22s} n={s['n']:6d} base={s['base_rate']:.3f} "
                  f"AUC={s['auc']:.4f} AP={s['ap']:.4f}")
            print(f"  {'':22s} TPR@FPR: "
                  + "  ".join(f"{t}={s[f'tpr@fpr{t}']:.3f}"
                              for t in (0.01, 0.003, 0.001, 0.0003)))
    print("\nDeployability: TPR at FPR 1e-3 is the number to watch. Anything above ~0.1 "
          "there is enough -- the agent gets many frames inside the radius, so recall is "
          "nearly free and false positives are what destroy success.")


if __name__ == "__main__":
    main()
