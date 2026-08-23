"""Per-episode probe records and their aggregation.

Two entry points, one format:

  `write_episode_records(out_dir, rollouts, results)` -- called by
  `longnav.scripts.eval` as each batch of rollouts lands, so the run has an early
  read and nothing is held in memory to the end.

  `aggregate(run_dir)` -- reads those records back off disk and produces the summary.
  Callable at the end of the eval script or standalone against a finished run
  (`python -m longnav.utils.probe_eval_report <run_dir>`), because everything needed
  is already on disk either way.

The record is one JSON line per episode in `probe_records.jsonl`:

    {"episode": str, "scene": str, "goal": str, "success": int, "steps": int,
     "true_distance": [...],     # env's own DistanceToGoal, per policy step
     "pred_distance": [...],     # distance head expectation, per step
     "p_stop": [...],            # P(d <= stop_radius), per step
     "value": [...],             # value head expectation, per step (may be absent)
     "reward": [...]}

Per-step series are kept in full and deliberately: they are small next to the video
the run already writes, and every question asked of this data so far (accuracy by
distance band, AUC at a stop radius, calibration, bias vs elapsed time) needs the
series rather than a summary. Aggregation is therefore pure post-processing.

What the summary reports, and why each line exists:

  * MAE / bias by TRUE-distance band -- an aggregate MAE is dominated by the far tail
    (episodes start 5-10 m out) and says nothing about the regime a stop rule lives in.
  * The no-perception baselines (global constant, per-scene constant, per-episode
    median) -- a head that does not beat these is not reading its input, and reporting
    them stops a good-looking MAE from being mistaken for perception.
  * AUC of P(d <= r) and of -E[d] at several radii -- the ranking quality a stop rule
    consumes, which is not the same thing as regression accuracy.
  * corr(prediction, step index) partialled on true distance -- the clock-shortcut
    signature. Demonstrations have corr(distance, step) = -0.60, so a head can score
    well by reading the clock; on-policy rollouts break that and this number exposes it.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

RECORD_FILE = "probe_records.jsonl"
SUMMARY_FILE = "probe_summary.json"


def _series(x) -> List[Optional[float]]:
    if x is None:
        return []
    a = np.asarray(x, dtype=float).reshape(-1)
    return [None if not np.isfinite(v) else float(v) for v in a]


def write_episode_records(out_dir: str,
                          rollouts: Sequence[Any],
                          results: Sequence[Dict[str, Any]],
                          stop_radius: float = 1.0) -> int:
    """Append one line per episode. Returns how many were written.

    `rollouts` are the trajectory dicts from `collect_rollouts` (probe outputs live
    there); `results` are the env-side per-episode dicts (labels, success, and the
    env's own distance series). Rows whose trajectory carries no probe output are
    still written -- with empty prediction series -- so a missing probe is visible in
    the record rather than silently reducing the sample.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, RECORD_FILE)
    n = 0
    with open(path, "a") as fh:
        for traj, res in zip(rollouts, results):
            if traj is None:
                continue
            t = traj[0] if isinstance(traj, (tuple, list)) else traj
            if not isinstance(t, dict):
                continue
            rec = {
                "episode": res.get("episode_label") or res.get("uid") or "",
                "scene": res.get("scene_id", ""),
                "goal": res.get("goal", ""),
                "success": int(res.get("success", 0) or 0),
                "oracle_success": int(res.get("oracle_success", 0) or 0),
                "steps": int(res.get("n_steps", 0) or 0),
                "stop_radius": float(stop_radius),
                "true_distance": _series(t.get("distance_to_goal")),
                "pred_distance": _series(t.get("probe_distance_m")),
                "p_stop": _series(t.get("probe_p_stop")),
                "value": _series(t.get("values")),
                "reward": _series(t.get("rewards")),
            }
            fh.write(json.dumps(rec) + "\n")
            n += 1
    return n


def _auc(score: np.ndarray, label: np.ndarray) -> float:
    if label.sum() == 0 or (~label).sum() == 0:
        return float("nan")
    order = np.argsort(score)
    lab = label[order]
    ranks = np.arange(1, len(lab) + 1)
    pos, neg = lab.sum(), (~lab).sum()
    return float((ranks[lab].sum() - pos * (pos + 1) / 2) / (pos * neg))


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float((a * b).sum() / d) if d > 0 else float("nan")


def aggregate(run_dir: str, bands=((0, 1), (1, 2), (2, 3), (3, 5), (5, 10), (10, 1e9)),
              radii=(0.5, 1.0, 1.5, 2.0), write: bool = True) -> Dict[str, Any]:
    """Read `probe_records.jsonl` from a finished (or running) eval and summarise."""
    path = os.path.join(run_dir, RECORD_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(f"no {RECORD_FILE} in {run_dir}")
    P, D, T, EP, SC = [], [], [], [], []
    n_ep = n_nopred = 0
    succ = []
    for i, line in enumerate(open(path)):
        r = json.loads(line)
        n_ep += 1
        succ.append(r.get("success", 0))
        td, pd = r.get("true_distance") or [], r.get("pred_distance") or []
        if not pd:
            n_nopred += 1
            continue
        m = min(len(td), len(pd))
        ps = (r.get("p_stop") or [None] * m)
        for k in range(m):
            if td[k] is None or pd[k] is None:
                continue
            D.append(td[k]); P.append(pd[k]); T.append(k)
            EP.append(i); SC.append(r.get("scene", ""))
    if not D:
        raise ValueError(f"{path} has no usable (true, predicted) pairs")
    D, P, T, EP = map(np.asarray, (D, P, T, EP))
    SC = np.asarray(SC)
    out: Dict[str, Any] = {
        "run_dir": run_dir, "episodes": n_ep, "episodes_without_predictions": n_nopred,
        "steps": int(len(D)), "success_rate": float(np.mean(succ)) if succ else None,
        "overall": {"mae_m": float(np.abs(P - D).mean()),
                    "bias_m": float((P - D).mean()),
                    "corr": _corr(P, D)},
    }
    # no-perception baselines: a head must beat these to be reading its input
    gmed = float(np.median(D))
    scene_med = {s: float(np.median(D[SC == s])) for s in np.unique(SC)}
    ep_med = {e: float(np.median(D[EP == e])) for e in np.unique(EP)}
    out["baselines"] = {
        "global_constant_mae_m": float(np.abs(D - gmed).mean()),
        "per_scene_constant_mae_m": float(np.abs(D - np.array([scene_med[s] for s in SC])).mean()),
        "per_episode_median_mae_m": float(np.abs(D - np.array([ep_med[e] for e in EP])).mean()),
    }
    out["bands"] = []
    for lo, hi in bands:
        m = (D >= lo) & (D < hi)
        if m.sum() < 20:
            continue
        out["bands"].append({"lo": lo, "hi": None if hi > 1e8 else hi, "n": int(m.sum()),
                             "pred_mean_m": float(P[m].mean()),
                             "mae_m": float(np.abs(P[m] - D[m]).mean()),
                             "bias_m": float((P[m] - D[m]).mean())})
    out["discrimination"] = []
    for r in radii:
        lab = D <= r
        out["discrimination"].append({"radius_m": r, "base_rate": float(lab.mean()),
                                      "auc_neg_expectation": _auc(-P, lab)})
    # clock-shortcut signature: does the prediction track the step index once the true
    # distance is accounted for?
    def _resid(y, x):
        A = np.vstack([x, np.ones_like(x)]).T
        return y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    out["clock"] = {
        "corr_pred_step": _corr(P, T.astype(float)),
        "corr_true_step": _corr(D, T.astype(float)),
        "partial_corr_pred_step_given_true": _corr(_resid(P, D), _resid(T.astype(float), D)),
    }
    if write:
        with open(os.path.join(run_dir, SUMMARY_FILE), "w") as fh:
            json.dump(out, fh, indent=2)
    return out


def format_summary(s: Dict[str, Any]) -> str:
    L = [f"probe eval: {s['episodes']} episodes ({s['episodes_without_predictions']} without "
         f"predictions), {s['steps']} steps, success {s['success_rate']:.3f}"
         if s.get("success_rate") is not None else
         f"probe eval: {s['episodes']} episodes, {s['steps']} steps"]
    o, b = s["overall"], s["baselines"]
    L.append(f"  distance: MAE {o['mae_m']:.2f} m  bias {o['bias_m']:+.2f} m  corr {o['corr']:.3f}")
    L.append(f"  no-perception baselines: global {b['global_constant_mae_m']:.2f} | "
             f"per-scene {b['per_scene_constant_mae_m']:.2f} | "
             f"per-episode median (cheats) {b['per_episode_median_mae_m']:.2f} m")
    for band in s["bands"]:
        hi = "inf" if band["hi"] is None else f"{band['hi']:g}"
        L.append(f"    {band['lo']:g}-{hi:>4s} m (n={band['n']:6d}): pred {band['pred_mean_m']:5.2f}  "
                 f"MAE {band['mae_m']:5.2f}  bias {band['bias_m']:+5.2f}")
    for d in s["discrimination"]:
        L.append(f"  AUC(-E[d]) at {d['radius_m']:.1f} m: {d['auc_neg_expectation']:.3f} "
                 f"(base rate {d['base_rate']:.1%})")
    c = s["clock"]
    L.append(f"  clock shortcut: corr(pred,step) {c['corr_pred_step']:+.3f}, "
             f"corr(true,step) {c['corr_true_step']:+.3f}, "
             f"PARTIAL corr(pred,step | true) {c['partial_corr_pred_step_given_true']:+.3f}")
    return "\n".join(L)


if __name__ == "__main__":
    import sys
    for d in sys.argv[1:]:
        print(format_summary(aggregate(d)))
