#!/usr/bin/env python
"""Build a probe-only stop/distance corpus from a CLEAN on-policy frame store.

Input is what `eval_objectnav_policy.py --save-frames` writes: the raw sensor render
per policy step plus a per-episode JSON of aligned labels. It is emphatically NOT the
debug MP4 -- that video has `distance_to_goal`, the goal name and the step index drawn
into the corner, and the corpora built from decoded video had the labels in the pixels
(retracted 2026-08-23).

Two structural properties, both enforced here rather than trusted:

* **Split by SCENE, from two separate rollout runs.** `--train-frames` is collected on
  HM3D train scenes and `--val-frames` on HM3D val. Disjointness is asserted and the
  build fails otherwise. Note the property that actually matters is not disjointness
  *within this corpus* but against everything the checkpoint has ever seen: HM3D val is
  the only split untouched by the PointNav corpus (100% HM3D train) and the ObjectNav
  demonstrations (MP3D). A scene-disjoint partition of a train-scene archive is NOT a
  valid held-out set for this lineage, which is how the v6/v7 numbers were invalidated.
* **Probe-only rows never train actions.** `action_chunks` is all-NaN and `probe_only`
  is True; the trainer refuses a probe_only row whose action targets are finite, and
  zeroes the action loss through a rank-uniform multiplier. Proven zero-gradient in
  `tests/test_gradient_isolation.py` (I1/I7: probe rows contribute BITWISE nothing to
  the flow head's accumulated gradient).

Ordered vs shuffled rows are disjoint BY EPISODE, not two views of the same frames, so
one epoch does not see every frame twice and the two conditions stay statistically
independent. Shuffling permutes the turns within a row; the per-frame stop label
permutes with its frame and means the same thing in both orderings, which is exactly
why the label is a distance predicate rather than "this is the last turn".
"""
import argparse, json, glob, os, hashlib, math
import numpy as np
from datasets import Dataset, DatasetDict, load_from_disk

SYSTEM = ("You are a robot navigating an indoor environment toward a goal object.\n"
          "Goal: {goal}\n"
          "At each step you receive the current RGB observation. Produce the next short "
          "trajectory of poses to follow, relative to your current pose.")

# Copied from a reference corpus, never invented: a wrong stamp makes the mixture guard
# refuse the build, and a plausible-but-wrong one makes it silently accept a corpus
# recorded at a different rate.
STAMP_KEYS = ("native_fps", "dt_native", "obs_hz", "obs_stride_frames",
              "action_chunk_len", "obs_interval", "effective_obs_hz", "chunk_duration",
              "overlap_fraction", "obs_spacing_mode", "v_max_mps", "w_max_radps",
              "recording_schema_version", "native_fps_source", "return_gamma")


def scene_of(rec):
    return os.path.basename(str(rec.get("scene_id") or "")).split(".")[0]


def load_episodes(frames_dir):
    out = []
    for f in sorted(glob.glob(os.path.join(frames_dir, "episodes", "*.json"))):
        try:
            out.append(json.load(open(f)))
        except json.JSONDecodeError:
            print(f"  skipping unreadable {os.path.basename(f)} (run still writing?)")
    return out


def build_rows(episodes, *, stamps, max_obs, min_obs, stop_radius, shuffle_seed,
               shuffle_frac, chunk_len, action_dim):
    rows, rng = [], np.random.default_rng(shuffle_seed)
    for ep in episodes:
        n = int(ep["num_steps"])
        if n < min_obs:
            continue
        # ordered/shuffled assignment is deterministic per episode, so a rebuild with
        # more episodes does not reshuffle the ones already in it
        h = int(hashlib.sha1(ep["uid"].encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        shuffle = h < shuffle_frac
        for start in range(0, n, max_obs):
            idx = list(range(start, min(start + max_obs, n)))
            if len(idx) < min_obs:
                continue
            if shuffle:
                idx = list(rng.permutation(idx))
            d = [ep["distance_to_goal"][i] for i in idx]
            msgs = [{"role": "user", "content": [
                {"type": "text", "text": SYSTEM.format(goal=ep.get("goal") or "object")}]}]
            for _ in idx:
                msgs.append({"role": "user", "content": [
                    {"type": "text", "text": "Observation:"},
                    {"type": "image", "text": None},
                    {"type": "text", "text": "Action:"}]})
                msgs.append({"role": "assistant",
                             "content": [{"type": "text", "text": "**____**"}]})
            row = {
                "episode_id": f"{ep['uid']}#{start//max_obs}",
                "scene_id": ep.get("scene_id"),
                "goal_text": ep.get("goal") or "",
                "num_observations": len(idx),
                "images": [ep["rgb_paths"][i] for i in idx],
                # All-NaN: these rows carry the POLICY's own actions, and training on
                # them would be self-imitation. The shape is kept so the collator's
                # contract holds; the trainer refuses the row if any entry is finite.
                "action_chunks": np.full((len(idx), chunk_len, action_dim),
                                         np.nan, dtype=np.float64).tolist(),
                "messages": msgs,
                "distance_targets": [float("nan") if v is None else float(v) for v in d],
                "stop_targets": [float("nan") if v is None else float(float(v) <= stop_radius)
                                 for v in d],
                # No value head in this experiment, but the column is kept NaN so the
                # schema matches the demonstration corpora exactly and the mixture
                # loader has nothing to reconcile.
                "return_targets": [float("nan")] * len(idx),
                "stop_radius_m": float(stop_radius),
                "probe_only": True,
                "turns_shuffled": bool(shuffle),
            }
            row.update(stamps)
            rows.append(row)
    return rows


def report(name, rows):
    n = pos = nan = 0
    per_scene = {}
    for r in rows:
        for v in r["stop_targets"]:
            n += 1
            if v != v:
                nan += 1
            elif v > 0.5:
                pos += 1
        s = scene_of(r)
        a, b = per_scene.get(s, (0, 0))
        per_scene[s] = (a + sum(1 for v in r["stop_targets"] if v == 1.0),
                        b + sum(1 for v in r["stop_targets"] if v == v))
    lab = n - nan
    rate = pos / lab if lab else float("nan")
    sh = sum(1 for r in rows if r["turns_shuffled"])
    print(f"[{name}] rows={len(rows)} ({sh} shuffled / {len(rows)-sh} ordered)  "
          f"frames={n}  labelled={lab} ({lab/max(n,1):.1%})")
    print(f"    positive rate {rate:.2%}   balanced pos_weight {(1-rate)/rate:.2f}")
    # No-perception baselines. "Beats a constant predictor" is the minimum bar for
    # claiming a head reads its input, and it is the bar the retracted runs never
    # actually cleared on unseen scenes.
    print(f"    baselines: always-negative acc {1-rate:.3f}   always-positive acc {rate:.3f}")
    if per_scene:
        pr = [p / max(t, 1) for p, t in per_scene.values()]
        print(f"    per-scene positive rate: min {min(pr):.2%} median "
              f"{sorted(pr)[len(pr)//2]:.2%} max {max(pr):.2%}  ({len(per_scene)} scenes)")
    return per_scene


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-frames", required=True)
    ap.add_argument("--val-frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stamp-from",
                    default="/Projects/data/v2_25hz_obs2.5hz/formatted_nopose_dist097_nostep_stop10",
                    help="reference corpus whose recording stamps are copied verbatim")
    ap.add_argument("--max-obs", type=int, default=175)
    ap.add_argument("--min-obs", type=int, default=12)
    ap.add_argument("--stop-radius", type=float, default=1.0)
    ap.add_argument("--shuffle-frac", type=float, default=0.5)
    ap.add_argument("--shuffle-seed", type=int, default=0)
    args = ap.parse_args()

    ref = load_from_disk(args.stamp_from)
    ref = ref["train"] if hasattr(ref, "keys") else ref
    r0 = ref[0]
    stamps = {k: r0[k] for k in STAMP_KEYS if k in ref.column_names}
    chunk_len = int(r0["action_chunk_len"])
    action_dim = int(np.array(r0["action_chunks"]).shape[-1])
    print(f"stamps from {args.stamp_from}: chunk_len={chunk_len} action_dim={action_dim}")
    missing = [k for k in STAMP_KEYS if k not in stamps]
    if missing:
        raise SystemExit(f"reference corpus lacks stamp columns {missing}; wrong "
                         "--stamp-from. Refusing to invent them.")

    tr_eps, va_eps = load_episodes(args.train_frames), load_episodes(args.val_frames)
    print(f"episodes: train={len(tr_eps)}  val={len(va_eps)}")
    tr_scenes = {scene_of(e) for e in tr_eps}
    va_scenes = {scene_of(e) for e in va_eps}
    overlap = tr_scenes & va_scenes
    if overlap:
        raise SystemExit(
            f"{len(overlap)} scenes appear in BOTH splits ({sorted(overlap)[:5]}). "
            "Train and eval must never share scenes -- an episode-level or "
            "within-archive split turns a generalization metric into within-scene "
            "localization. Refusing to build.")
    print(f"scenes: train={len(tr_scenes)}  val={len(va_scenes)}  overlap=0  OK")

    kw = dict(stamps=stamps, max_obs=args.max_obs, min_obs=args.min_obs,
              stop_radius=args.stop_radius, shuffle_seed=args.shuffle_seed,
              shuffle_frac=args.shuffle_frac, chunk_len=chunk_len,
              action_dim=action_dim)
    tr = build_rows(tr_eps, **kw)
    va = build_rows(va_eps, **kw)
    report("train", tr)
    report("validation", va)

    # Shuffle integrity, checked over the CORPUS rather than per row: a single shuffled
    # row can legitimately have its only positive last (probability 1/n under a uniform
    # permutation), and asserting that per row killed a run at step 46.
    for name, rows in (("train", tr), ("validation", va)):
        sh = [r for r in rows if r["turns_shuffled"]]
        if not sh:
            continue
        # Only rows whose LAST turn is actually labelled can answer the question; a
        # row ending on an unlabelled frame is neither evidence for nor against. Pooling
        # over frames (not per-row means) also avoids an all-NaN row poisoning the
        # expectation with NaN, which silently disabled this check.
        tail = [r for r in sh if r["stop_targets"] and r["stop_targets"][-1] == r["stop_targets"][-1]]
        if not tail:
            print(f"[{name}] no shuffled row ends on a labelled frame; check skipped")
            continue
        last = sum(1 for r in tail if r["stop_targets"][-1] == 1.0)
        flat = np.array([v for r in sh for v in r["stop_targets"]], dtype=float)
        mean_rate = float(np.nanmean(flat))
        frac_last = last / len(tail)
        print(f"[{name}] shuffled rows whose LAST turn is positive: {frac_last:.3f} "
              f"(expected ~{mean_rate:.3f} under a uniform permutation)")
        if frac_last > mean_rate + 0.15:
            raise SystemExit(
                "shuffled rows still concentrate positives at the end; the permutation "
                "did not take. This is the cue shuffling exists to destroy.")

    for name, rows in (("train", tr), ("validation", va)):
        if not rows:
            raise SystemExit(f"{name} split is empty; nothing to build")
    ds = DatasetDict({"train": Dataset.from_list(tr),
                      "validation": Dataset.from_list(va)})
    ds.save_to_disk(args.out)
    print(f"wrote {args.out}: " + ", ".join(f"{k}={len(v)}" for k, v in ds.items()))


if __name__ == "__main__":
    main()
