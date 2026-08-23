"""Turn mined on-policy rollout frames into a PROBE-ONLY training corpus.

Why this exists
---------------
The distance head is supervised only on successful expert demonstrations, where
elapsed time and remaining distance are confounded: within an ObjectNav demo,
corr(true distance, step) = -0.60 and 96% of episodes get monotonically closer.
The head learned that prior and it is measurably wrong on the policy's own
rollouts (corr -0.26, and absent in the long failing episodes that matter for
stopping): partial corr(prediction, step | true distance) = -0.13, i.e. it still
predicts "closer" as the clock advances after controlling for the truth. Removing
the "Observation N:" label did not fix it -- the model counts turns anyway.

The fix is distributional: supervise distance on trajectories where time does NOT
imply progress. The policy's own rollouts are exactly that, and they come with
ground-truth geodesic distance from the env (`sequence.json`).

PROBE-ONLY, and why the flow loss must be masked
------------------------------------------------
These states carry no expert action. Training the flow head on the policy's own
actions is self-imitation: it reinforces current behaviour, mistakes included, and
buys nothing. So rows built here set `probe_only=True` and carry a zero
`action_chunks` block purely so the collator's shape contract still holds; the
trainer zeroes the turn loss for these rows and keeps only the distance/value
losses.

--shuffle-turns
---------------
Permutes the observation turns within each row (image, distance target and the
zeroed action chunk move together). This destroys temporal context on purpose:
"later" can no longer mean anything, so the only way to predict distance is to
look at the current image. It is a deliberately extreme regime -- it makes the row
useless for policy learning of any kind, which is why it is probe-only twice over
-- and is meant to be MIXED with unshuffled on-policy rows so the model keeps both
skills: integrating context to localise itself, and reading the present frame.

Usage
-----
    python data_scripts/build_onpolicy_distance_dataset.py \
        --frames-dirs dump/eval_system/value_offline/onpolicy_frames_p* \
        --out /Projects/data/onpolicy_distance/formatted_ordered \
        --val-episodes 120
    # plus a shuffled twin from the same frames:
    ... --shuffle-turns --shuffle-seed 0 --out .../formatted_shuffled
"""

raise SystemExit(
    "RETIRED 2026-08-23. This script consumed frames recovered by decoding rollout "
    "video.mp4 files. Those videos carry a rendered HUD printing distance_to_goal, goal "
    "and step, so every corpus and every hidden-state cache built through this path had "
    "the labels in the pixels. mine_rollout_frames.py has been deleted and its outputs "
    "purged. If you need on-policy observations, have the rollout WRITE its frames, and "
    "evaluate on HM3D val -- never on scenes any corpus in this lineage trained on."
)


import argparse
import collections
import glob
import json
import os
import random

import numpy as np

SYSTEM = ("You are a robot navigating an indoor environment toward a goal object.\n"
          "Goal: {goal}\n"
          "At each step you receive the current RGB observation. Produce the next short "
          "trajectory of poses to follow, relative to your current pose.")


def build_rows(frames_dirs, min_obs, max_obs, shuffle, seed, stop_radius=1.0):
    rng = random.Random(seed)
    rows = []
    skipped_no_goal = []
    for fd in frames_dirs:
        lab = os.path.join(fd, "labels.jsonl")
        if not os.path.exists(lab):
            continue
        by = collections.defaultdict(list)
        for line in open(lab):
            r = json.loads(line)
            by[r["episode_tag"]].append(r)
        meta = {}
        mpath = os.path.join(fd, "mining_manifest.json")
        if os.path.exists(mpath):
            meta = json.loads(open(mpath).read())
        _index_goals(meta.get("runs", []))
        for tag, rs in by.items():
            rs.sort(key=lambda r: r["step"])
            keep = [r for r in rs if r["distance"] is not None and os.path.exists(r["image"])]
            if len(keep) < min_obs:
                continue
            keep = keep[:max_obs]
            if shuffle:
                rng.shuffle(keep)
            goal = _GOAL_CACHE.get(tag, "")
            if not goal:
                skipped_no_goal.append(tag); continue
            msgs = [{"role": "user", "content": [{"type": "text", "text": SYSTEM.format(goal=goal)}]}]
            for _ in keep:
                msgs.append({"role": "user", "content": [
                    {"type": "text", "text": "Observation:"},
                    {"type": "image"},
                    {"type": "text", "text": "Action:"}]})
                msgs.append({"role": "assistant", "content": [{"type": "text", "text": "**____**"}]})
            rows.append({
                "episode_id": tag,
                "goal_text": goal,
                "images": [r["image"] for r in keep],
                "messages": msgs,
                "distance_targets": [float(r["distance"]) for r in keep],
                # PER-FRAME stop label: the event is "within stop_radius of the goal",
                # NOT "this is the episode's last turn". The structural label in
                # stop_head.episode_stop_labels cannot be used here for two reasons:
                # it does not survive turn shuffling (the positive is defined as the
                # window's final position, so a permutation puts it at a random slot
                # and teaches position-reading -- the exact cue shuffling exists to
                # destroy), and these rollouts terminate when an ORACLE fires at 0.4 m,
                # so "final turn" would mean "at the oracle boundary" while frames at
                # 0.5-0.9 m are labelled negative. A per-frame metric label permutes
                # with its frame and means the same thing in both orderings.
                "stop_targets": [float(float(r["distance"]) <= stop_radius) for r in keep],
                # value targets are NaN on shuffled rows: G_t is a property of the
                # trajectory that FOLLOWS the state, and shuffling destroys the context
                # the model would need to estimate it. The probe losses NaN-mask per
                # element, so this disables the value head on these rows and leaves the
                # distance head (a pure function of the current pose) training.
                "return_targets": ([float("nan")] * len(keep) if shuffle
                                   else [float(r["return"]) for r in keep]),
                "return_gamma": 0.97,
                "probe_only": True,          # never eligible for action/flow training
                "turns_shuffled": bool(shuffle),
                "num_observations": len(keep),
            })
    if skipped_no_goal:
        print(f"skipped {len(skipped_no_goal)} episodes with no recoverable goal string "
              f"(e.g. {skipped_no_goal[:2]})")
    return rows


_GOAL_CACHE = {}
_INDEXED = set()


def _index_goals(runs):
    """One pass over each run's rollout tree: episode tag -> goal string.

    Scanning per tag was O(episodes^2) over ~13k summary.json files; this indexes
    each run once. Tags are sanitised the same way `mine_rollout_frames` does.
    """
    for run in runs:
        if run in _INDEXED:
            continue
        _INDEXED.add(run)
        for summ in glob.glob(os.path.join(run, "rollout", "*", "*", "summary.json")):
            tag = os.path.basename(os.path.dirname(summ))
            tag = tag.replace("@", "_").replace(":", "_").replace("#", "_")
            try:
                _GOAL_CACHE[tag] = json.loads(open(summ).read()).get("goal", "")
            except Exception:
                continue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dirs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-obs", type=int, default=12)
    ap.add_argument("--max-obs", type=int, default=175)
    ap.add_argument("--chunk-len", type=int, default=20,
                    help="width of the ZERO action-chunk block; must match the corpus this "
                         "will be mixed with, since the collator stacks them")
    ap.add_argument("--val-scenes", type=int, default=12,
                    help="number of SCENES held out entirely for validation. Splitting by "
                         "episode is NOT enough: a rollout archive revisits the same "
                         "scenes, so an episode split leaves 100%% of val scenes in train "
                         "and the metric measures localisation in familiar scenes rather "
                         "than distance perception (2026-08-22). Scene disjointness is "
                         "asserted below and printed; 0 is allowed only with "
                         "--allow-shared-scenes and a stated reason.")
    ap.add_argument("--allow-shared-scenes", default="",
                    help="explicit justification for sharing scenes between splits. "
                         "Non-empty value required to build such a corpus at all.")
    ap.add_argument("--stamp-from", default="/Projects/data/v2_25hz_obs2.5hz/formatted_nopose_dist097_nostep",
                    help="corpus whose stamped chunking/kinematic block these rows COPY. "
                         "build_mixture refuses to mix sources whose stamps disagree, and "
                         "inventing values here is how that guard gets tripped (it was, on "
                         "2026-08-22: v_max_mps 1.0 guessed vs 2.0 actual). Read, never guess.")
    ap.add_argument("--stop-radius", type=float, default=1.0,
                    help="a frame is a stop positive when its geodesic distance to the "
                         "goal is <= this. 1.0 m matches the offline harness's "
                         "--success-distance; the env's own oracle fires at 0.4 m, which "
                         "no observation ever sees (the episode ends there).")
    ap.add_argument("--shuffle-turns", action="store_true")
    ap.add_argument("--shuffle-seed", type=int, default=0)
    args = ap.parse_args()

    from datasets import Dataset, DatasetDict

    from datasets import load_from_disk as _lfd
    _ref = _lfd(args.stamp_from)["train"][0]
    # `task` is optional by design: absent (or None) MEANS objectnav, and the ObjectNav
    # corpus therefore does not carry the column at all. Requiring it here would refuse
    # the very corpus these rows are built to match.
    STAMP_KEYS = ("native_fps", "dt_native", "obs_stride_frames", "action_chunk_len",
                  "obs_spacing_mode", "v_max_mps", "w_max_radps")
    stamp = {k: _ref[k] for k in STAMP_KEYS if k in _ref}
    if "task" in _ref:
        stamp["task"] = _ref["task"]
    missing = [k for k in STAMP_KEYS if k not in stamp]
    if missing:
        raise SystemExit(f"--stamp-from corpus lacks {missing}; cannot copy the block")
    args.chunk_len = int(stamp["action_chunk_len"])
    print(f"stamps copied from {args.stamp_from}: {stamp}")
    rows = build_rows(args.frames_dirs, args.min_obs, args.max_obs,
                      args.shuffle_turns, args.shuffle_seed, args.stop_radius)
    if not rows:
        raise SystemExit("no rows built -- check --frames-dirs")
    # structural invariant, asserted rather than assumed: every row here is probe-only,
    # and a shuffled row additionally carries no value target.
    assert all(r["probe_only"] for r in rows), "probe_only must hold for every row"
    if args.shuffle_turns:
        assert all(r["turns_shuffled"] for r in rows)
        assert all(not np.isfinite(np.asarray(r["return_targets"], float)).any() for r in rows), \
            "shuffled rows must have NaN value targets"
    for r in rows:
        n = r["num_observations"]
        # NaN, deliberately, NOT zeros. These rows have no expert action; the block
        # exists only so the collator's shape contract holds. If the trainer's
        # probe-only masking is ever missing or mis-wired, a NaN turn loss stops the
        # run on step 1 -- whereas zeros would silently teach the policy to stand
        # still, which is both wrong and hard to notice.
        r["action_chunks"] = np.full((n, args.chunk_len, 3), np.nan, dtype=np.float32).tolist()
        r.update(stamp)
    assert all(np.isnan(np.asarray(r["action_chunks"], float)).all() for r in rows), \
        "action chunks must be NaN so a missing mask fails loudly"
    def _scene_of(r):
        # tag: <scene>_<episode>_<pid>_<timestamp>
        return r["episode_id"].split("_")[0]
    scenes = sorted({_scene_of(r) for r in rows})
    rng = random.Random(1234)
    rng.shuffle(rows)
    if args.val_scenes > 0:
        srng = random.Random(4321)
        shuffled_scenes = sorted(scenes); srng.shuffle(shuffled_scenes)
        val_scenes = set(shuffled_scenes[:args.val_scenes])
        val = [r for r in rows if _scene_of(r) in val_scenes]
        train = [r for r in rows if _scene_of(r) not in val_scenes]
        tr_s = {_scene_of(r) for r in train}; va_s = {_scene_of(r) for r in val}
        assert not (tr_s & va_s), f"scene leak: {sorted(tr_s & va_s)[:5]}"
        print(f"SCENE-HELD-OUT split: {len(tr_s)} train scenes / {len(va_s)} val scenes, "
              f"overlap {len(tr_s & va_s)}")
    else:
        if not args.allow_shared_scenes:
            raise SystemExit(
                "--val-scenes 0 shares scenes between train and eval, which is not "
                "allowed by default. Pass --allow-shared-scenes '<reason>' if an "
                "experiment genuinely needs it.")
        print(f"WARNING shared-scene split, justified as: {args.allow_shared_scenes}")
        val, train = rows[:120], rows[120:]
    dd = DatasetDict({"train": Dataset.from_list(train), "validation": Dataset.from_list(val)})
    dd.save_to_disk(args.out)
    obs = sum(r["num_observations"] for r in rows)
    d = np.concatenate([np.asarray(r["distance_targets"], float) for r in rows])
    print(f"wrote {args.out}: {len(train)} train / {len(val)} val rows, {obs} observations")
    print(f"  distance: mean {d.mean():.2f} m, under 2 m {np.mean(d <= 2):.1%}, under 1 m {np.mean(d <= 1):.1%}")
    print(f"  turns_shuffled={args.shuffle_turns}  probe_only=True (flow loss must be masked by the trainer)")
    # No-perception baselines on the val split, fitted on train. A model that does not
    # beat these is not reading its input; printing them here means the bar travels with
    # the corpus instead of being rediscovered per experiment.
    def _finite(r):
        x = np.asarray(r["distance_targets"], dtype=float)
        return x[np.isfinite(x)]
    tr_all = np.concatenate([_finite(r) for r in train if len(_finite(r))])
    gmed = float(np.median(tr_all))
    per_scene = {}
    for r in train:
        per_scene.setdefault(_scene_of(r), []).append(_finite(r))
    smed = {k: float(np.median(np.concatenate(v))) for k, v in per_scene.items()}
    e_glob, e_scene, e_epi = [], [], []
    for r in val:
        x = _finite(r)
        if not len(x):
            continue
        e_glob.append(np.abs(x - gmed))
        e_scene.append(np.abs(x - smed.get(_scene_of(r), gmed)))
        e_epi.append(np.abs(x - np.median(x)))
    # SHUFFLE INTEGRITY, checked over the corpus rather than per row. This verifies the
    # LABELS are per-frame, not that any shortcut was removed: if they had come from
    # episode structure ("the last turn is the stop"), every shuffled row's positive
    # would still sit at the final position, and the permutation would have moved the
    # frames without moving the labels. Under a real per-frame permutation the positions
    # are uniform.
    #
    # NOT a claim about model cheating: attention is causal, so at turn k the model
    # cannot know whether k is the episode's last turn -- "identify the final frame" is
    # not an available strategy. The cue shuffling actually removes is the ASSOCIATION
    # between elapsed turn count and proximity (the ordered corpus's hazard rises with
    # t), which is the measured clock shortcut: partial corr(prediction, step | true
    # distance) = -0.13.
    if args.shuffle_turns:
        last_share, n_checked = 0, 0
        for r in rows:
            t = np.asarray(r["stop_targets"], dtype=float)
            if t.sum() == 0:
                continue
            n_checked += 1
            last_share += float(t[-1] > 0.5)
        frac = last_share / max(n_checked, 1)
        # P(last slot is positive) under a uniform permutation is k/n for a row with k
        # positives, so the baseline is the mean PER-ROW positive rate -- not 1/n, which
        # assumes a single positive and understates it several-fold.
        expected = float(np.mean([
            float(np.mean(np.asarray(r["stop_targets"], dtype=float))) for r in rows]))
        print(f"  shuffle integrity: positives at the final position in {frac:.3%} of "
              f"rows (expected ~{expected:.3%} under a uniform permutation)")
        if frac > 3 * expected + 0.02:
            raise SystemExit(
                f"stop positives concentrate at the window end ({frac:.1%}); that is the "
                "structural label's signature, not a per-frame one")
    st = np.concatenate([np.asarray(r["stop_targets"], dtype=float) for r in rows])
    print(f"  stop label (d <= {args.stop_radius} m): positive rate {st.mean():.2%} "
          f"({int(st.sum())} of {len(st)} frames) -> suggested pos_weight 10-20 "
          f"(NOT the full {(1-st.mean())/max(st.mean(),1e-9):.0f}:1 ratio: AP is invariant "
          f"to it but calibration is not, and the threshold would sit on a brittle part "
          f"of the curve)")
    if e_glob:
        print(f"  val MAE of no-perception baselines: global const "
              f"{np.concatenate(e_glob).mean():.2f} m | per-scene const "
              f"{np.concatenate(e_scene).mean():.2f} m | per-episode median (cheats) "
              f"{np.concatenate(e_epi).mean():.2f} m")


if __name__ == "__main__":
    main()
