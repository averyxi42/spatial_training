#!/usr/bin/env python
"""Join per-frame STOP labels onto a formatted corpus that already carries distances.

`stop_target[i] = distance_targets[i] <= radius`, NaN where the distance is NaN.

Why this and not the structural label: `vector_sft.episode_stop_labels` derives the
positive from the *episode* ending -- one positive per episode, at the last observation,
regardless of where the agent actually is. That label is wrong twice over. It is wrong on
PointNav, whose episodes chain several goals and therefore ARRIVE several times; and it is
wrong at deployment, where the head must fire the moment the agent is close enough, not
when a trajectory happens to end. A distance threshold is the same predicate the harness
scores success with, so the head is trained on the decision it will be asked to make.

The radius MUST match the evaluation success distance. It is written into a
`stop_radius_m` column so a corpus is self-describing and a mismatch is greppable rather
than remembered.

NaN handling is the point, not an edge case: pointnav's distance sidecar covers ~79% of
frames, and a NaN target is masked inside the head (proven zero-gradient in
tests/test_gradient_isolation.py). A frame with no distance yields no stop label rather
than a fabricated negative -- inventing negatives at unlabelled frames is how a stop head
learns never to fire.
"""
import argparse
from datasets import DatasetDict, load_from_disk


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("in_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--radius", type=float, required=True,
                    help="success radius in metres; MUST match the eval "
                         "--success-distance the checkpoint will be scored at")
    ap.add_argument("--distance-column", default="distance_targets")
    ap.add_argument("--num-proc", type=int, default=8)
    args = ap.parse_args()

    ds = load_from_disk(args.in_dir)
    splits = ds if isinstance(ds, DatasetDict) else DatasetDict({"train": ds})
    r = float(args.radius)

    first = next(iter(splits.values()))
    if args.distance_column not in first.column_names:
        raise SystemExit(
            f"{args.in_dir} has no {args.distance_column!r} column; run "
            "add_distance_targets.py first -- stop labels are derived from distance, "
            "never from episode position")
    if "stop_targets" in first.column_names:
        raise SystemExit(
            f"{args.in_dir} already carries stop_targets; refusing to overwrite. "
            "Re-derive from the un-joined corpus if the radius changed.")

    def add(ex):
        d = ex[args.distance_column]
        out = []
        for v in d:
            if v is None or v != v:            # None or NaN
                out.append(float("nan"))
            else:
                out.append(1.0 if float(v) <= r else 0.0)
        return {"stop_targets": out, "stop_radius_m": r}

    done = DatasetDict({name: sp.map(add, num_proc=args.num_proc,
                                     desc=f"stop targets [{name}]")
                        for name, sp in splits.items()})

    for name, sp in done.items():
        n = pos = nan = 0
        for row in sp:
            for v in row["stop_targets"]:
                n += 1
                if v != v:
                    nan += 1
                elif v > 0.5:
                    pos += 1
        lab = n - nan
        rate = pos / lab if lab else float("nan")
        # pos_weight for BCEWithLogits that balances the classes; the head's default of
        # 15.0 was fitted to a corpus that no longer exists.
        pw = (lab - pos) / pos if pos else float("inf")
        print(f"[{name}] frames={n}  labelled={lab} ({lab/n:.1%})  "
              f"positives={pos} ({rate:.2%} of labelled)  "
              f"balanced pos_weight={pw:.2f}")

    done.save_to_disk(args.out_dir)
    print(f"wrote {args.out_dir}: " + ", ".join(f"{k}={len(v)}" for k, v in done.items()))


if __name__ == "__main__":
    main()
