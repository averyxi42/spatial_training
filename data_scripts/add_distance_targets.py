"""Join the distance sidecar into an EXISTING formatted (or episode) dataset, in place of
a full rebuild: rows already record `episode_id` and `frame_indices` (the realized anchor
frames), so the per-observation distances are an exact gather -- no window re-derivation,
no spacing-rng reproduction, originals untouched (output goes to a new dir).

Emits the same three columns the format script would have
(`distance_targets` / `return_targets` / per-row `return_gamma`), via
`state_probe.distance_return_targets`. Refuses a missing sidecar episode (the sidecar
was built to 100% coverage; a miss means the wrong dir).

Usage:
  python data_scripts/add_distance_targets.py \
      --in-dir data/continuous_sft_formatted --out-dir data/continuous_sft_formatted_dist097 \
      --sidecar data/annotations/v2_25hz_distance_to_goal --return-gamma 0.97
"""
import argparse
import json
from pathlib import Path

from datasets import DatasetDict, load_from_disk

from longnav.utils.state_probe import distance_return_targets


def load_sidecar(sidecar_dir: Path):
    lookup = {}
    for shard in sorted(sidecar_dir.glob("*.jsonl")):
        with open(shard) as f:
            for line in f:
                r = json.loads(line)
                # keep JSON nulls as None; the consumer maps them to NaN
                lookup[r["episode_id"]] = r["distance_to_goal"]
    return lookup


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--sidecar", required=True)
    p.add_argument("--return-gamma", type=float, required=True)
    p.add_argument("--reward-clip", type=float, default=0.75)
    p.add_argument("--num-proc", type=int, default=8)
    p.add_argument("--allow-missing", action="store_true",
                   help="episodes absent from the sidecar get all-NaN distance targets "
                        "and all-NaN returns (masked by the probe loss) instead of a "
                        "hard error. For sidecars with validated-refusal gaps (e.g. the "
                        "PointNav island-checksum refusals); the fill count is printed "
                        "and silence would misrepresent coverage, so it also refuses "
                        "if MORE than half the rows are missing.")
    args = p.parse_args()
    out = Path(args.out_dir)
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    lookup = load_sidecar(Path(args.sidecar))
    print(f"sidecar: {len(lookup)} episodes")
    ds = load_from_disk(args.in_dir)
    splits = ds if isinstance(ds, DatasetDict) else DatasetDict({"train": ds})
    g, clip = float(args.return_gamma), float(args.reward_clip)

    def add(ex):
        series = lookup.get(ex["episode_id"])
        if series is None:
            if not args.allow_missing:
                raise KeyError(f"sidecar has no episode {ex['episode_id']}; wrong --sidecar?")
            n = len(ex["frame_indices"])
            return {"distance_targets": [float("nan")] * n,
                    "return_targets": [float("nan")] * n,
                    "return_gamma": g}
        d = [series[i] for i in ex["frame_indices"]]
        return {
            "distance_targets": [float("nan") if v is None else float(v) for v in d],
            "return_targets": distance_return_targets(d, gamma=g, reward_clip=clip),
            "return_gamma": g,
        }

    done = DatasetDict({name: split.map(add, num_proc=args.num_proc,
                                        desc=f"distance/return targets [{name}]")
                        for name, split in splits.items()})
    if args.allow_missing:
        for name, split in done.items():
            miss = sum(1 for r in split if r["episode_id"] not in lookup)
            print(f"  [{name}] NaN-filled rows (missing from sidecar): {miss}/{len(split)}")
            if miss > len(split) / 2:
                raise SystemExit("more than half the rows are missing from the sidecar; "
                                 "this is the wrong sidecar, not a coverage gap")
    done.save_to_disk(str(out))
    print(f"wrote {out}: " + ", ".join(f"{k}={len(v)}" for k, v in done.items()))


if __name__ == "__main__":
    main()
