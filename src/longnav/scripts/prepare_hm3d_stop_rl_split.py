"""Create reproducible disjoint HM3D train/eval UID files for stop-head RL."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict, deque
from pathlib import Path

import numpy as np


def balanced_take(groups, count, rng):
    """Round-robin sample categories without replacement after a seeded shuffle."""
    queues = {}
    for category, values in sorted(groups.items()):
        copied = list(values)
        rng.shuffle(copied)
        queues[category] = deque(copied)
    chosen = []
    while len(chosen) < count:
        progressed = False
        for category in sorted(queues):
            if queues[category] and len(chosen) < count:
                chosen.append(queues[category].popleft())
                progressed = True
        if not progressed:
            raise ValueError(f"requested {count} UIDs from only {len(chosen)} eligible episodes")
    remaining = {category: list(values) for category, values in queues.items()}
    return chosen, remaining


def write_uids(path: Path, uids):
    path.write_text("\n".join(uids) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-count", type=int, default=128)
    parser.add_argument("--eval-count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exclude-category", action="append", default=["plant"])
    args = parser.parse_args()

    from objectnav_eval.episodes import ObjectNavFileSource

    excluded = {value.strip().lower() for value in args.exclude_category}
    grouped = defaultdict(list)
    for episode in ObjectNavFileSource(args.episodes).load():
        category = (episode.object_category or "").strip().lower()
        if category and category not in excluded:
            grouped[category].append(episode.uid)
    rng = np.random.default_rng(args.seed)
    eval_uids, remaining = balanced_take(grouped, args.eval_count, rng)
    train_uids, _ = balanced_take(remaining, args.train_count, rng)
    if set(train_uids) & set(eval_uids):
        raise AssertionError("train/eval UID overlap")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_hash = write_uids(output_dir / "train128.txt", train_uids)
    eval_hash = write_uids(output_dir / "eval32.txt", eval_uids)
    manifest = {
        "episodes": str(Path(args.episodes).resolve()),
        "seed": args.seed,
        "excluded_categories": sorted(excluded),
        "train_count": len(train_uids),
        "eval_count": len(eval_uids),
        "train_sha256": train_hash,
        "eval_sha256": eval_hash,
        "category_counts": {
            category: len(values) for category, values in sorted(grouped.items())
        },
        "train_categories": {
            category: sum(uid in train_uids for uid in values)
            for category, values in sorted(grouped.items())
        },
        "eval_categories": {
            category: sum(uid in eval_uids for uid in values)
            for category, values in sorted(grouped.items())
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
