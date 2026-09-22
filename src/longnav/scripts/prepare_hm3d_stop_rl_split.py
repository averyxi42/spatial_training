"""Create reproducible disjoint HM3D train/eval UID files for stop-head RL."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
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


def scene_balanced_take(episodes, count, seed):
    rng = random.Random(seed)
    groups = defaultdict(lambda: defaultdict(list))
    by_uid = {episode.uid: episode for episode in episodes}
    for episode in episodes:
        groups[episode.object_category][episode.uid.split(":", 1)[0]].append(episode.uid)
    for scenes in groups.values():
        for values in scenes.values():
            rng.shuffle(values)
    scene_counts = Counter()
    selected = []
    while len(selected) < count:
        previous = len(selected)
        for category, scenes in sorted(groups.items()):
            available = [scene for scene, values in scenes.items() if values]
            if available and len(selected) < count:
                minimum = min(scene_counts[scene] for scene in available)
                scene = rng.choice(sorted(scene for scene in available if scene_counts[scene] == minimum))
                uid = scenes[scene].pop()
                selected.append(uid)
                scene_counts[scene] += 1
        if len(selected) == previous:
            raise ValueError(f"only {len(selected)} eligible episodes for requested {count}")
    return selected, by_uid


def write_scene_splits(episodes, benchmark, output_dir, seed):
    scenes = sorted({episode.uid.split(":", 1)[0] for episode in episodes})
    if len(scenes) != 80:
        raise ValueError(f"expected 80 training scenes, got {len(scenes)}")
    random.Random(seed).shuffle(scenes)
    partitions = {"train": scenes[:64], "calibration": scenes[64:72], "dev": scenes[72:]}
    benchmark_scenes = {episode.uid.split(":", 1)[0] for episode in benchmark}
    if set(scenes) & benchmark_scenes:
        raise ValueError("train source and benchmark scenes overlap")
    manifest = {"seed": seed, "scene_partitions": partitions, "splits": {}}
    all_uids = {}
    for name, partition in partitions.items():
        pool = [episode for episode in episodes if episode.uid.split(":", 1)[0] in partition]
        selected, by_uid = scene_balanced_take(pool, 512 if name == "train" else 128, seed)
        for count in ([256, 512] if name == "train" else [128]):
            uids = selected[:count]
            key = f"{name}{count}"
            category_counts = Counter(by_uid[uid].object_category for uid in uids)
            selected_scenes = Counter(uid.split(":", 1)[0] for uid in uids)
            if len(category_counts) != 6 or set(selected_scenes) != set(partition):
                raise ValueError(f"{key} lacks category or scene coverage")
            all_uids[key] = set(uids)
            manifest["splits"][key] = {
                "count": len(uids), "sha256": write_uids(output_dir / f"{key}.txt", uids),
                "categories": dict(category_counts), "scenes": dict(sorted(selected_scenes.items())),
            }
    assert all_uids["train256"] < all_uids["train512"]
    assert not (all_uids["train512"] & all_uids["calibration128"]
                or all_uids["train512"] & all_uids["dev128"]
                or all_uids["calibration128"] & all_uids["dev128"])
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-count", type=int, default=128)
    parser.add_argument("--eval-count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exclude-category", action="append", default=["plant"])
    parser.add_argument("--scene-disjoint", action="store_true")
    parser.add_argument("--benchmark-episodes")
    args = parser.parse_args()

    from objectnav_eval.episodes import ObjectNavFileSource

    if args.scene_disjoint:
        if not args.benchmark_episodes:
            parser.error("--scene-disjoint requires --benchmark-episodes")
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if (output_dir / "manifest.json").exists():
            raise FileExistsError(output_dir / "manifest.json")
        manifest = write_scene_splits(
            ObjectNavFileSource(args.episodes).load(),
            ObjectNavFileSource(args.benchmark_episodes).load(), output_dir, args.seed)
        manifest.update(episodes=str(Path(args.episodes).resolve()),
                        benchmark_episodes=str(Path(args.benchmark_episodes).resolve()))
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps({key: {k: v for k, v in value.items() if k != "scenes"}
                          for key, value in manifest["splits"].items()}, indent=2))
        return

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
    train_hash = write_uids(output_dir / f"train{args.train_count}.txt", train_uids)
    eval_hash = write_uids(output_dir / f"eval{args.eval_count}.txt", eval_uids)
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
