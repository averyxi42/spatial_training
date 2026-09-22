"""Refuse a checkpoint gate unless two HM3D PE runs have identical semantics.

Wall-clock timings and generated-file paths are intentionally excluded. Every
other aggregate and per-episode field in ``result.json`` must match exactly.
"""

import argparse
import json
from pathlib import Path
from typing import Any


NON_SEMANTIC_KEYS = {
    "details_file",
    "mean_policy_latency_s",
    "policy_runtime_s",
    "trajectory_file",
    "video",
    "video_bev",
    "video_frames",
    "wall_seconds",
}


def normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: normalize(item)
            for key, item in value.items()
            if key not in NON_SEMANTIC_KEYS
        }
    if isinstance(value, list):
        return [normalize(item) for item in value]
    return value


def load(path: Path) -> dict:
    with path.open() as handle:
        payload = json.load(handle)
    return normalize({key: payload[key] for key in ("summary", "results", "skipped")})


def first_difference(left: Any, right: Any, path: str = "") -> str | None:
    if type(left) is not type(right):
        return f"{path}: type {type(left).__name__} != {type(right).__name__}"
    if isinstance(left, dict):
        if left.keys() != right.keys():
            return f"{path}: keys {sorted(left)} != {sorted(right)}"
        for key in left:
            difference = first_difference(left[key], right[key], f"{path}.{key}")
            if difference is not None:
                return difference
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return f"{path}: length {len(left)} != {len(right)}"
        for index, (item_left, item_right) in enumerate(zip(left, right)):
            difference = first_difference(item_left, item_right, f"{path}[{index}]")
            if difference is not None:
                return difference
        return None
    if left != right:
        return f"{path}: {left!r} != {right!r}"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    left, right = load(args.first), load(args.second)
    difference = first_difference(left, right, "result")
    report = {
        "first": str(args.first),
        "second": str(args.second),
        "identical": difference is None,
        "first_difference": difference,
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if difference is not None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
