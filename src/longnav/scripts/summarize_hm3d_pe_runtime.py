"""Summarize runtime probes emitted by an HM3D PE benchmark output directory.

The benchmark stores per-step policy, controller, and total wall time in each
``episode_details/*.json.gz`` file. This utility creates one compact JSON
report suitable for a checkpoint-gate log; it does not modify benchmark data.

Usage:
    python -m longnav.scripts.summarize_hm3d_pe_runtime \\
        --eval-dir /path/to/hm3d_pe_eval32 --wall-seconds 329
"""

import argparse
import gzip
import json
import math
from pathlib import Path


TIMING_KEYS = ("policy_latency_s", "execution_wall_s", "total_wall_s")


def quantiles(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    if not ordered:
        return {"count": 0}

    def percentile(p: float) -> float:
        index = (len(ordered) - 1) * p
        low, high = math.floor(index), math.ceil(index)
        if low == high:
            return ordered[low]
        return ordered[low] + (ordered[high] - ordered[low]) * (index - low)

    return {
        "count": len(ordered),
        "sum_s": round(sum(ordered), 6),
        "mean_s": round(sum(ordered) / len(ordered), 6),
        "p50_s": round(percentile(0.50), 6),
        "p95_s": round(percentile(0.95), 6),
        "max_s": round(ordered[-1], 6),
    }


def summarize(eval_dir: Path, wall_seconds: float | None) -> dict:
    details_dir = eval_dir / "episode_details"
    if not details_dir.is_dir():
        raise FileNotFoundError(f"missing episode details directory: {details_dir}")

    timings = {key: [] for key in TIMING_KEYS}
    episode_totals = []
    for path in sorted(details_dir.glob("*.json.gz")):
        with gzip.open(path, "rt") as handle:
            record = json.load(handle)
        step_timing = record.get("step_timing", {})
        total = 0.0
        for key in TIMING_KEYS:
            values = [float(value) for value in step_timing.get(key, [])]
            timings[key].extend(values)
            if key == "total_wall_s":
                total = sum(values)
        episode_totals.append(total)

    report = {
        "eval_dir": str(eval_dir),
        "episodes": len(episode_totals),
        "step_timing": {key: quantiles(values) for key, values in timings.items()},
        "episode_total_wall": quantiles(episode_totals),
    }
    if wall_seconds is not None:
        report["benchmark_wall_seconds"] = round(wall_seconds, 6)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--wall-seconds", type=float)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    report = summarize(args.eval_dir, args.wall_seconds)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
