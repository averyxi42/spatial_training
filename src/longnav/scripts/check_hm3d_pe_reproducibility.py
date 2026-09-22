"""Gate two fixed-set HM3D PE evaluations by per-episode outcome agreement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


OUTCOME_FIELDS = ("status", "success", "oracle_success", "terminated_by")


def load_outcomes(path: Path) -> dict[str, tuple[object, ...]]:
    payload = json.loads(path.read_text())
    outcomes = {}
    for result in payload["results"]:
        uid = result["uid"]
        if uid in outcomes:
            raise ValueError(f"duplicate UID in {path}: {uid}")
        outcomes[uid] = tuple(result.get(field) for field in OUTCOME_FIELDS)
    return outcomes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--max-outcome-mismatches", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.max_outcome_mismatches < 0:
        raise ValueError("max-outcome-mismatches must be non-negative")

    first, second = load_outcomes(args.first), load_outcomes(args.second)
    missing_first = sorted(set(second) - set(first))
    missing_second = sorted(set(first) - set(second))
    mismatches = [
        {
            "uid": uid,
            "first": dict(zip(OUTCOME_FIELDS, first[uid])),
            "second": dict(zip(OUTCOME_FIELDS, second[uid])),
        }
        for uid in sorted(set(first) & set(second))
        if first[uid] != second[uid]
    ]
    passed = not missing_first and not missing_second and len(mismatches) <= args.max_outcome_mismatches
    report = {
        "first": str(args.first),
        "second": str(args.second),
        "max_outcome_mismatches": args.max_outcome_mismatches,
        "outcome_mismatch_count": len(mismatches),
        "outcome_mismatches": mismatches,
        "missing_from_first": missing_first,
        "missing_from_second": missing_second,
        "passed": passed,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
