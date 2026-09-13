"""Summarize full-history STOP traces and choose a calibration-only threshold."""
import argparse
import json
import math

import numpy as np


def _load(path):
    with open(path) as file:
        return [json.loads(line) for line in file if line.strip()]


def _finite_points(rows):
    probabilities, targets = [], []
    for row in rows:
        for point in row.get("trace", []):
            probability = point.get("probe_p_stop")
            target = point.get("stop_target")
            if probability is None or target is None:
                continue
            if math.isfinite(float(probability)) and math.isfinite(float(target)):
                probabilities.append(float(probability))
                targets.append(float(target) > 0.5)
    return np.asarray(probabilities), np.asarray(targets, dtype=bool)


def _auc(probabilities, targets):
    positives, negatives = probabilities[targets], probabilities[~targets]
    if not len(positives) or not len(negatives):
        return float("nan")
    comparison = positives[:, None] - negatives[None, :]
    return float((comparison > 0).mean() + 0.5 * (comparison == 0).mean())


def _average_precision(probabilities, targets):
    if not targets.any():
        return float("nan")
    order = np.argsort(-probabilities, kind="stable")
    ranked = targets[order]
    precision = np.cumsum(ranked) / (np.arange(len(ranked)) + 1)
    return float(precision[ranked].mean())


def _ece(probabilities, targets, bins=10):
    values = []
    for lo, hi in zip(np.linspace(0, 1, bins, endpoint=False), np.linspace(0, 1, bins + 1)[1:]):
        mask = (probabilities >= lo) & (probabilities < hi if hi < 1 else probabilities <= hi)
        if mask.any():
            values.append(mask.mean() * abs(probabilities[mask].mean() - targets[mask].mean()))
    return float(sum(values))


def _first_stop_metrics(rows, threshold):
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    lags, categories = [], {
        "navigation_failure": 0,
        "reach_no_stop": 0,
        "premature_stop": 0,
        "reach_then_leave": 0,
        "correct_first_stop": 0,
    }
    for row in rows:
        points = [point for point in row.get("trace", []) if point.get("probe_p_stop") is not None
                  and point.get("stop_target") is not None
                  and math.isfinite(float(point["probe_p_stop"]))
                  and math.isfinite(float(point["stop_target"]))]
        targets = np.asarray([float(point["stop_target"]) > 0.5 for point in points], dtype=bool)
        predictions = np.asarray([float(point["probe_p_stop"]) >= threshold for point in points], dtype=bool)
        first_reach = int(np.flatnonzero(targets)[0]) if targets.any() else None
        first_stop = int(np.flatnonzero(predictions)[0]) if predictions.any() else None
        if first_stop is None:
            if first_reach is None:
                counts["tn"] += 1
                categories["navigation_failure"] += 1
            else:
                counts["fn"] += 1
                categories["reach_no_stop"] += 1
            continue
        if targets[first_stop]:
            counts["tp"] += 1
            categories["correct_first_stop"] += 1
            if first_reach is not None:
                lags.append(first_stop - first_reach)
        else:
            counts["fp"] += 1
            if first_reach is None or first_stop < first_reach:
                categories["premature_stop"] += 1
            else:
                categories["reach_then_leave"] += 1
    precision = counts["tp"] / max(counts["tp"] + counts["fp"], 1)
    recall = counts["tp"] / max(counts["tp"] + counts["fn"], 1)
    return {
        **counts,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "utility_tp_minus_fp": counts["tp"] - counts["fp"],
        "mean_first_stop_lag": float(np.mean(lags)) if lags else None,
        "episode_categories": categories,
    }


def summarize(rows):
    probabilities, targets = _finite_points(rows)
    return {
        "episodes": len(rows),
        "steps": int(len(targets)),
        "positive_steps": int(targets.sum()),
        "auroc": _auc(probabilities, targets),
        "average_precision": _average_precision(probabilities, targets),
        "brier": float(np.square(probabilities - targets).mean()) if len(targets) else float("nan"),
        "ece_10": _ece(probabilities, targets) if len(targets) else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--select-threshold", action="store_true")
    args = parser.parse_args()

    rows = _load(args.input)
    result = summarize(rows)
    if args.select_threshold:
        candidates = np.linspace(0.05, 0.99, 95)
        scored = [(float(threshold), _first_stop_metrics(rows, float(threshold)))
                  for threshold in candidates]
        valid = [item for item in scored if item[1]["tp"] + item[1]["fp"] > 0]
        if not valid:
            raise RuntimeError("no finite stop probabilities in calibration trace")
        threshold, selected = max(valid, key=lambda item: (
            item[1]["utility_tp_minus_fp"], item[1]["f1"], item[1]["precision"]
        ))
        result["selected_threshold"] = threshold
        result["selection_metrics"] = selected
    if args.threshold is not None:
        result["threshold"] = args.threshold
        result["first_stop_metrics"] = _first_stop_metrics(rows, args.threshold)
    with open(args.output, "w") as file:
        json.dump(result, file, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
