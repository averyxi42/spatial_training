"""Shared decoded-trajectory STOP semantics for training and evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np


def trajectory_motion_magnitudes(
    action: Any,
    *,
    prefix_points: int | None = None,
) -> tuple[float, float]:
    """Return cumulative XY and yaw motion from cumulative SE(2) waypoints."""
    values = np.asarray(action, dtype=np.float64)
    if values.ndim == 1:
        if values.size >= 3 and values.size % 3 == 0:
            trajectory = values.reshape(-1, 3)
        elif values.size >= 2:
            trajectory = np.zeros((1, 3), dtype=np.float64)
            trajectory[0, :2] = values[:2]
        else:
            return float("nan"), float("nan")
    elif values.shape[-1] >= 2:
        flattened = values.reshape(-1, values.shape[-1])
        trajectory = np.zeros((len(flattened), 3), dtype=np.float64)
        trajectory[:, :2] = flattened[:, :2]
        if values.shape[-1] >= 3:
            trajectory[:, 2] = flattened[:, 2]
    else:
        return float("nan"), float("nan")
    if prefix_points is not None:
        if prefix_points < 1:
            raise ValueError("prefix_points must be positive")
        trajectory = trajectory[:prefix_points]
    if len(trajectory) == 0 or not np.isfinite(trajectory).all():
        return float("nan"), float("nan")
    anchored = np.vstack((np.zeros((1, 3), dtype=np.float64), trajectory))
    deltas = np.diff(anchored, axis=0)
    return (
        float(np.linalg.norm(deltas[:, :2], axis=1).sum()),
        float(np.abs(deltas[:, 2]).sum()),
    )


def trajectory_stop_decision(
    action: Any,
    *,
    translation_threshold_m: float,
    yaw_threshold_rad: float,
    decision_index: int,
    min_steps: int,
    prefix_points: int | None = None,
) -> tuple[bool, float, float]:
    """Apply the shared action-based STOP rule to an executable chunk prefix."""
    if translation_threshold_m < 0.0 or yaw_threshold_rad < 0.0:
        raise ValueError("trajectory STOP thresholds must be non-negative")
    if min_steps < 0:
        raise ValueError("trajectory_stop_min_steps must be non-negative")
    translation_m, yaw_rad = trajectory_motion_magnitudes(
        action, prefix_points=prefix_points
    )
    stop = bool(
        decision_index >= min_steps
        and translation_m <= translation_threshold_m
        and yaw_rad <= yaw_threshold_rad
    )
    return stop, translation_m, yaw_rad
