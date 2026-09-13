"""Online RGB-D BEV coverage reward for continuous ObjectNav.

The policy never consumes this map.  It is an environment-side reward instrument built
only from depth observed so far and the robot pose already required for execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Set, Tuple

import numpy as np


Cell = Tuple[int, int]


@dataclass(frozen=True)
class ExplorationGainConfig:
    enabled: bool = False
    reward_weight: float = 0.0
    reward_sigma_m2: float = 1.0
    resolution_m: float = 0.1
    local_window_m: float = 24.0
    ray_range_m: float = 12.0
    n_rays: int = 31
    hfov_deg: float = 79.0
    path_samples: int = 4
    max_depth_m: float = 12.0


def _cells_on_segment(start: Cell, end: Cell) -> Iterable[Cell]:
    """Integer Bresenham cells, including the end point."""
    x0, y0 = start
    x1, y1 = end
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
    err = dx + dy
    while True:
        yield x0, y0
        if x0 == x1 and y0 == y1:
            return
        twice = 2 * err
        if twice >= dy:
            err += dy
            x0 += sx
        if twice <= dx:
            err += dx
            y0 += sy


class PredictedVisibilityGain:
    """Sparse global BEV with locally bounded, path-conditioned ray tracing."""

    def __init__(self, cfg: ExplorationGainConfig):
        self.cfg = cfg
        self._known: Set[Cell] = set()
        self._obstacles: Set[Cell] = set()
        self._credit_count: Dict[Cell, int] = {}

    def reset(self) -> None:
        self._known.clear()
        self._obstacles.clear()
        self._credit_count.clear()

    def reward(self, gain_m2: float) -> float:
        """Bound positive visible-area gain while retaining its small-gain slope."""
        if not self.cfg.enabled or self.cfg.reward_weight <= 0.0:
            return 0.0
        sigma = float(self.cfg.reward_sigma_m2)
        if sigma <= 0.0:
            raise ValueError("exploration reward sigma must be positive")
        gain = max(float(gain_m2), 0.0)
        return float(self.cfg.reward_weight * (-np.expm1(-gain / sigma)))

    def _cell(self, xy: np.ndarray) -> Cell:
        return tuple(np.floor(np.asarray(xy, dtype=np.float64) / self.cfg.resolution_m).astype(int))

    def update_depth(self, depth: np.ndarray, pose: np.ndarray) -> None:
        """Insert a conservative horizontal depth fan into the observed map.

        A robust low quantile across the central image band avoids floor and ceiling
        returns.  It is deliberately a map observation, never a policy input.
        """
        if not self.cfg.enabled:
            return
        d = np.asarray(depth, dtype=np.float32)
        if d.ndim == 3:
            d = d[..., 0]
        if d.ndim != 2 or d.size == 0:
            return
        h, w = d.shape
        band = d[h // 3 : max(h // 3 + 1, 2 * h // 3)]
        theta = float(pose[2])
        columns = np.linspace(0, w - 1, min(w, self.cfg.n_rays), dtype=int)
        angles = np.linspace(-np.deg2rad(self.cfg.hfov_deg) / 2,
                             np.deg2rad(self.cfg.hfov_deg) / 2, len(columns))
        origin = np.asarray(pose[:2], dtype=np.float64)
        origin_cell = self._cell(origin)
        for col, angle in zip(columns, angles):
            values = band[:, col]
            values = values[np.isfinite(values) & (values > 0.05)]
            if not len(values):
                continue
            distance = min(float(np.quantile(values, 0.2)), self.cfg.max_depth_m)
            direction = np.array([np.cos(theta + angle), np.sin(theta + angle)])
            endpoint = self._cell(origin + direction * distance)
            cells = list(_cells_on_segment(origin_cell, endpoint))
            self._known.update(cells)
            if distance < self.cfg.max_depth_m - self.cfg.resolution_m:
                self._obstacles.add(endpoint)

    def predicted_gain(self, cumulative_body_se2: np.ndarray, pose: np.ndarray) -> tuple[float, int]:
        """Credit previously unseen cells predicted visible along a candidate chunk."""
        if not self.cfg.enabled:
            return 0.0, 0
        chunk = np.asarray(cumulative_body_se2, dtype=np.float64).reshape(-1, 3)
        if not len(chunk):
            return 0.0, 0
        picks = np.unique(np.linspace(0, len(chunk) - 1, self.cfg.path_samples, dtype=int))
        base_xy = np.asarray(pose[:2], dtype=np.float64)
        base_theta = float(pose[2])
        rotation = np.array([[np.cos(base_theta), -np.sin(base_theta)],
                             [np.sin(base_theta), np.cos(base_theta)]])
        # A cell can be visible from several sampled poses in one chunk.  Count those
        # overlapping ray hits before granting its one-time credit, so redundant views
        # have diminishing marginal reward while revisiting an already credited cell
        # cannot inflate the reward.
        candidates: Dict[Cell, int] = {}
        half_window = self.cfg.local_window_m / 2.0
        for index in picks:
            local = chunk[index]
            xy = base_xy + rotation @ local[:2]
            heading = base_theta + float(local[2])
            start = self._cell(xy)
            for angle in np.linspace(-np.deg2rad(self.cfg.hfov_deg) / 2,
                                     np.deg2rad(self.cfg.hfov_deg) / 2,
                                     self.cfg.n_rays):
                endpoint = self._cell(xy + self.cfg.ray_range_m * np.array(
                    [np.cos(heading + angle), np.sin(heading + angle)]))
                for cell in _cells_on_segment(start, endpoint):
                    center = (np.asarray(cell, dtype=np.float64) + 0.5) * self.cfg.resolution_m
                    if np.max(np.abs(center - xy)) > half_window:
                        break
                    if cell in self._obstacles:
                        break
                    if cell not in self._known and cell not in self._credit_count:
                        candidates[cell] = candidates.get(cell, 0) + 1
        gain = 0.0
        for cell, overlap_count in candidates.items():
            gain += self.cfg.resolution_m ** 2 / np.sqrt(overlap_count)
            self._credit_count[cell] = overlap_count
        return float(gain), len(candidates)
