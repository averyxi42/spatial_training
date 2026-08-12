"""Continuous ObjectNav as a Ray env actor: one policy step = one action chunk.

Design: `docs/LATENT_RL_ENV.md`. Runs under `habitat_conda_env` (`vln`), which already has
`habitat_sim`, `habitat`, `continuous_demos` and `objectnav_eval` importable.

BUILT ON THE INSTALLED `continuous_demos` / `objectnav_eval`, NOT ON A COPY. The forked
`longnav/env/sim.py` was deleted for this reason: it had drifted 174 lines behind and was
missing scene switching, whose old form failed *silently* -- render, navmesh and spawn snap
all still answering for the previous scene. Episode sourcing, screening with reason codes,
per-dataset navmesh filename resolution, the live-height geodesic adapter and the metrics all
live in `objectnav_eval` because each of them, done wrong, produces a plausible wrong number
rather than an error. The reward depends on every one of them: "geodesic progress" means
nothing unless it is measured on the robot's navmesh against a goal on a reachable island.

THE ACTION IS A CHUNK, AND ONLY ITS PREFIX RUNS. `step(chunk)` executes the first `gap` rows
and discards the rest, exactly as training and the eval executor do -- the remainder would be
superseded by the next observation. Inference is synchronous and assumed instantaneous. There
is deliberately no latency masking and no prefix conditioning here; both are real future work
(`habitat_physical_nav/docs/LATENCY_MASKING.md`) and a stub that quietly did nothing would be
worse than their absence.

REWARD: geodesic progress, with NO TERMINAL TERM. The policy has no stop head, so a success
reward would optimise the termination heuristic rather than navigation. Termination is the
env's: goal reached, budget exhausted, or the robot escaped the world.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


class ContinuousObjectNavEnvActor:
    """The five-method env-actor interface `DummyEnvActor` defines, over a real robot.

    `reset()` / `step(action)` return `(rgb, state_dict)` where `state_dict` carries `obs`,
    `reward`, `done`, `is_exhausted` and `info`, matching what `rollout_core` consumes.
    """

    def __init__(
        self,
        episodes: str,
        scene_root: str,
        gap: int = 10,
        dt: float = 0.04,
        max_steps: int = 175,
        success_distance: float = 1.0,
        navmesh: str = "dataset",
        distance_to: str = "VIEW_POINTS",
        slack_penalty: float = 0.0,
        collision_penalty: float = 0.0,
        seed: int = 0,
        logging_output_dir: Optional[str] = None,
        logger_actor: Any = None,
        **kwargs: Any,
    ):
        self.cfg = dict(
            episodes=episodes, scene_root=scene_root, gap=int(gap), dt=float(dt),
            max_steps=int(max_steps), success_distance=float(success_distance),
            navmesh=navmesh, distance_to=distance_to, seed=int(seed),
        )
        self.slack_penalty = float(slack_penalty)
        self.collision_penalty = float(collision_penalty)
        self.logging_output_dir = logging_output_dir
        self.logger_actor = logger_actor

        self._shard: Optional[List[str]] = None
        self._episode_iter = None
        self._episode = None
        self._steps = 0
        self._prev_geodesic: Optional[float] = None
        self._sim = None
        self._task = None
        self._executor = None

    # -- interface ---------------------------------------------------------------------
    def assign_shard(self, episodes: Optional[List[str]] = None) -> None:
        """Take a slice of the split. `None` means every episode this actor can see."""
        self._shard = list(episodes) if episodes is not None else None
        self._episode_iter = None

    def is_exhausted(self) -> bool:
        return self._episode_iter is not None and self._exhausted

    def flush_logs_to_disk(self):
        return None

    def reset(self):
        self._ensure_built()
        self._episode = self._next_episode()
        self._steps = 0
        self._task.reset(self._episode)
        self._prev_geodesic = self._geodesic()
        rgb = self._observe()
        return rgb, {
            "obs": {"instr_or_goal": self._episode.object_category},
            "reward": 0.0,
            "done": False,
            "is_exhausted": self.is_exhausted(),
            "info": {"episode_label": str(getattr(self._episode, "episode_id", "?")),
                     "distance_to_goal": self._prev_geodesic},
        }

    def step(self, action, supplementary_logs: Optional[Dict[str, Any]] = None):
        """`action` is the `(gap, 3)` chunk prefix already selected by the policy head.

        The head truncates; this asserts rather than truncating again, so a mismatch between
        the head's `gap` and the env's is a loud error instead of a silently shorter step
        that would make sim time and policy steps disagree.
        """
        chunk = np.asarray(action, dtype=np.float64).reshape(-1, 3)
        if len(chunk) != self.cfg["gap"]:
            raise ValueError(
                f"env gap={self.cfg['gap']} but received a {len(chunk)}-row chunk. The "
                "policy head and the env must agree on ticks-per-step, or sim time and "
                "policy steps quietly mean different things."
            )
        collided = self._execute(chunk)
        self._steps += 1

        geodesic = self._geodesic()
        # Shaped reward: reduction in geodesic distance since the last policy step. No
        # terminal bonus -- see the module docstring.
        progress = (self._prev_geodesic - geodesic
                    if np.isfinite(geodesic) and np.isfinite(self._prev_geodesic) else 0.0)
        reward = progress - self.slack_penalty - (self.collision_penalty if collided else 0.0)
        self._prev_geodesic = geodesic

        reached = np.isfinite(geodesic) and geodesic <= self.cfg["success_distance"]
        done = bool(reached or self._steps >= self.cfg["max_steps"] or self._escaped())
        return self._observe(), {
            "obs": {"instr_or_goal": self._episode.object_category},
            "reward": float(reward),
            "done": done,
            "is_exhausted": self.is_exhausted(),
            "info": {
                "distance_to_goal": float(geodesic) if np.isfinite(geodesic) else None,
                "success": bool(reached),
                "steps": self._steps,
                "collided": bool(collided),
            },
        }

    # -- the parts that touch the simulator ---------------------------------------------
    # Deliberately thin and separated: everything above is testable against a stub, and only
    # these need habitat. They are the pieces to fill in against `objectnav_eval`'s runner,
    # which already owns spawn, screening, navmesh selection and the executor.
    def _ensure_built(self):
        raise NotImplementedError(
            "wire to objectnav_eval: build the sim + RobotObjectNavTask + chunk executor "
            "from self.cfg, reusing objectnav_eval.navmesh / .screening / .task"
        )

    def _next_episode(self):
        raise NotImplementedError("wire to objectnav_eval.episodes.EpisodeSource")

    def _observe(self) -> np.ndarray:
        raise NotImplementedError("robot_sim.get_obs() -> the configured sensor's rgb")

    def _execute(self, chunk: np.ndarray) -> bool:
        raise NotImplementedError(
            "track the chunk for gap ticks via objectnav_eval.executor; return whether the "
            "step collided"
        )

    def _geodesic(self) -> float:
        raise NotImplementedError("task.evaluate()'s distance_to_goal, live-height adapter")

    def _escaped(self) -> bool:
        raise NotImplementedError("continuous_demos.drive_failure's escape check")

    @property
    def _exhausted(self) -> bool:
        return False
