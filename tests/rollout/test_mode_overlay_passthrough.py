"""The mode-overlay pass-through: head fills -> rollout ships -> env receives, video-only.

Exercises the REAL run_episode chain branch with a scripted chain head and a recording
env actor: the `mode_chunks` kwarg reaches `step` exactly when the head filled modes, the
head's slot is read-and-cleared, and with the overlay off the step call is byte-identical
to the historical signature (no kwarg at all -- every other env keeps working).
"""
from typing import Any, Dict, List, Optional

import numpy as np
import pytest
import ray

from _stub_vlm import MINIMAL_ROLLOUT_CONFIG, StubEpisodeWorker

GAP, T3 = 3, 6


class _RecordingEnvActor:
    """Minimal continuous-env surface; records every step call's kwargs."""

    def __init__(self, n_steps: int = 3):
        self.n_steps = int(n_steps)
        self.calls: List[Dict[str, Any]] = []
        self._t = 0

    def reset(self):
        self._t = 0
        return self._obs(False)

    def step(self, action, supplementary_logs: Optional[Dict[str, Any]] = None,
             **kwargs):
        self._t += 1
        self.calls.append({"kwargs": dict(kwargs), "t": self._t})
        return self._obs(self._t >= self.n_steps)

    def get_calls(self):
        return self.calls

    def _obs(self, done):
        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        return rgb, {"obs": {"instr_or_goal": "sofa"}, "reward": 0.0, "done": done,
                     "is_exhausted": False, "info": {}}


class _ScriptedCodeHead:
    """The exact surface run_episode's chain branch touches, with overlay control."""

    def __init__(self, fill_modes: bool):
        self.fill_modes = bool(fill_modes)
        self.last_mode_chunks = None
        self.n_ticks, self.gap = T3, GAP

    def sample_chain_np(self, h, prefix=None):
        assert prefix is None
        if self.fill_modes:
            self.last_mode_chunks = np.full((2, T3, 3), 0.25, dtype=np.float32)
        return (np.array([1.0, 2.0], np.float32), np.zeros(1, np.int64), -1.5,
                np.zeros((GAP, 3), np.float32))


class _ChainStubWorker(StubEpisodeWorker):
    def __init__(self, fill_modes: bool):
        super().__init__(policy_head_type="continuous",
                         rollout_config=dict(MINIMAL_ROLLOUT_CONFIG))
        head = _ScriptedCodeHead(fill_modes)
        self.model = type("_M", (), {})()
        self.model.action_head = head

    def infer_probs(self, images, messages, temperature, pos_id_kwargs=None):
        return {"h": np.zeros(16, np.float32)}, None, None


@pytest.fixture(scope="module")
def local_ray():
    """local_mode: actors run in-process, so a TEST-module actor class needs no worker
    import (the module is not importable from a Ray worker's fresh interpreter)."""
    ray.init(local_mode=True, include_dashboard=False, ignore_reinit_error=True)
    yield
    ray.shutdown()


@pytest.mark.parametrize("fill_modes", [True, False])
def test_mode_chunks_reach_the_env_iff_the_head_filled_them(local_ray, fill_modes):
    env = ray.remote(_RecordingEnvActor).remote(n_steps=3)
    initial = ray.get(env.reset.remote())
    worker = _ChainStubWorker(fill_modes)
    worker.run_episode(env, initial, collect_trajectory=True, compute_value=False)
    calls = ray.get(env.get_calls.remote())
    assert len(calls) == 3
    for c in calls:
        if fill_modes:
            assert "mode_chunks" in c["kwargs"], "head filled modes; env never saw them"
            assert np.asarray(c["kwargs"]["mode_chunks"]).shape == (2, T3, 3)
        else:
            # the historical signature exactly: no kwarg, so envs without the
            # parameter (replay, dummies, habitat) are untouched
            assert c["kwargs"] == {}
    # read-and-clear: nothing stale on the head after the episode
    assert worker.model.action_head.last_mode_chunks is None or fill_modes is False
