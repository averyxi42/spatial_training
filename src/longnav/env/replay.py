"""A scripted, fully-deterministic sim actor for plumbing/shape tests.

Implements the same duck-typed interface as DummyEnvActor/DummyContinuousEnvActor
(src/longnav/env/env_base.py): reset()/step(action) -> (rgb, state_dict) with
state_dict carrying obs/done/reward/is_exhausted/info, plus assign_shard and
flush_logs_to_disk. Unlike the dummy envs, nothing here is random -- every
reset()/step() call replays the next entry of a fixed `script`, so tests can
assert exact trajectory shapes/values without fighting env randomness.

`action` is accepted but ignored: this actor is for testing what happens to
scripted observations moving through rollout/collation code, not for testing
whether an action changes an environment.
"""
from typing import Any, Dict, List, Optional


class ReplayEnvActor:
    def __init__(
        self,
        script: List[Dict[str, Any]],
        logging_output_dir: Optional[str] = None,
        logger_actor: Optional[Any] = None,
        **kwargs,
    ):
        if not script:
            raise ValueError("ReplayEnvActor requires a non-empty script")
        self.script = script
        self.logging_output_dir = logging_output_dir
        self.logger_actor = logger_actor
        self._idx = 0

    def _state_from(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "obs": entry.get("obs", {}),
            "done": entry.get("done", False),
            "reward": entry.get("reward", 0.0),
            "is_exhausted": False,
            "info": entry.get("info", {}),
        }

    def reset(self):
        self._idx = 0
        entry = self.script[self._idx]
        return entry["rgb"], self._state_from(entry)

    def step(self, action, supplementary_logs: Optional[Dict[str, Any]] = None):
        # `action` intentionally ignored -- see module docstring.
        self._idx = min(self._idx + 1, len(self.script) - 1)
        entry = self.script[self._idx]
        return entry["rgb"], self._state_from(entry)

    def assign_shard(self, episodes: Optional[list] = None):
        pass

    def flush_logs_to_disk(self):
        pass

    def is_exhausted(self):
        # No-op matching DummyEnvActor's contract -- shard exhaustion isn't a
        # concept this scripted actor models.
        return False
