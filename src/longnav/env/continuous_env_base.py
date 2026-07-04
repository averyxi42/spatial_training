import random
from typing import Any, Dict, Optional

import numpy as np


def get_continuous_dummy_state() -> Dict[str, Any]:
    return {
        "obs": {
            "instr_or_goal": "drive velocity to zero",
        },
        "done": False,
        "reward": 0.0,
        "is_exhausted": False,
        "info": {},
    }


class ContinuousDummyEnvActor:
    """
    Continuous-action environment for smoke testing the continuous PPO path.

    Action space: R^D vector (default D=2), expected range approximately [-1, 1].
    Reward: negative L2 norm (encourages action vector near zero).
    Episode ends if the action norm is very small or max_steps is reached.
    """

    def __init__(self, action_dim: int = 2, max_steps: int = 32):
        self.action_dim = action_dim
        self.max_steps = max_steps
        self.sc = 0

    def step(self, action: np.ndarray, supplementary_logs: Optional[Dict[str, Any]] = None):
        self.sc += 1
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != self.action_dim:
            raise ValueError(
                f"Expected continuous action dim={self.action_dim}, got {action.shape[0]}"
            )

        rgb = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        norm = float(np.linalg.norm(action))

        state = get_continuous_dummy_state()
        state["reward"] = -norm
        state["done"] = (norm < 0.05) or (self.sc >= self.max_steps) or (random.random() < 0.03)
        state["info"] = {
            "action_norm": norm,
            "action_abs_mean": float(np.mean(np.abs(action))),
        }
        return rgb, state

    def reset(self):
        self.sc = 0
        rgb = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        state = get_continuous_dummy_state()
        state["done"] = False
        return rgb, state

    def assign_shard(self, episodes: Optional[list[str]] = None):
        return None

    def flush_logs_to_disk(self):
        return None

    def is_exhausted(self):
        return False
