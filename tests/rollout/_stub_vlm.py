"""Lightweight stand-in for the VLM side of EpisodeRolloutMixin.run_episode.

Real VLM inference (VLMWorker.infer_probs) needs a loaded 2B-param model --
overkill for testing what happens to *scripted* observations moving through
the rollout loop and _pack_trajectory. This stub implements exactly the
surface EpisodeRolloutMixin.run_episode touches on `self`, with
`infer_probs` replaced by a deterministic, test-controlled policy.
"""
from typing import Any, Dict, List, Optional

import numpy as np

from longnav.utils.rollout_core import EpisodeRolloutMixin

MINIMAL_ROLLOUT_CONFIG: Dict[str, Any] = {
    "max_steps": 16,
    "temperature": 1.0,
    "action_space": ["stop", "forward", "left", "right"],
    "convo_start_template": [
        {"role": "user", "content": [{"type": "text", "text": "goal: $instr_or_goal"}]},
        {"role": "user", "content": [{"type": "image"}]},
    ],
    "convo_turn_template": [
        {"role": "assistant", "content": [{"type": "text", "text": "**$action**"}]},
        {"role": "user", "content": [{"type": "image"}]},
    ],
    "stop_prob_threshold": None,
}


class StubEpisodeWorker(EpisodeRolloutMixin):
    """Drives EpisodeRolloutMixin.run_episode with a scripted policy instead
    of a real VLM forward pass."""

    def __init__(
        self,
        policy_head_type: str,
        action_probs_sequence: Optional[List[np.ndarray]] = None,
        continuous_action_sequence: Optional[List[np.ndarray]] = None,
        rollout_config: Optional[Dict[str, Any]] = None,
    ):
        self.policy_head_config = {"type": policy_head_type}
        self.rollout_config = rollout_config or dict(MINIMAL_ROLLOUT_CONFIG)
        self._action_probs_sequence = action_probs_sequence or []
        self._continuous_action_sequence = continuous_action_sequence or []
        self._call_idx = 0

    def reset(self):
        # VLMWorker.reset() clears cached generation state; nothing to do here.
        self._call_idx = 0

    def infer_probs(self, images, messages, temperature, pos_id_kwargs=None):
        idx = min(self._call_idx, max(len(self._action_probs_sequence), len(self._continuous_action_sequence)) - 1)
        self._call_idx += 1
        if self.policy_head_config["type"] == "continuous":
            action = self._continuous_action_sequence[idx]
            return action.astype(np.float32), np.float32(0.0), None
        probs = self._action_probs_sequence[idx]
        logprobs = np.log(probs + 1e-9)
        return probs, logprobs, None

    def _compute_value(self, outputs):
        import torch

        return torch.tensor([0.0])
