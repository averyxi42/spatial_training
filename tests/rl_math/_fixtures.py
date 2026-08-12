"""Small, hand-built synthetic tensors shared across tests/rl_math/*.

Deliberately tiny (3-5 trajectories x 4-8 steps) so shapes/values are easy
to reason about by hand, per the regression-test plan's Phase 4 spec.
"""
import torch

from longnav.config_schema import RLAlgoConfig


def make_rl_algo_config(**overrides) -> RLAlgoConfig:
    cfg = RLAlgoConfig()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def make_reward_batch():
    """3 trajectories x 5 steps. Trajectory lengths vary (padding via
    response_mask): traj 0 uses all 5 steps, traj 1 uses 3, traj 2 uses 4."""
    torch.manual_seed(0)
    token_level_rewards = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, -0.1, -1.0, 0.0, 0.0],
            [0.1, 0.0, 0.0, 0.5, 0.0],
        ],
        dtype=torch.float32,
    )
    response_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0],
        ],
        dtype=torch.long,
    )
    return token_level_rewards, response_mask


def make_distances_for(response_mask):
    """Monotonically-decreasing per-trajectory distance-to-goal, masked
    positions set to 0 (matches how padding is handled elsewhere: distances
    get multiplied by response_mask before binning)."""
    bs, seq_len = response_mask.shape
    base = torch.arange(seq_len, 0, -1, dtype=torch.float32).unsqueeze(0).expand(bs, seq_len).clone()
    return base * response_mask.float()
