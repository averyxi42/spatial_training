import numpy as np

from longnav.utils.exploration_gain import ExplorationGainConfig, PredictedVisibilityGain


def test_predicted_visibility_credit_is_finite_and_single_use():
    tracker = PredictedVisibilityGain(ExplorationGainConfig(enabled=True, n_rays=5))
    depth = np.full((12, 16), 4.0, dtype=np.float32)
    tracker.update_depth(depth, np.array([0.0, 0.0, 0.0]))
    chunk = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    first, cells = tracker.predicted_gain(chunk, np.array([0.0, 0.0, 0.0]))
    second, _ = tracker.predicted_gain(chunk, np.array([0.0, 0.0, 0.0]))
    assert np.isfinite(first)
    assert cells > 0
    assert first > 0.0
    assert second == 0.0


def test_exploration_reward_is_positive_bounded_and_saturating():
    tracker = PredictedVisibilityGain(ExplorationGainConfig(
        enabled=True, reward_weight=1.25, reward_sigma_m2=31.8,
    ))
    low = tracker.reward(2.3)
    high = tracker.reward(22.0)
    assert 0.0 < low < high < 1.25
    assert np.isclose(high, 0.625, atol=0.01)
