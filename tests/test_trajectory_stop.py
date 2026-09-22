import numpy as np
import pytest

from longnav.utils.trajectory_stop import trajectory_stop_decision


def test_stop_uses_only_the_executable_prefix():
    action = np.vstack((np.zeros((10, 3)), [[1.0, 0.0, 0.0]])).astype(np.float32)
    stop, translation, yaw = trajectory_stop_decision(
        action,
        translation_threshold_m=0.05,
        yaw_threshold_rad=0.05,
        decision_index=30,
        min_steps=30,
        prefix_points=10,
    )
    assert stop
    assert translation == 0.0
    assert yaw == 0.0


def test_stop_checks_yaw_and_initial_guard():
    action = np.asarray([[0.0, 0.0, 0.06]], dtype=np.float32)
    stop, _, yaw = trajectory_stop_decision(
        action,
        translation_threshold_m=0.05,
        yaw_threshold_rad=0.05,
        decision_index=30,
        min_steps=30,
        prefix_points=10,
    )
    assert not stop
    assert yaw == pytest.approx(0.06)
    guarded, _, _ = trajectory_stop_decision(
        np.zeros((10, 3), dtype=np.float32),
        translation_threshold_m=0.05,
        yaw_threshold_rad=0.05,
        decision_index=29,
        min_steps=30,
        prefix_points=10,
    )
    assert not guarded
