import pytest
import torch

from longnav.utils.rl_core import compute_navigation_metric_advantages


def outcomes():
    return {
        "response_mask": torch.ones(3, 2, dtype=torch.bool),
        "success": torch.tensor([[0., 1.], [0., 0.], [0., 0.]]),
        "categorical_stop_action": torch.tensor([[0., 1.]] * 3),
        "spl_fix": torch.tensor([[0., .5], [0., 0.], [0., 0.]]),
        "oracle_reached": torch.tensor([[1., 1.], [1., 1.], [0., 0.]]),
        "oracle_spl": torch.tensor([[.5, .5], [.8, .8], [0., 0.]]),
    }


def test_reaching_then_false_stop_keeps_motion_credit_separate():
    motion, stop, *_ = compute_navigation_metric_advantages(outcomes(), "separate_metrics")
    assert motion[1, 0] > 0 and stop[1, -1] < 0
    assert stop[0, -1] > 0
    assert motion[2, 0] < 0


def test_better_ospl_increases_motion_credit_without_rewarding_false_stop():
    batch = outcomes()
    before = compute_navigation_metric_advantages(batch, "separate_metrics")
    batch["oracle_spl"][1] += .1
    after = compute_navigation_metric_advantages(batch, "separate_metrics")
    assert after[0][1, 0] > before[0][1, 0]
    torch.testing.assert_close(after[1], before[1])


def test_better_stop_spl_increases_stop_credit_without_changing_motion():
    batch = outcomes()
    before = compute_navigation_metric_advantages(batch, "separate_metrics")
    batch["spl_fix"][0, -1] += .1
    after = compute_navigation_metric_advantages(batch, "separate_metrics")
    assert after[1][0, -1] > before[1][0, -1]
    torch.testing.assert_close(after[0], before[0])


def test_shared_control_uses_identical_advantages_in_both_actors():
    motion, stop, *_ = compute_navigation_metric_advantages(outcomes(), "shared_metrics")
    torch.testing.assert_close(motion, stop)


def test_padding_and_trajectory_length_do_not_change_episode_baseline():
    batch = outcomes()
    before = compute_navigation_metric_advantages(batch, "separate_metrics")
    longer = {k: v.repeat_interleave(3, dim=1) for k, v in batch.items()}
    longer = {k: torch.nn.functional.pad(v, (0, 2)) for k, v in longer.items()}
    after = compute_navigation_metric_advantages(longer, "separate_metrics")
    torch.testing.assert_close(after[0][:, :6:3], before[0])
    torch.testing.assert_close(after[1][:, :6:3], before[1])
    assert not after[0][:, -2:].any()


def test_timeout_proximity_does_not_receive_strict_stop_success_credit():
    batch = outcomes()
    batch["categorical_stop_action"][0, -1] = 0
    _, _, _, stop_returns, _ = compute_navigation_metric_advantages(batch, "separate_metrics")
    assert not stop_returns.any()


def test_invalid_metric_fails_before_optimizer():
    batch = outcomes()
    batch["oracle_spl"][1, -1] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        compute_navigation_metric_advantages(batch, "separate_metrics")


def future_outcomes():
    batch = outcomes()
    batch["oracle_reached_before"] = torch.tensor([[0., 1.], [0., 1.], [0., 0.]])
    batch["oracle_spl_before"] = torch.tensor([[0., .5], [0., .8], [0., 0.]])
    batch["stop_now_utility"] = torch.tensor([[0., 3.5], [0., 0.], [0., 0.]])
    batch["stop_now_success"] = torch.tensor([[0., 1.], [0., 0.], [0., 0.]])
    batch["stop_now_spl"] = torch.tensor([[0., .5], [0., 0.], [0., 0.]])
    return batch


@pytest.mark.parametrize("mode", ["separate_future", "shared_future",
                                "separate_stop_baseline", "shared_stop_baseline"])
def test_movement_after_reach_has_no_credit_in_any_mode(mode):
    batch = future_outcomes()
    result = compute_navigation_metric_advantages(batch, mode)
    assert result[0][0, 1] == result[0][1, 1] == 0
    assert result[0][2, 0] < 0
    assert result[2][1, -1] == 0


@pytest.mark.parametrize("mode", ["separate_future", "shared_future",
                                "separate_stop_baseline", "shared_stop_baseline"])
def test_starting_inside_goal_has_no_movement_credit(mode):
    batch = future_outcomes()
    batch["oracle_reached_before"][0] = 1
    batch["oracle_spl_before"][0] = .5
    result = compute_navigation_metric_advantages(batch, mode)
    assert not result[0][0].any()


@pytest.mark.parametrize("mode", ["separate_stop_baseline", "shared_stop_baseline"])
def test_stop_baseline_zeroes_stop_and_penalizes_worse_near_continue(mode):
    batch = future_outcomes()
    batch["oracle_reached_before"][0] = 1
    batch["oracle_spl_before"][0] = .5
    batch["stop_now_utility"][0, 0] = 3.8
    batch["stop_now_success"][0, 0] = 1
    batch["stop_now_spl"][0, 0] = .8
    result = compute_navigation_metric_advantages(batch, mode)
    assert result[1][0, 0] < 0
    torch.testing.assert_close(result[1][:, -1], torch.zeros(3))
    torch.testing.assert_close(result[0], compute_navigation_metric_advantages(
        batch, "shared_future" if mode.startswith("shared") else "separate_future")[0])


@pytest.mark.parametrize("mode", ["separate_future", "shared_future",
                                "separate_stop_baseline", "shared_stop_baseline"])
def test_stop_credit_uses_only_terminal_score_against_physical_stop_now(mode):
    batch = future_outcomes()
    result = compute_navigation_metric_advantages(batch, mode)
    # Episode 1 reaches later but never makes a successful STOP.
    assert result[1][1, 0] == 0
    # Continuing after goal and then failing is worse than the recorded immediate STOP.
    batch["oracle_reached_before"][0, 0] = 1
    batch["oracle_spl_before"][0, 0] = .5
    batch["stop_now_success"][0, 0] = 1
    batch["stop_now_spl"][0, 0] = .8
    batch["stop_now_utility"][0, 0] = 3.8
    batch["categorical_stop_action"][0, -1] = 0
    result = compute_navigation_metric_advantages(batch, mode)
    assert result[1][0, 0] < 0


@pytest.mark.parametrize("mode", ["separate_future", "shared_future",
                                "separate_stop_baseline", "shared_stop_baseline"])
def test_future_timeout_never_receives_terminal_success_credit(mode):
    batch = future_outcomes()
    batch["categorical_stop_action"][0, -1] = 0
    batch["stop_now_utility"][0, -1] = 3.5
    result = compute_navigation_metric_advantages(batch, mode)
    assert result[3][0, -1] == 0
    assert result[1][0, -1] <= 0
    if mode.endswith("stop_baseline"):
        assert result[1][0, -1] < 0


def test_missing_or_invalid_pre_action_metrics_fail_closed():
    with pytest.raises(KeyError, match="oracle_reached_before"):
        compute_navigation_metric_advantages(outcomes(), "separate_future")
    batch = future_outcomes()
    batch["oracle_spl_before"][2, 0] = .1
    with pytest.raises(ValueError, match="decreased"):
        compute_navigation_metric_advantages(batch, "shared_future")
    batch["oracle_spl_before"][2, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        compute_navigation_metric_advantages(batch, "shared_future")


def test_double_precision_telemetry_does_not_create_stop_credit_from_roundoff():
    batch = {key: value.double() for key, value in future_outcomes().items()}
    batch["oracle_spl"][0] = .7620661762433102
    batch["oracle_spl_before"][0, -1] = .7620661762433102
    batch["spl_fix"][0, -1] = .6960951799736825
    batch["stop_now_spl"][0, -1] = .6960951799736825
    batch["stop_now_utility"][0, -1] = 3.6960951799736828
    for mode in ("separate_stop_baseline", "shared_stop_baseline"):
        result = compute_navigation_metric_advantages(batch, mode)
        assert result[1].dtype == torch.float32
        assert result[1][0, -1] == 0
