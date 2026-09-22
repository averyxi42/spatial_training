import math

import torch

from longnav.utils.logging_workers import WandbLoggerActor
from longnav.utils.train_loop import aggregate_cycle_metrics


class _Run:
    def __init__(self):
        self.definitions = []
        self.logged = []

    def define_metric(self, *args, **kwargs):
        self.definitions.append((args, kwargs))

    def log(self, payload):
        self.logged.append(payload)

    def finish(self):
        pass


class _Table:
    def __init__(self, columns, **_kwargs):
        self.columns = columns
        self.data = []

    def add_data(self, *values):
        self.data.append(values)


def test_compact_logger_uses_cycle_axis_and_keeps_episode_scalars_out_of_history(
    monkeypatch,
):
    run = _Run()
    monkeypatch.setattr("longnav.utils.logging_workers.wandb.init", lambda **_kwargs: run)
    logger = WandbLoggerActor({}, compact_metrics=True)
    logger.set_context(7, "train")
    logger.log_row({"episode_label": "scene_1", "success": 1, "mean_reward": 0.5})

    assert run.logged == []
    logger.log_cycle_metrics({"rollout/success_rate": 1.0}, 7)

    payload = run.logged[-1]
    assert payload["cycle"] == 7
    assert payload["rollout/success_rate"] == 1.0
    assert "success" not in payload
    assert "details/episodes" not in payload
    assert (("rollout/*",), {"step_metric": "cycle"}) in run.definitions


def test_eval_videos_are_logged_outside_episode_tables(monkeypatch):
    run = _Run()
    monkeypatch.setattr("longnav.utils.logging_workers.wandb.init", lambda **_kwargs: run)
    monkeypatch.setattr(
        "longnav.utils.logging_workers.wandb.Video",
        lambda path, **kwargs: (path, kwargs),
    )

    logger = WandbLoggerActor({}, compact_metrics=True)
    logger.set_context(20, "eval")
    logger.log_row(
        {
            "episode_label": "scene_1",
            "success": 1,
            "vid/episode_video": "/tmp/scene_1.mp4",
            "img/thumbnail": "/tmp/scene_1.jpg",
        }
    )
    logger.log_eval_metrics({"eval/success_rate": 1.0}, 20)

    payload = run.logged[-1]
    assert "details/episodes" not in payload
    assert payload["eval/video"] == [
        (
            "/tmp/scene_1.mp4",
            {"caption": "step=20 episode=scene_1", "format": "mp4"},
        )
    ]


def test_cycle_aggregation_has_only_decision_metrics():
    rows = [
        {
            "rollout/success": 1.0,
            "rollout/oracle_success": 1.0,
            "rollout/ep_rew": 2.0,
            "rollout/ep_rtn": 1.5,
            "rollout/ep_len": 10,
            "termination_reason": "success",
            "truncated": False,
            "loss/pg_loss_scaled": torch.tensor(0.2),
            "rollout/baseline_mse": torch.tensor(0.4),
            "actor/ppo_kl": 0.01,
            "actor/pg_clipfrac": 0.1,
            "train/grad_norm": 2.0,
            "train/lr": 1e-6,
            "ref/kl_k2": 0.03,
            "chain/abs_log_ratio_mean": 0.04,
            "chain/h_drift_from_init": 0.05,
        },
        {
            "rollout/success": 0.0,
            "rollout/oracle_success": 1.0,
            "rollout/ep_rew": -1.0,
            "rollout/ep_rtn": -0.5,
            "rollout/ep_len": 20,
            "termination_reason": "terrain_out_of_bounds",
            "truncated": True,
            "loss/pg_loss_scaled": 0.4,
            "rollout/baseline_mse": 0.8,
            "actor/ppo_kl": 0.03,
            "actor/pg_clipfrac": 0.3,
            "train/grad_norm": 4.0,
            "train/lr": 1e-6,
            "ref/kl_k2": 0.05,
            "chain/abs_log_ratio_mean": 0.06,
            "chain/h_drift_from_init": 0.07,
        },
    ]

    metrics = aggregate_cycle_metrics(rows)
    assert set(key.split("/", 1)[0] for key in metrics) == {
        "rollout", "train", "policy", "probe", "exploration", "reward"
    }
    assert metrics["rollout/success_rate"] == 0.5
    assert metrics["rollout/oracle_success_rate"] == 1.0
    assert metrics["rollout/out_of_bounds_rate"] == 0.5
    assert metrics["rollout/bad_orientation_rate"] == 0.0
    assert metrics["rollout/truncated_rate"] == 0.5
    assert math.isclose(metrics["train/policy_loss"], 0.3, abs_tol=1e-7)
    assert math.isclose(metrics["policy/ref_kl"], 0.04)


def test_termination_rates_require_complete_metadata():
    stopped = {"termination_reason": "policy_stop", "truncated": False}
    timeout = {"termination_reason": "max_steps", "truncated": True}
    metrics = aggregate_cycle_metrics([stopped, timeout])
    assert metrics["rollout/policy_stop_rate"] == 0.5
    assert metrics["rollout/truncated_rate"] == 0.5
    for rows in ([{}], [stopped, {}]):
        assert math.isnan(aggregate_cycle_metrics(rows)["rollout/policy_stop_rate"])
