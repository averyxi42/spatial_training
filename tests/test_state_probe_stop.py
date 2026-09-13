from types import SimpleNamespace

import torch

from longnav.utils.state_probe import StateProbe, StateProbeConfig
from longnav.utils.vlm_worker import VLMTrainingMixin, _is_probe_objective_term


def test_stop_head_has_bce_and_firstpass_gradients():
    cfg = StateProbeConfig(
        distance=None,
        value=None,
        stop={
            "hidden_dims": [8],
            "pos_weight": 1.0,
            "radius_m": 1.0,
            "loss_weight": 1.0,
            "firstpass_weight": 1.0,
            "target_eps": 0.01,
        },
    )
    probe = StateProbe(4, cfg)
    hidden = torch.randn(1, 4, 4, requires_grad=True)
    targets = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    losses = probe.losses(hidden, stop_targets=targets, ordered=True)
    assert {"probe/stop_loss", "probe/stop_bce_loss", "probe/stop_firstpass_loss"} <= set(losses)
    losses["probe/stop_loss"].backward()
    # The persisted stop readout starts with a zero final layer, so its first update
    # intentionally has no gradient to the shared hidden state.  The trainable readout
    # itself must still receive the BCE plus first-arrival gradient.
    assert any(parameter.grad is not None for parameter in probe.stop_head.parameters())


def test_shadow_stop_loss_has_gradients_and_is_temperature_sensitive():
    cfg = StateProbeConfig(
        distance=None,
        value=None,
        stop={"hidden_dims": [8], "pos_weight": 1.0, "radius_m": 1.0,
              "loss_weight": 1.0, "firstpass_weight": 1.0, "target_eps": 0.01},
    )
    probe = StateProbe(4, cfg)
    with torch.no_grad():
        probe.stop_head.mlp[-1].weight.fill_(0.1)
    hidden = torch.randn(1, 3, 4, requires_grad=True)
    common = {
        "stop_targets": torch.tensor([[0.0, 0.0, 1.0]]),
        "shadow_stop_actions": torch.tensor([[0.0, 1.0, 0.0]]),
        "shadow_stop_rewards": torch.tensor([[0.0, -1.0, -1.0]]),
        "shadow_stop_weight": 1.0,
    }
    cold = probe.losses(hidden, shadow_stop_temperature=0.5, **common)
    warm = probe.losses(hidden, shadow_stop_temperature=2.0, **common)
    assert torch.isfinite(cold["probe/shadow_stop_rl_loss"])
    assert not torch.allclose(cold["probe/shadow_stop_rl_loss"],
                              warm["probe/shadow_stop_rl_loss"])
    cold["probe/stop_loss"].backward()
    assert any(parameter.grad is not None for parameter in probe.stop_head.parameters())


def test_stop_gradient_diagnostics_are_metrics_not_new_objective_terms():
    cfg = StateProbeConfig(
        distance=None,
        value=None,
        stop={"hidden_dims": [8], "pos_weight": 1.0, "radius_m": 1.0,
              "loss_weight": 1.0, "firstpass_weight": 1.0, "target_eps": 0.01},
    )
    probe = StateProbe(4, cfg)
    with torch.no_grad():
        probe.stop_head.mlp[-1].weight.fill_(0.1)
    losses = probe.losses(
        torch.randn(1, 3, 4, requires_grad=True),
        stop_targets=torch.tensor([[0.0, 0.0, 1.0]]),
        shadow_stop_actions=torch.tensor([[0.0, 1.0, 0.0]]),
        shadow_stop_rewards=torch.tensor([[0.0, -1.0, -1.0]]),
        gradient_diagnostics=True,
    )
    assert "probe/grad_bce_norm" in losses
    assert "probe/grad_firstpass_norm" in losses
    assert "probe/grad_shadow_norm" in losses
    assert torch.isfinite(losses["probe/grad_bce_norm"])
    assert torch.isfinite(losses["probe/stop_loss"])


def test_stop_loss_components_are_not_added_twice_to_the_training_objective():
    assert _is_probe_objective_term("probe/stop_loss")
    assert not _is_probe_objective_term("probe/stop_bce_loss")
    assert not _is_probe_objective_term("probe/stop_firstpass_loss")
    assert not _is_probe_objective_term("probe/shadow_stop_rl_loss")
    assert not _is_probe_objective_term("probe/grad_cos_bce_firstpass")


def test_action_head_stop_without_probe_drops_probe_training_inputs():
    captured = {}

    def ddp_model(**kwargs):
        captured.update(kwargs)
        return {}, None

    worker = SimpleNamespace(
        state_probe_trainable=False,
        rl_algo_config=SimpleNamespace(
            value_head=None,
            state_probe=None,
            state_probe_stop_temperature=1.0,
            state_probe_shadow_rl_weight=1.0,
            state_probe_firstpass_weight=None,
            state_probe_gradient_diagnostic_interval=0,
        ),
        ddp_model=ddp_model,
    )
    VLMTrainingMixin._training_forward(
        worker,
        embeds_inputs={"inputs_embeds": torch.zeros(1)},
        stop_targets=torch.ones(1, 2),
        shadow_stop_actions=torch.ones(1, 2),
        shadow_stop_rewards=torch.ones(1, 2),
    )
    assert captured["stop_targets"] is None
    assert captured["shadow_stop_actions"] is None
    assert captured["shadow_stop_rewards"] is None
