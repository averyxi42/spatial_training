from types import SimpleNamespace

import torch

from longnav.utils.state_probe import BinaryStopHead, StateProbe, StateProbeConfig
from longnav.utils.vlm_worker import VLMTrainingMixin, VLMWrapper, _is_probe_objective_term
from longnav.utils.rl_core import collate_trajectories


def test_balanced_stop_bce_is_invariant_to_repeated_negative_frames():
    head = BinaryStopHead(4, hidden_dims=[8])
    short = head.loss(torch.tensor([-1.0, 2.0]), torch.tensor([0.0, 1.0]), balanced=True)
    long = head.loss(torch.tensor([-1.0] * 100 + [2.0]),
                     torch.tensor([0.0] * 100 + [1.0]), balanced=True)
    assert torch.allclose(short, long)
    logits = torch.tensor([2.0, -1.0], requires_grad=True)
    masked = head.loss(logits, torch.tensor([float('nan'), 0.0]), balanced=True)
    masked.backward()
    assert logits.grad[0] == 0 and logits.grad[1] > 0


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


def test_threshold_margin_uses_the_deployed_stop_boundary():
    labels = torch.tensor([0.0, 1.0])
    satisfied = BinaryStopHead.threshold_margin_loss(
        torch.tensor([-1.0, 4.0]), labels, threshold=0.95,
        positive_margin=0.25, negative_margin=0.25,
    )
    violated = BinaryStopHead.threshold_margin_loss(
        torch.tensor([4.0, -1.0]), labels, threshold=0.95,
        positive_margin=0.25, negative_margin=0.25,
    )
    assert satisfied < violated


def test_first_visit_margin_ignores_later_return_visits():
    labels = torch.tensor([0.0, 1.0, 1.0, 0.0, 1.0])
    logits = torch.tensor([-1.0, 4.0, 4.0, -1.0, -4.0])
    all_visits = BinaryStopHead.threshold_margin_loss(
        logits, labels, threshold=0.95, positive_margin=0.25,
        negative_margin=0.25, mode="all",
    )
    first_visit = BinaryStopHead.threshold_margin_loss(
        logits, labels, threshold=0.95, positive_margin=0.25,
        negative_margin=0.25, mode="first_visit",
    )
    assert first_visit < all_visits


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
    assert not _is_probe_objective_term("probe/stop_threshold_margin_loss")
    assert not _is_probe_objective_term("probe/shadow_stop_rl_loss")
    assert not _is_probe_objective_term("probe/grad_cos_bce_firstpass")


def test_action_head_stop_without_probe_drops_probe_training_inputs():
    captured = {}

    def ddp_model(**kwargs):
        captured.update(kwargs)
        return {}, None

    worker = SimpleNamespace(
        model=SimpleNamespace(),
        state_probe_trainable=False,
        rl_algo_config=SimpleNamespace(
            value_head=None,
            state_probe=None,
            state_probe_stop_temperature=1.0,
            state_probe_shadow_rl_weight=1.0,
            state_probe_firstpass_weight=None,
            state_probe_gradient_diagnostic_interval=0,
            state_probe_balanced_bce=False,
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


def test_frozen_probe_stays_in_eval_during_ppo_postprocess():
    class Wrapper(torch.nn.Module):
        def __init__(self, probe):
            super().__init__()
            self.state_probe = probe

    probe = StateProbe(4, StateProbeConfig(distance=None, value=None, stop={
        "hidden_dims": [8], "pos_weight": 1.0, "dropout": 0.5,
    }))
    probe.eval()
    worker = SimpleNamespace(
        ddp_model=Wrapper(probe),
        state_probe=probe,
        state_probe_trainable=False,
        is_merged=lambda: False,
        training_parameter_ids=set(),
        gradient_checkpointing=False,
        reset=lambda: None,
        accelerator=SimpleNamespace(wait_for_everyone=lambda: None),
    )

    VLMTrainingMixin._setup_training(worker)

    assert worker.ddp_model.training
    assert not probe.training


def test_frozen_stop_keeps_execution_but_skips_stop_ppo_recompute():
    from longnav.utils.rollout_core import EpisodeRolloutMixin

    assert not EpisodeRolloutMixin._categorical_stop_ppo_enabled(
        SimpleNamespace(state_probe_trainable=False)
    )
    assert EpisodeRolloutMixin._categorical_stop_ppo_enabled(
        SimpleNamespace(state_probe_trainable=True)
    )


def test_wrapper_without_probe_emits_continuous_policy_stats():
    wrapper = object.__new__(VLMWrapper)
    torch.nn.Module.__init__(wrapper)
    wrapper.vlm = SimpleNamespace(action_head=lambda hidden: {"mu": hidden})
    wrapper.action_space_type = "continuous"
    object.__setattr__(
        wrapper,
        "_forward_hidden",
        lambda _inputs: (torch.zeros(1, 2, 4), 1, 1, None),
    )

    policy_stats, values = wrapper._forward_embeds({})

    assert values is None
    assert set(policy_stats) == {"mu"}


def test_bce_can_be_disabled_without_freezing_the_stop_policy():
    probe = StateProbe(4, StateProbeConfig(distance=None, value=None, stop={
        "hidden_dims": [8], "pos_weight": 1.0, "loss_weight": 1.0,
        "firstpass_weight": 0.0,
    }))
    hidden = torch.randn(1, 3, 4)
    losses = probe.losses(hidden, stop_targets=torch.tensor([[0., 1., 0.]]),
                          bce_weight=0.0, firstpass_weight=0.0)
    losses['probe/stop_loss'].backward()
    assert losses['probe/stop_bce_loss'].item() == 0
    assert all(p.requires_grad for p in probe.stop_head.parameters())
    assert all(p.grad is not None and torch.count_nonzero(p.grad) == 0
               for p in probe.stop_head.parameters())


def test_stop_logprobs_match_the_bounded_behavior_sampler():
    import numpy as np
    from longnav.utils.state_probe import binary_stop_log_probs

    logits = torch.tensor([-100., -15., -2., 0., 2., 15., 100.], requires_grad=True)
    actual = binary_stop_log_probs(logits)
    p = np.clip(logits.detach().sigmoid().numpy().astype(np.float64), 1e-6, 1-1e-6)
    expected = torch.tensor(np.stack((np.log1p(-p), np.log(p)), axis=-1), dtype=torch.float32)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0)
    (-actual[:, 1].sum()).backward()
    assert torch.isfinite(logits.grad).all()


def test_stop_wrapper_uses_execution_readout_instead_of_value_readout():
    wrapper = object.__new__(VLMWrapper)
    torch.nn.Module.__init__(wrapper)
    wrapper.vlm = SimpleNamespace(action_head=lambda hidden: {"mu": hidden})
    wrapper.action_space_type = 'continuous'
    wrapper.state_probe = StateProbe(4, StateProbeConfig(distance=None, value=None,
        stop={"hidden_dims": [8], "pos_weight": 1., "firstpass_weight": 0.}))
    with torch.no_grad():
        wrapper.state_probe.stop_head.mlp[-1].weight.fill_(0.2)
    hidden = torch.randn(1, 4, 4)
    object.__setattr__(wrapper, '_forward_hidden', lambda _: (
        hidden, torch.tensor([3]), torch.tensor([1]), torch.tensor([2])))
    policy, _ = wrapper._forward_embeds({})
    torch.testing.assert_close(policy['stop_logits'], wrapper.state_probe.stop_head(hidden[:, [2]]))
    assert not torch.allclose(policy['stop_logits'], wrapper.state_probe.stop_head(hidden[:, [1]]))


def test_frozen_stop_placeholder_is_collatable_and_masked_from_ppo():
    trajectory = {
        "actions_continuous": [0.0, 0.0],
        "old_log_prob": [0.0, 0.0],
        "categorical_stop_action": [0.0, 1.0],
        "categorical_stop_recompute_logprob": [0.0, 0.0],
        "old_categorical_stop_logprob": [0.0, 0.0],
        "stop_policy_mask": [False, False],
    }
    batch, _ = collate_trajectories([trajectory, trajectory])
    assert not batch["stop_policy_mask"].any()
    assert batch["old_categorical_stop_logprob"].numel() > 0
