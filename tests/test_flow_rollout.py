"""
The closed-loop loader for the flow-matching head: `longnav.utils.flow_rollout`.

`tests/test_flow_matching_head.py` pins the head; this file pins the three things that
stand between those weights and a closed-loop number in
`habitat_physical_nav`'s `FlowRolloutBackend`:

  * **the decode rule.** `FlowActionCodec.DECODES` is `("sample", "context")` and has no
    `"mean"` on purpose -- averaging ODE solutions reproduces the conditional mean, which is
    the creeping failure the head exists to remove. The loader must refuse one rather than
    approximate it, and must refuse it BEFORE loading a 2B-parameter checkpoint, so the error
    is the same in an interpreter that cannot load one at all.
  * **the integration steps come from the checkpoint.** `num_inference_steps` round-trips
    through `turn_vector_head_config.json`; the loader leaves it alone unless a sweep asks
    for a different one. Hardcoding 10 here would silently make every eval disagree with the
    run's own logged metrics the moment a run changed it.
  * **seeding, on the right object and the right device.** This head's `.codec` IS the
    normalizer-slot object (the AR head's is the VQ table -- the trap
    `TurnFlowActionRegressor.codec`'s docstring records), and the generator has to live where
    the context does, since that is where `generate()` allocates `x_1`.

No backbone is loaded anywhere below: a real `FlowActionCodec` over a toy velocity field is
the whole object under test, and the loader's wiring is checked with the model class stubbed
out -- what it must get right is which attributes it writes, not how a VLM loads.

    pytest tests/test_flow_rollout.py -q
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for p in (str(_ROOT), str(_ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import pytest  # noqa: E402
import torch  # noqa: E402

from longnav.utils import flow_rollout  # noqa: E402
from longnav.utils.flow_matching_head import (  # noqa: E402
    FlowActionCodec,
    FlowActionDecoder,
)

N_TICKS, D_MODEL, N_CTX = 4, 8, 2
CTX = D_MODEL * N_CTX


def make_codec(num_inference_steps=3, seed=0):
    torch.manual_seed(seed)
    decoder = FlowActionDecoder(context_dim=CTX, n_ticks=N_TICKS, d_model=D_MODEL,
                                n_layers=2, n_heads=2, dim_ff=16, dropout=0.0,
                                n_context_tokens=N_CTX)
    return FlowActionCodec(decoder, num_inference_steps=num_inference_steps).eval()


class _Policy:
    """The two attributes `seed_sampling` touches, i.e. `policy.model.codec`."""

    def __init__(self, codec):
        self.model = type("M", (), {"codec": codec, "normalizer": codec})()


def _context(n=2, seed=5):
    return torch.randn(n, CTX, generator=torch.Generator().manual_seed(seed))


# ======================================================================================
# what the loader refuses, without touching a checkpoint
# ======================================================================================


def test_a_mean_decode_is_refused_and_says_why():
    """The one decode this head must never grow. `/nonexistent` never gets opened."""
    with pytest.raises(ValueError, match="conditional mean"):
        flow_rollout.load_flow_policy("/nonexistent", decode="mean")


def test_an_unknown_decode_is_refused_against_the_codecs_own_tuple():
    with pytest.raises(ValueError, match=r"decode must be one of"):
        flow_rollout.load_flow_policy("/nonexistent", decode="argmax")
    with pytest.raises(ValueError, match=r"decode must be one of"):
        flow_rollout.load_flow_policy("/nonexistent", decode="logits")


def test_the_codec_still_has_no_mean_decode():
    """Stated as a test so re-adding one is a deliberate act with a red suite attached."""
    assert FlowActionCodec.DECODES == ("sample", "context")
    assert "mean" not in FlowActionCodec.DECODES


def test_a_nonpositive_step_count_is_refused_before_the_load():
    with pytest.raises(ValueError, match="num_inference_steps"):
        flow_rollout.load_flow_policy("/nonexistent", num_inference_steps=0)


# ======================================================================================
# what the loader writes onto the model
# ======================================================================================


class _FakeModel:
    def __init__(self, codec):
        self.codec = codec
        self.normalizer = codec


def _patched_loader(monkeypatch, codec):
    """`load_flow_policy` with the checkpoint load and the policy construction stubbed.

    What is under test is which attributes the loader sets and on which object -- writing
    `model.normalizer.decode` when this head's rule lives on `model.codec` (or the reverse
    for the AR head) sets an attribute nothing reads, and the head then decodes under its
    own default. That failure is silent, which is why it is pinned rather than eyeballed.
    """
    captured = {}

    class _FakeRegressor:
        @staticmethod
        def from_pretrained(checkpoint_dir, processor, dtype=None, device=None):
            captured["checkpoint_dir"] = Path(checkpoint_dir)
            captured["dtype"] = dtype
            captured["device"] = device
            return _FakeModel(codec)

    def _fake_policy(model, processor, cfg):
        captured["model"] = model
        captured["cfg"] = cfg
        return captured

    monkeypatch.setattr(flow_rollout, "TurnFlowActionRegressor", _FakeRegressor)
    monkeypatch.setattr(flow_rollout, "VectorRolloutPolicy", _fake_policy)
    return captured


def test_the_decode_rule_is_written_on_the_normalizer_slot_object(monkeypatch):
    codec = make_codec()
    codec.decode = "context"
    _patched_loader(monkeypatch, codec)
    flow_rollout.load_flow_policy("/ckpt", decode="sample", processor=object())
    assert codec.decode == "sample"


def test_the_checkpoints_own_step_count_survives_by_default(monkeypatch):
    """`None` means "whatever the run trained under", not a constant in this file."""
    codec = make_codec(num_inference_steps=7)
    _patched_loader(monkeypatch, codec)
    flow_rollout.load_flow_policy("/ckpt", processor=object())
    assert codec.num_inference_steps == 7


def test_an_explicit_step_count_overrides_it_for_a_sweep(monkeypatch):
    codec = make_codec(num_inference_steps=7)
    _patched_loader(monkeypatch, codec)
    flow_rollout.load_flow_policy("/ckpt", num_inference_steps=2, processor=object())
    assert codec.num_inference_steps == 2


def test_the_rollout_config_reaches_the_checkpoint_load(monkeypatch):
    from longnav.utils.vector_rollout import RolloutConfig

    codec = make_codec()
    captured = _patched_loader(monkeypatch, codec)
    cfg = RolloutConfig(device="cpu", dtype=torch.float32)
    flow_rollout.load_flow_policy("/ckpt", cfg, processor=object())
    assert captured["device"] == "cpu" and captured["dtype"] is torch.float32
    assert captured["cfg"] is cfg


# ======================================================================================
# seeding: the whole reason a closed-loop number is reproducible
# ======================================================================================


def test_seed_sampling_sets_a_generator_on_the_codec():
    codec = make_codec()
    policy = _Policy(codec)
    assert codec.generator is None
    flow_rollout.seed_sampling(policy, 11)
    assert isinstance(codec.generator, torch.Generator)
    # On a CPU model the generator is a CPU one; `action_scales` is how that is known.
    assert codec.generator.device.type == codec.action_scales.device.type


def test_seed_sampling_none_clears_it():
    codec = make_codec()
    policy = _Policy(codec)
    flow_rollout.seed_sampling(policy, 11)
    flow_rollout.seed_sampling(policy, None)
    assert codec.generator is None


def test_the_same_seed_gives_the_same_chunk():
    """The claim a reproducible rollout rests on, through `denormalize` -- the exact call
    `VectorRolloutPolicy.step()` makes, not a hand-rolled `generate()`."""
    codec = make_codec()
    policy = _Policy(codec)
    context = _context()

    flow_rollout.seed_sampling(policy, 1234)
    first = codec.denormalize(context)
    flow_rollout.seed_sampling(policy, 1234)
    second = codec.denormalize(context)
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_a_different_seed_gives_a_different_chunk():
    """Otherwise the seeding is decorative and every episode draws the same noise."""
    codec = make_codec()
    policy = _Policy(codec)
    context = _context()

    flow_rollout.seed_sampling(policy, 1234)
    first = codec.denormalize(context)
    flow_rollout.seed_sampling(policy, 5678)
    other = codec.denormalize(context)
    assert not torch.allclose(first, other)


def test_successive_calls_under_one_seed_advance_the_stream():
    """Within an episode the stream must keep moving: two observations that happen to
    produce the same context must still be able to yield different chunks, or the policy
    is deterministic-by-accident under a sampling decode."""
    codec = make_codec()
    policy = _Policy(codec)
    context = _context()

    flow_rollout.seed_sampling(policy, 99)
    first = codec.denormalize(context)
    second = codec.denormalize(context)
    assert not torch.allclose(first, second)


def test_the_chunk_is_the_shape_the_executor_expects():
    codec = make_codec()
    flow_rollout.seed_sampling(_Policy(codec), 3)
    chunk = codec.denormalize(_context(n=1))
    assert tuple(chunk.shape) == (1, N_TICKS, 3)
    assert torch.isfinite(chunk).all()


def test_the_step_count_changes_the_chunk():
    """`num_inference_steps` is policy, not bookkeeping: the same weights and the same
    noise integrated differently are different actions."""
    codec = make_codec(num_inference_steps=2)
    policy = _Policy(codec)
    context = _context()

    flow_rollout.seed_sampling(policy, 7)
    coarse = codec.denormalize(context)
    codec.num_inference_steps = 10
    flow_rollout.seed_sampling(policy, 7)
    fine = codec.denormalize(context)
    assert not torch.allclose(coarse, fine)
