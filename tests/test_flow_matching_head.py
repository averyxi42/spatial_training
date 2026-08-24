"""Tests for the flow-matching action head (`longnav.utils.flow_matching_head`).

CPU only, no data, no checkpoints, no backbone -- pure module-level assertions on the
pieces that are easy to get subtly wrong and impossible to notice from a loss curve. Same
brief and same style as `tests/test_ar_head_variants.py` / `tests/test_ar_head_v2.py`.

What is pinned here, and why each one is worth a test rather than a comment:

  * THE FLOW CONVENTION ITSELF. `t = 1` is NOISE and `t = 0` is DATA, reversed from most
    papers. A sign error here trains a perfectly plausible-looking loss curve and generates
    noise at deploy time. Checked at both ends of the interval AND end to end, by
    integrating a field that returns exactly `u_t` and requiring it to land on the actions.
  * THE BLOCKWISE MASK, and that `is_causal` is False. `is_causal=True` permits the SDPA
    fast path to substitute plain causal masking and DISCARD a custom mask silently -- the
    same footgun `ar_action_head._attn_mask` documents. Mirrored from
    `test_ar_head_variants.test_markov_mask_is_not_discarded_by_is_causal`.
  * THE ACTION BLOCK BEING BIDIRECTIONAL. That is the entire structural claim of this head
    (no prefix, no teacher forcing, no exposure bias); a causal mask here would quietly turn
    it into a worse AR head with an identical-looking loss.
  * STRATIFIED TIME DRAWS actually covering the Beta support more evenly than i.i.d. ones.
  * REPRODUCIBILITY of a seeded generation, which the rollout harness relies on.

Run: <any env with torch + pytest>  python -m pytest tests/test_flow_matching_head.py
"""

import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
for p in (str(_ROOT), str(_ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from longnav.utils.flow_matching_head import (  # noqa: E402
    ACTION_SCALES,
    FLOW_METRIC_KEYS,
    FlowActionCodec,
    FlowActionDecoder,
    FlowMatchingConfig,
    TIME_ALPHA,
    TIME_OFFSET,
    TIME_SCALE,
    beta_icdf,
    euler_integrate,
    flow_interpolate,
    sample_noise,
    sample_time,
    sinusoidal_time_embedding,
)

T, D, NCTX = 6, 16, 4
CTX = NCTX * D


def make(**kw):
    torch.manual_seed(0)
    kw.setdefault("n_context_tokens", NCTX)
    ctx_dim = kw.pop("context_dim", kw["n_context_tokens"] * D)
    return FlowActionDecoder(context_dim=ctx_dim, n_ticks=T, d_model=D, n_layers=2,
                             n_heads=2, dim_ff=32, dropout=0.0, **kw).eval()


def make_codec(**kw):
    return FlowActionCodec(make(**kw), num_inference_steps=10)


# ======================================================================================
# The convention: t = 1 is NOISE, t = 0 is DATA
# ======================================================================================
def test_interpolation_endpoints_are_data_at_zero_and_noise_at_one():
    """`x_t = t*noise + (1-t)*actions`. Reversed from most of the literature; if this ever
    flips, everything downstream still runs and generates garbage."""
    torch.manual_seed(0)
    actions = torch.randn(5, T, 3)
    noise = torch.randn(5, T, 3)

    x0, u0 = flow_interpolate(actions, noise, torch.zeros(5))
    torch.testing.assert_close(x0, actions)          # t = 0 IS the data
    x1, _ = flow_interpolate(actions, noise, torch.ones(5))
    torch.testing.assert_close(x1, noise)            # t = 1 IS the noise

    # u_t = noise - actions, i.e. it points from data TOWARDS noise, and does not depend on t
    torch.testing.assert_close(u0, noise - actions)
    _, u_mid = flow_interpolate(actions, noise, torch.full((5,), 0.37))
    torch.testing.assert_close(u0, u_mid)


def test_exact_velocity_field_integrates_back_to_the_actions():
    """The strong end-to-end check on sign and direction: a field returning exactly
    `u_t = noise - actions` must carry `x_1 = noise` to `x_0 = actions` under the Euler loop
    (`dt = -1/num_steps`, t: 1 -> 0). Exact for ANY step count, because `u_t` is constant
    along the path -- so any residual is a convention error, not discretisation error."""
    torch.manual_seed(0)
    actions = torch.randn(4, T, 3)
    noise = torch.randn(4, T, 3)
    u_t = noise - actions

    for steps in (1, 2, 10, 25):
        x0 = euler_integrate(lambda x, t: u_t, noise, steps)
        torch.testing.assert_close(x0, actions, rtol=1e-5, atol=1e-6)


def test_euler_walks_time_downwards_from_one():
    """The loop must evaluate the field at t = 1, 1-1/S, ..., 1/S -- never at t = 0 (the
    data end is where it ARRIVES, not where it samples) and never upwards."""
    seen = []

    def field(x, t):
        seen.append(float(t[0]))
        return torch.zeros_like(x)

    euler_integrate(field, torch.zeros(2, T, 3), 4)
    assert seen == pytest.approx([1.0, 0.75, 0.5, 0.25])


def test_interpolate_rejects_mismatched_time_shape():
    with pytest.raises(ValueError, match="time must be"):
        flow_interpolate(torch.zeros(3, T, 3), torch.zeros(3, T, 3), torch.zeros(2))


# ======================================================================================
# The blockwise-causal mask
# ======================================================================================
@pytest.mark.parametrize("state", [False, True])
def test_blockwise_mask_and_is_causal_is_false(state):
    """Context bidirectional among itself; action block bidirectional among itself AND
    attending to the context; context does NOT attend to the actions. The only -inf quadrant
    is context-queries x action-keys.

    `is_causal` must be False: it is a HINT that permits the fast path to apply plain causal
    masking, which would silently discard this mask.
    """
    dec = make(use_incoming_motion=state)
    mask, is_causal = dec._attn_mask(torch.device("cpu"), torch.float32)

    assert is_causal is False
    C, L = dec.n_context_tokens, dec.seq_len
    assert L == C + (1 if state else 0) + T
    assert mask.shape == (L, L)

    assert (mask[:C, :C] == 0).all()          # context sees itself, bidirectionally
    assert torch.isinf(mask[:C, C:]).all() and (mask[:C, C:] < 0).all()   # ...and no further
    assert (mask[C:, :] == 0).all()           # the action block sees everything
    # explicitly: a tick attends to a LATER tick. This is the whole point of the head.
    assert mask[C, L - 1] == 0


def test_action_block_is_bidirectional_in_practice():
    """Not just the mask: perturbing the LAST tick's input must move the FIRST tick's
    output. A causal mask here would look identical in the loss and would reintroduce
    exactly the exposure bias this head exists to eliminate."""
    dec = make()
    ctx = torch.randn(3, CTX)
    t = torch.full((3,), 0.5)
    a = torch.randn(3, T, 3)
    b = a.clone()
    b[:, -1] = torch.randn(3, 3)
    va, vb = dec(ctx, a, t), dec(ctx, b, t)
    assert not torch.allclose(va[:, 0], vb[:, 0]), \
        "tick 0 must be able to attend to tick T-1; the action block is bidirectional"


def test_time_actually_conditions_the_field():
    """Concatenation-then-MLP fusion is easy to wire up so that the time channel is dropped
    (e.g. by slicing the wrong half). Two different t on the same x_t must differ."""
    dec = make()
    ctx = torch.randn(3, CTX)
    x = torch.randn(3, T, 3)
    v_lo = dec(ctx, x, torch.full((3,), 0.05))
    v_hi = dec(ctx, x, torch.full((3,), 0.95))
    assert not torch.allclose(v_lo, v_hi)


def test_tick_embedding_breaks_permutation_symmetry():
    """Without a learned tick embedding this decoder has NO positional signal at all (no
    RoPE, no sinusoidal position code), so identical per-tick inputs would produce identical
    velocities and the chunk would have no temporal shape."""
    dec = make()
    ctx = torch.randn(2, CTX)
    x = torch.ones(2, T, 3) * 0.3           # every tick identical
    v = dec(ctx, x, torch.full((2,), 0.5))
    assert not torch.allclose(v[:, 0], v[:, 1])


# ======================================================================================
# Structural claims mirroring the v2 AR decoder
# ======================================================================================
def test_no_context_projection_and_width_is_enforced():
    dec = make()
    assert not any(n.startswith("context_proj") for n in dict(dec.named_parameters()))
    with pytest.raises(ValueError, match="no context projection"):
        make(context_dim=CTX + 1)


def test_state_token_is_off_by_default():
    """It is off so that both heads gain or lack it together; enabling it on one side would
    break the comparison this baseline exists to make."""
    assert make().use_incoming_motion is False
    assert make().n_state_tokens == 0
    assert make(use_incoming_motion=True).n_state_tokens == 1


def test_state_token_absent_is_unknown_not_at_rest():
    """The validity bit is the whole point: without it, zeros would claim 'stationary'."""
    dec = make(use_incoming_motion=True)
    absent = dec._state_token(None, 2, torch.float32, torch.device("cpu"))
    at_rest = torch.zeros(2, 4)
    at_rest[:, 3] = 1.0                       # known, and stationary
    assert not torch.allclose(absent, dec._state_token(at_rest, 2, torch.float32,
                                                       torch.device("cpu")))


def test_forward_rejects_unexpanded_context():
    """The K-sample expansion must be applied to the context AND the noisy actions; getting
    only one of them right is a shape error here rather than a silent mispairing."""
    dec = make()
    with pytest.raises(ValueError, match="K-sample expansion"):
        dec(torch.randn(3, CTX), torch.randn(6, T, 3), torch.full((6,), 0.5))


# ======================================================================================
# Time sampling
# ======================================================================================
def test_beta_icdf_is_the_inverse_of_the_beta_cdf():
    """Beta(alpha, 1) has CDF x^alpha, so the inverse is u^(1/alpha)."""
    u = torch.linspace(0.0, 1.0, 11)
    x = beta_icdf(u, alpha=TIME_ALPHA, beta=1.0)
    torch.testing.assert_close(x.pow(TIME_ALPHA), u)
    with pytest.raises(NotImplementedError, match="closed form only"):
        beta_icdf(u, alpha=1.5, beta=2.0)


def test_time_draws_stay_inside_the_scaled_beta_support():
    t = sample_time(64, 8)
    assert t.shape == (64 * 8,)
    assert float(t.min()) >= TIME_OFFSET - 1e-6
    assert float(t.max()) <= TIME_OFFSET + TIME_SCALE + 1e-6


def test_stratified_time_covers_the_beta_support_more_evenly_than_iid():
    """Two assertions, because the claim has two halves.

    COVERAGE: stratification guarantees exactly one draw in each of the K equal-probability
    strata for EVERY example, rather than in expectation. i.i.d. draws collide.

    VARIANCE: the per-example mean of the K times is therefore a lower-variance estimate of
    the time marginal's mean. That is the whole reason for doing it.
    """
    torch.manual_seed(0)
    n, k = 256, 8

    def strata(t):
        # invert the affine scaling and the Beta ICDF to recover the uniform, then bucket
        u = ((t - TIME_OFFSET) / TIME_SCALE).clamp(0, 1).pow(TIME_ALPHA)
        return (u * k).clamp(0, k - 1e-6).floor().long().view(n, k)

    strat = sample_time(n, k, stratified=True)
    iid = sample_time(n, k, stratified=False)

    n_covered_strat = torch.stack([row.unique().numel() * torch.ones(1) for row in strata(strat)])
    n_covered_iid = torch.stack([row.unique().numel() * torch.ones(1) for row in strata(iid)])
    assert float(n_covered_strat.min()) == k, "every stratum must be hit, every example"
    assert float(n_covered_iid.mean()) < k - 0.5, "i.i.d. draws should collide"

    var_strat = strat.view(n, k).mean(1).var()
    var_iid = iid.view(n, k).mean(1).var()
    assert float(var_strat) < 0.5 * float(var_iid), (float(var_strat), float(var_iid))


def test_time_layout_matches_repeat_interleave():
    """`sample_time` returns [ex0_k0..ex0_kK-1, ex1_k0, ...], the order
    `context.repeat_interleave(K)` produces. Pairing a time with the wrong context is
    invisible in the loss curve."""
    n, k = 5, 4
    t = sample_time(n, k, stratified=True).view(n, k)
    # within an example the stratified draws are strictly increasing in k, by construction
    assert bool((t[:, 1:] > t[:, :-1]).all())


def test_antithetic_noise_pairs_are_exact_negatives():
    eps = sample_noise(4, 6, T, 3, antithetic=True).view(4, 6, T, 3)
    torch.testing.assert_close(eps[:, :3], -eps[:, 3:])
    with pytest.raises(ValueError, match="even k_samples"):
        sample_noise(4, 5, T, 3, antithetic=True)


def test_sinusoidal_time_embedding_is_injective_and_bounded():
    emb = sinusoidal_time_embedding(torch.linspace(0.0, 1.0, 32), D)
    assert emb.shape == (32, D)
    assert float(emb.abs().max()) <= 1.0 + 1e-6
    # distinct times must give distinct codes
    assert float((emb[1:] - emb[:-1]).abs().sum(-1).min()) > 1e-6
    with pytest.raises(ValueError, match="divisible by 2|must be even"):
        sinusoidal_time_embedding(torch.zeros(3), 15)


# ======================================================================================
# Generation through the codec (the normalizer slot)
# ======================================================================================
@pytest.mark.parametrize("steps", [1, 2, 5, 10, 25])
def test_generation_round_trip_shapes_and_finiteness(steps):
    """Integrating the LEARNED (randomly initialised) field must produce a well-shaped,
    finite chunk at any step count -- the plumbing check that the Euler loop, the action
    scaling and `compose_chunk` all agree about shapes."""
    codec = make_codec()
    ctx = torch.randn(3, CTX)
    diffs = codec.generate(ctx, num_steps=steps)
    assert diffs.shape == (3, T, 3) and torch.isfinite(diffs).all()
    chunk = codec.denormalize(ctx)
    assert chunk.shape == (3, T, 3) and torch.isfinite(chunk).all()


def test_action_scaling_round_trips_and_matches_the_ar_head_constants():
    codec = make_codec()
    torch.testing.assert_close(codec.action_scales,
                               torch.tensor(ACTION_SCALES, dtype=torch.float64))
    x = torch.randn(4, T, 3, dtype=torch.float64)
    torch.testing.assert_close(codec.unscale(codec.scale(x)), x)


def test_normalize_is_the_scaled_differential_of_the_chunk():
    """`normalize` must be `compose_chunk`'s inverse composed with the scaling, or training
    targets and rollout outputs would live in different spaces."""
    from longnav.utils.bin_codec import compose_chunk

    codec = make_codec()
    diffs = torch.randn(3, T, 3, dtype=torch.float64) * 0.02
    chunk = compose_chunk(diffs)
    torch.testing.assert_close(codec.unscale(codec.normalize(chunk)), diffs)


def test_seeded_generation_is_reproducible():
    """The rollout harness relies on this to replay an episode."""
    codec = make_codec()
    ctx = torch.randn(3, CTX)

    codec.generator = torch.Generator().manual_seed(1234)
    a = codec.denormalize(ctx)
    codec.generator = torch.Generator().manual_seed(1234)
    b = codec.denormalize(ctx)
    torch.testing.assert_close(a, b)

    codec.generator = torch.Generator().manual_seed(4321)
    c = codec.denormalize(ctx)
    assert not torch.allclose(a, c), "a different seed must give a different sample"


def test_supplying_the_noise_makes_generation_deterministic():
    codec = make_codec()
    ctx = torch.randn(2, CTX)
    noise = torch.randn(2, T, 3)
    torch.testing.assert_close(codec.generate(ctx, noise=noise),
                               codec.generate(ctx, noise=noise))


def test_codec_has_no_mean_decode():
    """Averaging ODE samples reproduces the conditional mean, which IS the creeping failure
    this line of work is about. The rule is one draw, or best-of-k under an explicit
    criterion -- never a mean, so the mode must not exist."""
    assert "mean" not in FlowActionCodec.DECODES
    codec = make_codec()
    codec.decode = "mean"
    with pytest.raises(ValueError, match="decode must be one of"):
        codec.denormalize(torch.randn(2, CTX))


def test_context_decode_is_a_passthrough():
    codec = make_codec()
    ctx = torch.randn(2, CTX)
    codec.decode = "context"
    torch.testing.assert_close(codec.denormalize(ctx), ctx)


# ======================================================================================
# Config round trips
# ======================================================================================
def test_decoder_config_round_trip():
    dec = make(use_incoming_motion=True, min_period=1e-2, max_period=2.0)
    cfg = dec.to_config()
    assert cfg["use_incoming_motion"] is True and cfg["min_period"] == 1e-2
    twin = FlowActionDecoder(context_dim=CTX, n_ticks=T, **cfg)
    twin.load_state_dict(dec.state_dict())         # must not raise: same architecture
    assert twin.use_incoming_motion is True and twin.max_period == 2.0
    assert twin.to_config() == cfg


def test_flow_matching_config_validates_stratification_against_beta():
    """Stratification needs a closed-form inverse CDF, which only exists for beta == 1."""
    assert FlowMatchingConfig().k_samples == 8            # the design's default
    assert FlowMatchingConfig().num_inference_steps == 10
    with pytest.raises(ValueError, match="stratified_time requires"):
        FlowMatchingConfig(time_beta=2.0)
    FlowMatchingConfig(time_beta=2.0, stratified_time=False)   # allowed
    with pytest.raises(ValueError, match="action_scales"):
        FlowMatchingConfig(action_scales=(0.0, 1.0, 1.0))


def test_no_free_running_metric_family():
    """There is one generation mode, so a `free_*` key would be meaningless -- and its
    presence would invite comparing it against the AR head's teacher-forced table."""
    assert not any("free" in k for k in FLOW_METRIC_KEYS)


# ======================================================================================
# Capacity, reported rather than padded
# ======================================================================================
def test_velocity_field_is_smaller_than_the_ar_decoder_at_matched_shape():
    """The design accepts the shortfall rather than widening dim_ff to hide it: the missing
    parameters are exactly the codebook embedding and the K-way output projection, which a
    continuous head does not have."""
    from longnav.utils.ar_action_head_v2 import ActionDecoderV2

    torch.manual_seed(0)
    ar = ActionDecoderV2(context_dim=CTX, n_codes=1024, n_ticks=T, d_model=D, n_layers=2,
                         n_heads=2, dim_ff=32, dropout=0.0, n_context_tokens=NCTX)
    fm = make()
    assert sum(p.numel() for p in fm.parameters()) < sum(p.numel() for p in ar.parameters())

# ======================================================================================
# RTC prefix conditioning (docs/RTC_TRAINING.md)
# ======================================================================================
def test_scalar_and_constant_per_tick_time_are_bit_identical():
    """The (N,) path is the historical one; a constant (N, T) must reproduce it exactly,
    because the per-tick change is a reshape of the same fixed featurization, not a new
    computation. Any drift here would silently shift `h`-adjacent quantities the SDE
    density amplifies by 1/sigma^2."""
    dec = make()
    torch.manual_seed(1)
    ctx = torch.randn(3, CTX)
    x_t = torch.randn(3, T, 3)
    t_row = torch.rand(3)
    with torch.no_grad():
        v_scalar = dec(ctx, x_t, t_row)
        v_tick = dec(ctx, x_t, t_row[:, None].expand(3, T))
    assert torch.equal(v_scalar, v_tick)


def test_per_tick_interpolation_puts_clean_actions_at_zero_rows():
    """With per-tick time, rows at t = 0 are the actions themselves -- the clean committed
    prefix -- and rows at t = 1 are pure noise, per the reversed-axis convention."""
    torch.manual_seed(2)
    actions, noise = torch.randn(2, T, 3), torch.randn(2, T, 3)
    time = torch.rand(2, T)
    time[:, :2], time[:, -1] = 0.0, 1.0
    x_t, u_t = flow_interpolate(actions, noise, time)
    assert torch.equal(x_t[:, :2], actions[:, :2])
    assert torch.equal(x_t[:, -1], noise[:, -1])
    assert torch.equal(u_t, noise - actions)


def test_prefix_mask_and_time_construction():
    from longnav.utils.flow_matching_head import prefix_mask_from_len, prefix_time

    mask = prefix_mask_from_len(torch.tensor([0, 3]), T)
    assert mask.tolist()[0] == [False] * T
    assert mask.tolist()[1] == [True] * 3 + [False] * (T - 3)
    t = prefix_time(torch.full((2,), 0.7), mask)
    assert t.shape == (2, T)
    assert torch.equal(t[1, :3], torch.zeros(3)) and float(t[1, 3]) == pytest.approx(0.7)
    # d >= n_ticks is a caller bug (the schedule guarantees d <= H - gap < H), not a
    # degenerate loss to average over.
    with pytest.raises(ValueError, match="prefix_len"):
        prefix_mask_from_len(torch.tensor([T]), T)
    with pytest.raises(ValueError, match="prefix_len"):
        prefix_mask_from_len(torch.tensor([-1]), T)


def test_sample_prefix_len_support_and_laws():
    from longnav.utils.flow_matching_head import sample_prefix_len

    torch.manual_seed(3)
    d = sample_prefix_len(512, 4)
    assert d.dtype == torch.int64 and int(d.min()) >= 0 and int(d.max()) <= 4
    assert torch.equal(sample_prefix_len(64, 4, zero_frac=1.0), torch.zeros(64).long())
    d_exp = sample_prefix_len(2048, 4, dist="exp")
    counts = torch.bincount(d_exp, minlength=5).float()
    assert bool((counts[:-1] >= counts[1:]).all()), "exp weights must decay per tick"
    # A gentler base spreads mass toward high d: at 0.8 the top half of the range
    # must be drawn much more often than the default halving draws it.
    d_gentle = sample_prefix_len(4096, 10, dist="exp", base=0.8)
    d_halving = sample_prefix_len(4096, 10, dist="exp", base=0.5)
    assert (d_gentle >= 5).float().mean() > 2 * (d_halving >= 5).float().mean()
    assert int(d_gentle.max()) <= 10
    with pytest.raises(ValueError, match="dist"):
        sample_prefix_len(4, 4, dist="linear")
    with pytest.raises(ValueError, match="base"):
        sample_prefix_len(4, 4, dist="exp", base=1.0)


def test_postfix_mean_ignores_committed_rows_only():
    """The masked reduction IS the loss semantics: values at committed rows must not move
    it, values anywhere else must."""
    from longnav.utils.flow_matching_head import postfix_mean, prefix_mask_from_len

    torch.manual_seed(4)
    vals = torch.randn(2, T, 3)
    mask = prefix_mask_from_len(torch.tensor([2, 0]), T)
    out = postfix_mean(vals, mask)
    assert torch.allclose(out[0], vals[0, 2:].mean())
    assert torch.allclose(out[1], vals[1].mean())
    poked = vals.clone()
    poked[0, :2] += 100.0
    assert torch.equal(postfix_mean(poked, mask), out)
    poked[0, 2] += 1.0
    assert not torch.equal(postfix_mean(poked, mask), out)


def test_generate_prefix_rows_are_the_prefix_exactly():
    """Rows [0, d) of a conditioned generation equal the committed differentials (up to the
    float32 scale/unscale round trip); the rest are generated. And the conditioning is
    real: a different prefix under the SAME base noise changes the postfix."""
    codec = make_codec()
    torch.manual_seed(5)
    ctx = torch.randn(2, CTX)
    noise = torch.randn(2, T, 3)
    prefix_a = torch.randn(2, 3, 3, dtype=torch.float64) * 0.02
    out_a = codec.generate(ctx, noise=noise, prefix=prefix_a)
    assert torch.allclose(out_a[:, :3], prefix_a, atol=1e-7)
    prefix_b = -prefix_a
    out_b = codec.generate(ctx, noise=noise, prefix=prefix_b)
    assert not torch.allclose(out_a[:, 3:], out_b[:, 3:])


def test_empty_prefix_is_bit_identical_to_none():
    """d = 0 (and None) is the historical unconditioned path, bit for bit -- the first
    decision of every episode rides on this."""
    codec = make_codec()
    torch.manual_seed(6)
    ctx = torch.randn(2, CTX)
    noise = torch.randn(2, T, 3)
    base = codec.generate(ctx, noise=noise)
    empty = codec.generate(ctx, noise=noise, prefix=torch.zeros(2, 0, 3))
    assert torch.equal(base, empty)


def test_flow_matching_config_validates_rtc_fields():
    FlowMatchingConfig(rtc_delay_max=4, rtc_delay_dist="exp", rtc_zero_frac=0.1)
    FlowMatchingConfig(rtc_delay_max=4, rtc_delay_dist="exp", rtc_delay_base=0.8)
    with pytest.raises(ValueError, match="rtc_delay_base"):
        FlowMatchingConfig(rtc_delay_base=1.0)
    with pytest.raises(ValueError, match="rtc_delay_max"):
        FlowMatchingConfig(rtc_delay_max=-1)
    with pytest.raises(ValueError, match="rtc_delay_dist"):
        FlowMatchingConfig(rtc_delay_max=2, rtc_delay_dist="normal")
    with pytest.raises(ValueError, match="rtc_zero_frac"):
        FlowMatchingConfig(rtc_zero_frac=1.5)
