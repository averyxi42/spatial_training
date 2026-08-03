"""
Pose injection and the stop head on the AUTOREGRESSIVE head.

`tests/test_pose_injection.py` covers the mechanism and the regression path. This file
covers the one thing that file cannot: `TurnARActionClassifier` overrides `forward`
wholesale, so every piece of that wiring exists in this head only because it was mirrored
there. A change to the base forward that is not mirrored here is invisible to every test in
that file, and the symptom is not an error -- it is a pose-injection run whose injected
values silently reach nothing.

What is pinned, in the order it would ruin the experiment:

  * **the values actually reach the backbone.** Without `ModalityBatch.pop_from` +
    `pending`, the `modality_*` keys either crash the backbone or (with a tolerant one)
    vanish; either way, "the pose did not help" is the conclusion that gets drawn.
  * **train/rollout frame agreement, through this model's forward.** The collator
    transforms a whole window, the rollout an accumulating prefix. Both must produce the
    same context out of THIS head.
  * **stop-grad isolates** -- here that means it must not reach the causal decoder either,
    which the regression path has no equivalent of.
  * **inertness**: with neither declared, the forward's outputs are exactly what they were.
    `dump/pose_injection/ar_inertness_check.py` is what pins that against the base commit;
    this is the in-tree companion.

    pytest tests/test_ar_pose_injection.py -q
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for p in (str(_ROOT), str(_ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import pytest  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from longnav.utils.ar_action_head import (  # noqa: E402
    ARActionCodec,
    CausalActionDecoder,
    ChunkVQCodec,
    TurnARActionClassifier,
)
from longnav.utils.modality_embed import ModalityEmbedder, ModalityEmbedSpec  # noqa: E402
from longnav.utils.pose_frame import POSE_DIM, relative_se2, relative_se2_last  # noqa: E402
from longnav.utils.stop_head import StopHead, StopHeadConfig  # noqa: E402
from longnav.utils.turn_vectors import TurnVectorHead  # noqa: E402
from longnav.utils.vector_sft import LossConfig, ModelConfig  # noqa: E402

HIDDEN, CTX, N_CODES, N_TICKS = 32, 16, 8, 4
POSE_TOKEN_ID = 50

POSE_SPEC = ModalityEmbedSpec(
    token="<pose>", n_features=POSE_DIM, encoder="fourier_se2",
    column="obs_poses", transform="pose_relative_first",
    encoder_kwargs={"n_freqs": 4, "hidden_dims": [16]},
)

# Three turns of `<pose> 7 <content> 8`: the marker sits in the user block, ahead of the
# assistant opening, exactly where `format_action_chunk_dataset.py --modality-marker`
# writes it.
_IDS = torch.tensor([[1, POSE_TOKEN_ID, 7, 20, 8,
                      2, POSE_TOKEN_ID, 7, 21, 8,
                      3, POSE_TOKEN_ID, 7, 22, 8]])
_BARE_IDS = torch.tensor([[1, 7, 20, 8, 2, 7, 21, 8, 3, 7, 22, 8]])
_STOPS = torch.tensor([0.0, 0.0, 1.0])


class _StubBackbone(nn.Module):
    """Just enough of the backbone for the forward: an `nn.Embedding` the modality hooks
    can attach to, and a differentiable path to `last_hidden_state`.

    Deliberately not position-wise. A `proj(embed(ids))` stub (which is all the inertness
    probe needs) would make every test here that asks "did the injected value reach the
    readout?" vacuously fail, because the marker and the readout are different positions
    and nothing would carry one to the other. The causal running mean is the cheapest thing
    with the property that matters: position t depends on positions <= t, and on nothing
    after -- so a value injected at a marker reaches the readout that follows it, in the
    same direction real attention would.
    """

    def __init__(self, vocab=64, hidden=HIDDEN):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.proj = nn.Linear(hidden, hidden)

    def get_input_embeddings(self):
        return self.embed

    def forward(self, input_ids=None, **kw):
        h = self.embed(input_ids)
        n = torch.arange(1, h.shape[1] + 1, device=h.device, dtype=h.dtype)
        return {"last_hidden_state": self.proj(h.cumsum(dim=1) / n[None, :, None])}


def _toy_ar(specs=(), stop_cfg=None, seed=0, zero_init_encoder=False):
    """A `TurnARActionClassifier` with a stub backbone: the real forward, no 2B model.

    Built through the constructor rather than `build()` on purpose -- `build()` downloads
    and instantiates Qwen3-VL-2B, and nothing under test here is about the backbone.
    """
    # Every module is seeded independently, rather than letting one RNG stream run through
    # all of them. Otherwise declaring a pose spec would shift the stream and give a
    # DIFFERENT backbone and stop head at the same `seed` -- which would make the
    # warm-start equivalence test compare two unrelated models and quietly pass or fail for
    # the wrong reason.
    torch.manual_seed(seed)
    head = TurnVectorHead(hidden_size=HIDDEN, out_dim=CTX, mode="mean")
    codec = ChunkVQCodec(n_codes=N_CODES, n_dims=3)
    with torch.no_grad():
        torch.manual_seed(seed + 100)
        codec.centroids.copy_(torch.randn(N_CODES, 3, dtype=torch.float64) * 0.02)
        codec.fitted.fill_(True)
    torch.manual_seed(seed + 200)
    decoder = CausalActionDecoder(context_dim=CTX, n_codes=N_CODES, n_ticks=N_TICKS,
                                  d_model=8, n_layers=2, n_heads=2, dim_ff=16, dropout=0.0)
    torch.manual_seed(seed + 300)
    embedder = ModalityEmbedder(specs, d_model=HIDDEN) if specs else ModalityEmbedder()
    if specs:
        # Ids we choose ourselves: no tokenizer is involved in a toy model, and
        # `register()` would need one.
        embedder.token_ids = {s.key: POSE_TOKEN_ID + i for i, s in enumerate(embedder.specs)}
        if not zero_init_encoder:
            # The encoders are zero-initialised so step 0 cannot depend on the values (that
            # is the inertness guarantee, tested in test_modality_embed.py). Every test
            # here that asks whether the value REACHES anything has to move them off that
            # init first, or it would pass with the mechanism unplugged.
            for enc in embedder.encoders.values():
                for p in enc.parameters():
                    nn.init.normal_(p, std=0.3)
    torch.manual_seed(seed + 400)
    stop = None if stop_cfg is None else StopHead(head.pooled_dim, stop_cfg)
    torch.manual_seed(seed + 500)
    backbone = _StubBackbone()
    model = TurnARActionClassifier(
        backbone=backbone, head=head,
        normalizer=ARActionCodec(codec, decoder),
        target_shape=(CTX,), prefix_ids=[7], postfix_ids=[8],
        model_cfg=ModelConfig(shift_left=True, modality_specs=specs, stop_head=stop_cfg),
        loss_cfg=LossConfig(kind="cross_entropy", normalize_targets=False),
        modality_embedder=embedder,
        stop_head=stop,
    )
    model.n_ticks = N_TICKS
    model.attach_modality_hooks()
    return model


def _targets(seed=11, n_turns=3):
    torch.manual_seed(seed)
    return torch.randn(n_turns, N_TICKS, 3) * 0.05


def _poses(n=3, seed=3):
    """A raw scene-frame trajectory: far from the origin, so an untransformed value is
    obviously different from a transformed one."""
    g = torch.Generator().manual_seed(seed)
    xy = torch.rand((n, 2), generator=g) * 4.0 + torch.tensor([37.5, -12.25])
    theta = (torch.rand((n,), generator=g) * 2 - 1) * 3.14159
    return torch.cat([xy, theta[:, None]], dim=1).double()


def _kw(values):
    return {"modality_pose_values": values.float(),
            "modality_pose_counts": torch.tensor([values.shape[0]], dtype=torch.long)}


# ======================================================================================
# The modality values reach the backbone at all
# ======================================================================================
def test_forward_accepts_and_consumes_the_modality_keys():
    """`pop_from` before the backbone call, `pending` around it. If either is missing the
    keys are forwarded wholesale to a backbone that does not accept them, or nothing is
    pending when the embedding fires -- both raise, loudly, which is the point."""
    model = _toy_ar(specs=(POSE_SPEC,), seed=1)
    out = model(input_ids=_IDS, targets=_targets(), **_kw(relative_se2(_poses())))
    assert torch.isfinite(out["loss"])
    # And the mechanism really did consume them, rather than the hook being a no-op.
    assert model.modality_embedder._pending is None


def test_the_injected_value_changes_the_context():
    """The experiment's premise. Two different pose windows through the same weights must
    give different losses -- otherwise the run measures nothing."""
    model = _toy_ar(specs=(POSE_SPEC,), seed=1)
    tgt = _targets()
    a = model(input_ids=_IDS, targets=tgt.clone(), **_kw(relative_se2(_poses(seed=3))))
    b = model(input_ids=_IDS, targets=tgt.clone(), **_kw(relative_se2(_poses(seed=9))))
    assert not torch.allclose(a["loss"], b["loss"])


def test_a_zero_init_encoder_leaves_the_forward_untouched():
    """Step 0 of a pose run must equal step 0 of the same run without pose: the encoder's
    output layer is zero-initialised, so the values contribute nothing until trained. That
    is what makes warm-starting from a no-pose checkpoint a clean starting point."""
    model = _toy_ar(specs=(POSE_SPEC,), seed=4, zero_init_encoder=True)
    tgt = _targets()
    a = model(input_ids=_IDS, targets=tgt.clone(), **_kw(relative_se2(_poses(seed=3))))
    b = model(input_ids=_IDS, targets=tgt.clone(), **_kw(relative_se2(_poses(seed=9))))
    assert torch.equal(a["loss"], b["loss"])


def test_gradient_reaches_the_pose_encoder():
    """A parameter left out of the backward graph is silently frozen, and "the injection
    did not help" is exactly what that looks like from the loss curve."""
    model = _toy_ar(specs=(POSE_SPEC,), seed=1)
    model(input_ids=_IDS, targets=_targets(), **_kw(relative_se2(_poses())))["loss"].backward()
    grads = [p.grad for p in model.modality_embedder.parameters() if p.grad is not None]
    assert grads, "no gradient reached the modality encoder at all"
    assert any(float(g.abs().sum()) > 0 for g in grads)


def test_zero_touch_reaches_the_encoder_when_the_window_has_no_markers():
    """A window with no occurrences leaves the encoder out of the graph, which under
    `ddp_find_unused_parameters=False` hangs. The keep-alive term must be in this forward
    too, and must not perturb the loss."""
    model = _toy_ar(specs=(POSE_SPEC,), seed=1)
    empty = torch.zeros(0, POSE_DIM)
    out = model(input_ids=_BARE_IDS, targets=_targets(),
                modality_pose_values=empty,
                modality_pose_counts=torch.tensor([0], dtype=torch.long))
    out["loss"].backward()
    for name, p in model.modality_embedder.named_parameters():
        assert p.grad is not None, name
        assert float(p.grad.abs().sum()) == 0.0, f"{name}: the touch term must add no grad"

    # The reference is the SAME model with the mechanism swapped out, not a second model
    # built with no specs: constructing the encoders consumes RNG, so a same-seed no-spec
    # model would differ everywhere and prove nothing.
    encoders, hooked = model.modality_embedder, model.backbone.get_input_embeddings()
    encoders.detach()
    model.modality_embedder = ModalityEmbedder()
    try:
        ref = model(input_ids=_BARE_IDS, targets=_targets())
    finally:
        model.modality_embedder = encoders
        encoders.attach(hooked)
    assert torch.equal(out["loss"], ref["loss"]), "the touch term moved the loss"


def test_an_undeclared_modality_key_is_rejected():
    model = _toy_ar(specs=(POSE_SPEC,), seed=1)
    with pytest.raises(Exception):
        model(input_ids=_IDS, targets=_targets(),
              modality_elevation_values=torch.zeros(3, 1),
              modality_elevation_counts=torch.tensor([3]))


# ======================================================================================
# Train / rollout frame agreement, through this head's forward
# ======================================================================================
def test_rollout_prefix_values_reproduce_the_collator_window_through_the_forward():
    """The collator transforms the whole window at once; the rollout accumulates raw poses
    and keeps `relative_se2(prefix)[-1]` per step. Fed to THIS model they must produce
    bit-identical outputs -- if they do not, the AR policy is deployed on a coordinate
    convention it never trained on and nothing anywhere errors."""
    from longnav.utils.vector_rollout import VectorRolloutPolicy

    poses = _poses(n=3, seed=17)
    collated = POSE_SPEC.apply_transform(poses.float())

    policy = object.__new__(VectorRolloutPolicy)
    policy._raw_poses = []
    stepwise = torch.cat([policy.pose_values(p) for p in poses], dim=0)
    assert torch.allclose(stepwise, collated, atol=1e-6)

    model = _toy_ar(specs=(POSE_SPEC,), seed=1)
    tgt = _targets()
    a = model(input_ids=_IDS, targets=tgt.clone(), **_kw(collated))
    b = model(input_ids=_IDS, targets=tgt.clone(), **_kw(stepwise))
    assert torch.equal(a["loss"], b["loss"])
    for k in a:
        assert torch.equal(torch.as_tensor(a[k]), torch.as_tensor(b[k])), k


def test_a_window_is_anchored_on_its_own_first_row_through_the_forward():
    """A training window is a fresh episode: the same physical trajectory, arriving as a
    mid-episode window, must give what a rollout started at that frame gives."""
    poses = _poses(n=8, seed=21)
    start = 4
    window = poses[start:start + 3]
    assert torch.equal(relative_se2(window)[0], torch.zeros(POSE_DIM))

    model = _toy_ar(specs=(POSE_SPEC,), seed=1)
    tgt = _targets()
    a = model(input_ids=_IDS, targets=tgt.clone(), **_kw(relative_se2(window)))
    incremental = torch.cat(
        [relative_se2_last(window[: i + 1]) for i in range(window.shape[0])], dim=0
    )
    b = model(input_ids=_IDS, targets=tgt.clone(), **_kw(incremental))
    assert torch.equal(a["loss"], b["loss"])
    # And emphatically not the same as the episode-anchored values for those rows.
    c = model(input_ids=_IDS, targets=tgt.clone(),
              **_kw(relative_se2(poses)[start:start + 3]))
    assert not torch.allclose(a["loss"], c["loss"])


# ======================================================================================
# The stop head
# ======================================================================================
def _motion_grads(model, with_stop):
    model.zero_grad()
    kw = {"stop_targets": _STOPS} if with_stop else {}
    out = model(input_ids=_BARE_IDS, targets=_targets(), **kw)
    out["loss"].backward()
    return {n: p.grad.clone() for n, p in model.named_parameters()
            if p.grad is not None and not n.startswith("stop_head.")}


def test_stop_grad_leaves_the_motion_and_decoder_gradients_bit_identical():
    """The claim that makes `--stop-loss-weight` untuned. On this head it is a stronger
    claim than on the regression one: the stop loss must not reach the CAUSAL DECODER
    either, which is a fresh module on the same fresh-parameter learning rate.

    `zero_init=False` in both arms deliberately -- with the zero-initialised output layer
    the gradient into the pooled context is identically zero at step 0, so this would pass
    even if `stop_grad` did nothing at all.
    """
    cfg = StopHeadConfig(stop_grad=True, loss_weight=1000.0, zero_init=False)
    a = _motion_grads(_toy_ar(stop_cfg=cfg, seed=5), True)
    b = _motion_grads(_toy_ar(stop_cfg=None, seed=5), False)
    assert set(a) == set(b)
    assert any(n.startswith("normalizer.decoder.") for n in a), "no decoder grads compared"
    for k in a:
        assert torch.allclose(a[k], b[k], atol=1e-6), k


def test_stop_grad_off_does_reach_the_motion_parameters():
    cfg = StopHeadConfig(stop_grad=False, loss_weight=1000.0, zero_init=False)
    a = _motion_grads(_toy_ar(stop_cfg=cfg, seed=5), True)
    b = _motion_grads(_toy_ar(stop_cfg=None, seed=5), False)
    assert any(not torch.allclose(a[k], b[k], atol=1e-6) for k in a)


def test_stop_head_reports_its_own_loss_and_logits_separately():
    model = _toy_ar(stop_cfg=StopHeadConfig(), seed=5)
    out = model(input_ids=_BARE_IDS, targets=_targets(), stop_targets=_STOPS)
    assert out["stop_logits"].shape == (3,)
    assert out["stop_labels"].tolist() == [0.0, 0.0, 1.0]
    assert "motion_loss_sum" in out and "stop_loss_sum" in out
    assert float(out["stop_n"]) == 3.0
    # The split has to be a real split: the motion term is the cross-entropy alone.
    bare = _toy_ar(stop_cfg=None, seed=5)
    ref = bare(input_ids=_BARE_IDS, targets=_targets())
    assert torch.allclose(out["motion_loss_sum"] / 3.0, ref["loss"], atol=1e-6)


def test_stop_head_without_labels_stays_in_the_backward_graph():
    """Inference, or a probe harvesting logits: no labels, but under DDP the head still
    has to be reached or `ddp_find_unused_parameters=False` errors."""
    model = _toy_ar(stop_cfg=StopHeadConfig(zero_init=False), seed=5)
    out = model(input_ids=_BARE_IDS, targets=_targets())
    out["loss"].backward()
    assert "stop_loss_sum" not in out and "stop_logits" in out
    for name, p in model.stop_head.named_parameters():
        assert p.grad is not None, name


def test_stop_targets_must_be_sliced_with_the_window():
    model = _toy_ar(stop_cfg=StopHeadConfig(), seed=5)
    with pytest.raises(RuntimeError, match="not sliced with the turn window"):
        model(input_ids=_BARE_IDS, targets=_targets(), stop_targets=torch.zeros(7))


def test_the_stop_head_reads_the_same_pooled_context_the_trunk_head_did():
    """One extraction, one pooling. A second `extract_turn_vectors` call would only mostly
    agree, and the disagreement would be invisible."""
    model = _toy_ar(stop_cfg=StopHeadConfig(zero_init=False), seed=5)
    out = model(input_ids=_BARE_IDS, targets=_targets(), stop_targets=_STOPS)
    from longnav.utils.turn_vectors import extract_turn_vectors

    states, _, mask = extract_turn_vectors(
        model.backbone(input_ids=_BARE_IDS), _BARE_IDS, None,
        prefix_ids=[7], postfix_ids=[8], shift_left=True, strict=True, return_mask=True,
    )
    expected = model.stop_head(model.head.pooled_context(states, mask))
    assert torch.allclose(out["stop_logits"], expected.detach().float(), atol=1e-6)


# ======================================================================================
# Inertness, in-tree companion to dump/pose_injection/ar_inertness_check.py
# ======================================================================================
def test_no_stop_head_and_no_specs_means_no_extra_keys_at_all():
    """A run without either flag produces exactly the outputs it always did --
    `motion_loss_sum` included, which is absent on purpose: with one objective it would be
    `turn_loss` under a second name, a new logged series on every existing run for no
    information."""
    out = _toy_ar(seed=5)(input_ids=_BARE_IDS, targets=_targets())
    assert not any(k.startswith("stop") for k in out if k != "sum_stop_pred"
                   and k != "sum_stop_gt")
    assert "motion_loss_sum" not in out
    assert set(out) == {
        "loss", "loss_sum", "n_turns", "n_tokens", "n_dense_tokens", "n_steps",
        "sum_sq_err", "sum_abs_err", "n_rows", "sum_correct", "sum_topk",
        "sum_stop_pred", "sum_stop_gt", "sum_creep_pred", "sum_creep_gt",
    }


def test_declaring_nothing_leaves_the_module_list_unchanged():
    model = _toy_ar(seed=5)
    assert model.stop_head is None
    assert not model.modality_embedder
    assert list(model.modality_embedder.parameters()) == []


# ======================================================================================
# Warm start: v3's weights, this run's new modules
# ======================================================================================
def _save(model, path):
    model.save_pretrained(path)
    return path


#: What `save_pretrained` writes. The backbone is not in it (two toy models built with
#: different seeds differ there by construction), and a positional zip over
#: `named_parameters()` would silently misalign the moment one model has a module the
#: other does not -- which is the whole situation warm start is about.
_SAVED = ("head.", "normalizer.", "modality_embedder.", "stop_head.")


def _assert_saved_params_equal(src, dst, only=_SAVED):
    a, b = dict(src.named_parameters()), dict(dst.named_parameters())
    names = [n for n in a if n.startswith(only)]
    assert names, "nothing compared"
    for n in names:
        assert n in b, n
        assert torch.equal(a[n], b[n]), n


def test_warm_start_loads_shared_modules_and_leaves_new_ones_fresh(tmp_path):
    """The `--init-from` contract: everything the source checkpoint has loads, and only
    modules it never knew about are left at their init."""
    src = _toy_ar(seed=1)              # no pose, no stop head -- a v3-shaped checkpoint
    _save(src, tmp_path)

    dst = _toy_ar(specs=(POSE_SPEC,), stop_cfg=StopHeadConfig(), seed=2)
    before_enc = [p.clone() for p in dst.modality_embedder.parameters()]
    fresh = dst.warm_start(tmp_path, adapter=False)

    assert any("modality_embedder" in f for f in fresh) and "stop_head" in fresh
    # The shared modules really came across -- including the causal decoder, which lives
    # under `normalizer` and is most of what a warm start is for.
    _assert_saved_params_equal(src, dst, only=("head.", "normalizer."))
    # ...and the new ones were not touched by the load.
    for was, now in zip(before_enc, dst.modality_embedder.parameters()):
        assert torch.equal(was, now)


def test_warm_start_at_step_zero_reproduces_the_source_checkpoint(tmp_path):
    """The property that makes this a clean starting point rather than a perturbation.

    Given the same input, a warm-started model that declares a zero-initialised pose
    encoder and a stop head must reproduce the source checkpoint's motion forward BIT FOR
    BIT before any training step. Compared against the source model loaded normally, not
    against a remembered number.

    Note what "the same input" has to mean, and what it does not. The marker token is a new
    position in the sequence; its embedding is replaced by the encoder's output, which at
    zero init is a zero vector rather than nothing at all. So step 0 is invariant to the
    pose VALUES (the test below) but not to the marker's PRESENCE -- a pose run reads a
    conversation the source run never saw. `test_pose_injection.py` pins the part of that
    which is controllable: the marker sits in the user block, so readout spans, span
    lengths and `train_content_len` are unchanged.
    """
    src = _toy_ar(seed=1)
    _save(src, tmp_path)
    ref = _toy_ar(seed=7)
    ref.load_trainable(tmp_path, adapter=False)

    dst = _toy_ar(specs=(POSE_SPEC,), stop_cfg=StopHeadConfig(), seed=7,
                  zero_init_encoder=True)
    dst.warm_start(tmp_path, adapter=False)
    # Same backbone as `ref` by construction (the toy backbone is not part of the
    # checkpoint; in the real thing it is the frozen VLM plus the saved adapter, which
    # `warm_start` does load). Anything left differing is what the test is about.
    _assert_saved_params_equal(ref, dst, only=("backbone.", "head.", "normalizer."))

    tgt = _targets()
    a = ref(input_ids=_BARE_IDS, targets=tgt.clone())
    b = dst(input_ids=_BARE_IDS, targets=tgt.clone(), stop_targets=_STOPS,
            modality_pose_values=torch.zeros(0, POSE_DIM),
            modality_pose_counts=torch.tensor([0], dtype=torch.long))
    # The stop loss is a separate additive term, so `loss` legitimately differs; the motion
    # objective -- the thing being warm-started -- must not have moved at all.
    assert torch.equal(a["loss_sum"], b["loss_sum"])
    assert torch.equal(b["motion_loss_sum"] / 3.0, a["loss"])
    for k in ("sum_correct", "sum_sq_err", "sum_abs_err", "sum_topk"):
        assert torch.equal(a[k], b[k]), k


def test_warm_start_at_step_zero_is_invariant_to_the_injected_values(tmp_path):
    """The other half of the same claim, with the markers actually present: at step 0 the
    pose values move nothing, so the warm start is a starting point rather than a
    perturbation of one."""
    _save(_toy_ar(seed=1), tmp_path)
    dst = _toy_ar(specs=(POSE_SPEC,), stop_cfg=StopHeadConfig(), seed=7,
                  zero_init_encoder=True)
    dst.warm_start(tmp_path, adapter=False)

    tgt = _targets()
    a = dst(input_ids=_IDS, targets=tgt.clone(), **_kw(relative_se2(_poses(seed=3))))
    b = dst(input_ids=_IDS, targets=tgt.clone(), **_kw(relative_se2(_poses(seed=9))))
    assert torch.equal(a["loss_sum"], b["loss_sum"])


def test_warm_start_still_refuses_an_unexpected_module(tmp_path):
    """Only *absent* new modules are forgiven. Weights for a module the config does not
    declare are a genuine mismatch and must still raise."""
    _save(_toy_ar(stop_cfg=StopHeadConfig(), seed=1), tmp_path)
    with pytest.raises(RuntimeError, match="config declares no stop head"):
        _toy_ar(seed=2).warm_start(tmp_path, adapter=False)


def test_warm_start_still_refuses_shape_drift_on_a_shared_module(tmp_path):
    """`head` loads with `strict=True`: a checkpoint whose trunk head is a different width
    is a config mismatch wearing a warm start's clothes."""
    _save(_toy_ar(seed=1), tmp_path)
    other = _toy_ar(seed=2)
    other.head = TurnVectorHead(hidden_size=HIDDEN, out_dim=CTX * 2, mode="mean")
    with pytest.raises(RuntimeError):
        other.warm_start(tmp_path, adapter=False)


def test_warm_start_from_a_checkpoint_that_has_the_modules_loads_them(tmp_path):
    """The other direction: warm-starting from a run that already had both must load both,
    strictly -- otherwise `--init-from` would quietly discard them."""
    src = _toy_ar(specs=(POSE_SPEC,), stop_cfg=StopHeadConfig(), seed=1)
    with torch.no_grad():  # move them off their inits so equality means something
        for p in src.modality_embedder.parameters():
            p.add_(torch.randn_like(p) * 0.05)
        for p in src.stop_head.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    _save(src, tmp_path)

    dst = _toy_ar(specs=(POSE_SPEC,), stop_cfg=StopHeadConfig(), seed=2)
    assert dst.warm_start(tmp_path, adapter=False) == []
    _assert_saved_params_equal(src, dst)


def test_resume_is_unchanged_and_still_strict(tmp_path):
    """The resume rule must not have been weakened on the way. A model declaring a pose
    spec cannot `load_trainable` a checkpoint that has no encoder weights."""
    _save(_toy_ar(seed=1), tmp_path)
    with pytest.raises(RuntimeError, match="no encoder weights"):
        _toy_ar(specs=(POSE_SPEC,), seed=2).load_trainable(tmp_path, adapter=False)


# ======================================================================================
# Checkpoint round trip with both additions on the AR head
# ======================================================================================
def test_ar_checkpoint_round_trips_both_additions_alongside_the_ar_block(tmp_path):
    """`TurnARActionClassifier.save_pretrained` rewrites the config JSON to add its `ar_*`
    keys. That rewrite must not drop what the base wrote into it."""
    import json

    from longnav.utils.vector_sft import HEAD_CONFIG_FILE, HEAD_WEIGHTS_FILE

    src = _toy_ar(specs=(POSE_SPEC,), stop_cfg=StopHeadConfig(pos_weight=90.0), seed=1)
    src.save_pretrained(tmp_path)

    meta = json.loads((tmp_path / HEAD_CONFIG_FILE).read_text())
    assert meta["model"]["modality_specs"][0]["transform"] == "pose_relative_first"
    assert meta["model"]["stop_head"]["pos_weight"] == 90.0
    assert meta["ar_n_ticks"] == N_TICKS and meta["ar_context_dim"] == CTX
    blob = torch.load(tmp_path / HEAD_WEIGHTS_FILE, weights_only=False)
    assert {"head", "normalizer", "modality", "stop_head"} == set(blob)

    dst = _toy_ar(specs=(POSE_SPEC,), stop_cfg=StopHeadConfig(pos_weight=90.0), seed=2)
    dst.load_trainable(tmp_path, adapter=False)
    _assert_saved_params_equal(src, dst)


def test_an_ar_checkpoint_with_neither_omits_both_fields(tmp_path):
    """Byte-compatible with every checkpoint `run_v3_ctx8_pose` has already written."""
    import json

    from longnav.utils.vector_sft import HEAD_CONFIG_FILE, HEAD_WEIGHTS_FILE

    _toy_ar(seed=1).save_pretrained(tmp_path)
    meta = json.loads((tmp_path / HEAD_CONFIG_FILE).read_text())
    assert "stop_head" not in meta["model"] and "modality_specs" not in meta["model"]
    assert set(torch.load(tmp_path / HEAD_WEIGHTS_FILE, weights_only=False)) == {
        "head", "normalizer"
    }
