"""
Pose injection on the FLOW-MATCHING head, and the encoder-only warm start.

`tests/test_pose_injection.py` covers the mechanism and the regression path;
`tests/test_ar_pose_injection.py` covers the AR head. This file covers the one thing
neither can: `TurnFlowActionRegressor` overrides `forward` wholesale, so every piece of the
injection wiring exists in this head only because it was mirrored there. A change to the
base forward that is not mirrored here is invisible to both other files, and the symptom is
not an error -- it is a pose run whose injected values silently reach nothing.

What is pinned, in the order it would ruin the experiment:

  * **the values actually reach the backbone.** Without `ModalityBatch.pop_from` +
    `pending`, the `modality_*` keys either crash the backbone or vanish; either way "the
    pose did not help" is the conclusion that gets drawn.
  * **`zero_touch` keeps the encoders in the backward graph** in a window with no markers,
    without moving the loss. `ddp_find_unused_parameters=False` hangs otherwise.
  * **train/rollout frame agreement, through THIS model's forward.**
  * **inertness**: with no specs declared, the forward's outputs and the checkpoint are
    exactly what they were. `dump/flow_pose/flow_inertness_check.py` pins that against the
    pre-port commit; this is the in-tree companion.
  * **the warm start loads the ENCODERS AND NOTHING ELSE**, and fails loudly rather than
    silently skipping.

NOTE ON DETERMINISM. This head's forward draws K times and K noises from the global RNG and
its metric path integrates the ODE with it, so every comparison of two forwards seeds
immediately before each call. A test that forgot to would fail for the wrong reason.

    pytest tests/test_flow_pose_injection.py -q
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

from longnav.utils.flow_matching_head import (  # noqa: E402
    FlowActionCodec,
    FlowActionDecoder,
    FlowMatchingConfig,
    TurnFlowActionRegressor,
)
from longnav.utils.modality_embed import ModalityEmbedder, ModalityEmbedSpec  # noqa: E402
from longnav.utils.pose_frame import POSE_DIM, relative_se2, relative_se2_last  # noqa: E402
from longnav.utils.turn_vectors import TurnVectorHead  # noqa: E402
from longnav.utils.vector_sft import (  # noqa: E402
    HEAD_CONFIG_FILE, HEAD_WEIGHTS_FILE, LossConfig, ModelConfig,
)

HIDDEN, D_MODEL, N_CTX, N_TICKS = 32, 8, 2, 4
CTX = D_MODEL * N_CTX
POSE_TOKEN_ID = 50
SEED_FWD = 13

#: The spec `run_v8_flow_planar_pose` will actually run, shrunk to toy widths. `planar_se2`
#: rather than the AR test's `fourier_se2` on purpose: it is what `pose_spec_planar.json`
#: declares, and its `gain_init` is the knob the step-0 tests below turn.
def _spec(gain_init=1.0):
    return ModalityEmbedSpec(
        token="<pose>", n_features=POSE_DIM, encoder="planar_se2",
        column="obs_poses", transform="pose_relative_first",
        encoder_kwargs={"pos_scale": 4.0, "hidden_dims": [16], "out_norm": 0.4,
                        "gain_init": gain_init, "n_heading_harmonics": 1,
                        "use_radius": False},
    )


POSE_SPEC = _spec()

# Three turns of `<pose> 7 <content> 8`: the marker sits in the user block, ahead of the
# assistant opening, exactly where `format_action_chunk_dataset.py --modality-marker`
# writes it.
_IDS = torch.tensor([[1, POSE_TOKEN_ID, 7, 20, 8,
                      2, POSE_TOKEN_ID, 7, 21, 8,
                      3, POSE_TOKEN_ID, 7, 22, 8]])
_BARE_IDS = torch.tensor([[1, 7, 20, 8, 2, 7, 21, 8, 3, 7, 22, 8]])


class _StubBackbone(nn.Module):
    """Just enough of the backbone: an `nn.Embedding` the modality hooks can attach to and a
    differentiable path to `last_hidden_state`.

    Deliberately not position-wise -- a `proj(embed(ids))` stub would make every test that
    asks "did the injected value reach the readout?" vacuously fail, because the marker and
    the readout are different positions and nothing would carry one to the other. The causal
    running mean is the cheapest thing with the property that matters: position t depends on
    positions <= t and nothing after, so a value injected at a marker reaches the readout
    that follows it, in the same direction real attention would.
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


def _toy_flow(specs=(), seed=0, randomize_encoder=True, k_samples=3):
    """A `TurnFlowActionRegressor` with a stub backbone: the real forward, no 2B model.

    Built through the constructor rather than `build()` on purpose -- `build()` downloads
    and instantiates Qwen3-VL-2B, and nothing under test here is about the backbone.

    Every module is seeded independently rather than letting one RNG stream run through all
    of them, so that declaring a spec does not shift the stream and hand back a DIFFERENT
    backbone at the same `seed` -- which would make the warm-start comparisons compare two
    unrelated models and pass or fail for the wrong reason.
    """
    torch.manual_seed(seed)
    head = TurnVectorHead(hidden_size=HIDDEN, out_dim=CTX, mode="mean")
    torch.manual_seed(seed + 200)
    decoder = FlowActionDecoder(context_dim=CTX, n_ticks=N_TICKS, d_model=D_MODEL,
                                n_layers=2, n_heads=2, dim_ff=16, dropout=0.0,
                                n_context_tokens=N_CTX)
    fm_cfg = FlowMatchingConfig(k_samples=k_samples, num_inference_steps=2,
                                metric_inference_steps=2)
    torch.manual_seed(seed + 300)
    embedder = ModalityEmbedder(specs, d_model=HIDDEN) if specs else ModalityEmbedder()
    if specs:
        # Ids we choose ourselves: no tokenizer is involved in a toy model, and `register()`
        # would need one.
        embedder.token_ids = {s.key: POSE_TOKEN_ID + i for i, s in enumerate(embedder.specs)}
        if randomize_encoder:
            # Move the encoders off their init so that "the value reached something" means
            # something. `planar_se2` at `gain_init=1.0` is already live at init, but the
            # trunk is a fresh random projection either way.
            for enc in embedder.encoders.values():
                for name, p in enc.named_parameters():
                    nn.init.normal_(p, std=0.3)
    torch.manual_seed(seed + 500)
    backbone = _StubBackbone()
    model = TurnFlowActionRegressor(
        backbone=backbone, head=head,
        normalizer=FlowActionCodec(decoder, num_inference_steps=2,
                                   action_scales=fm_cfg.action_scales),
        target_shape=(CTX,), prefix_ids=[7], postfix_ids=[8],
        model_cfg=ModelConfig(shift_left=True, modality_specs=specs),
        loss_cfg=LossConfig(kind="flow_matching", normalize_targets=False),
        modality_embedder=embedder,
    )
    model.n_ticks = N_TICKS
    model.fm_cfg = fm_cfg
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


def _run(model, **kw):
    """One forward with the sampler pinned. See the module docstring's determinism note."""
    torch.manual_seed(SEED_FWD)
    return model(**kw)


def _context(model, **kw):
    """`(pooled context, outputs)` from one forward, by hooking the readout MLP.

    THE LOSS IS THE WRONG INSTRUMENT for "did the injected value reach the readout", and
    measurably so on this head: the flow MSE is `mse(v_theta, noise - actions)`, which at
    fresh init is dominated by the noise term, so a realistic 0.4-norm injection at one
    position moves it by ~4e-7 relative -- under `torch.allclose`'s default tolerance, while
    being a completely real change. Turning the encoder's `out_norm` up until the loss
    noticed would be tuning the test until it passed. The context vector is the quantity the
    injection is supposed to change and the quantity the objective consumes, so it is what
    these tests read.
    """
    grabbed = {}
    handle = model.head.register_forward_hook(
        lambda mod, args, out: grabbed.setdefault("v", out.detach().clone())
    )
    try:
        out = _run(model, **kw)
    finally:
        handle.remove()
    return grabbed["v"], out


def _rel_diff(a, b) -> float:
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


# ======================================================================================
# The modality values reach the backbone at all
# ======================================================================================
def test_forward_accepts_and_consumes_the_modality_keys():
    """`pop_from` before the backbone call, `pending` around it. If either is missing the
    keys are forwarded wholesale to a backbone that does not accept them, or nothing is
    pending when the embedding fires -- both raise, loudly, which is the point."""
    model = _toy_flow(specs=(POSE_SPEC,), seed=1)
    out = _run(model, input_ids=_IDS, targets=_targets(), **_kw(relative_se2(_poses())))
    assert torch.isfinite(out["loss"])
    # And the mechanism really did consume them, rather than the hook being a no-op.
    assert model.modality_embedder._pending is None


def test_the_injected_value_changes_the_context():
    """The experiment's premise. Two different pose windows through the same weights must
    give a different context -- otherwise the run measures nothing."""
    model = _toy_flow(specs=(POSE_SPEC,), seed=1)
    tgt = _targets()
    a, _ = _context(model, input_ids=_IDS, targets=tgt.clone(),
                    **_kw(relative_se2(_poses(seed=3))))
    b, _ = _context(model, input_ids=_IDS, targets=tgt.clone(),
                    **_kw(relative_se2(_poses(seed=9))))
    assert _rel_diff(a, b) > 1e-3


def test_a_zero_gain_encoder_leaves_the_forward_untouched():
    """`gain_init=0.0` is the step-0 guarantee: the injected vector is exactly zero, so the
    forward cannot depend on the pose values.

    Note this is a property of the SPEC, not of the mechanism: `pose_spec_planar.json`
    passes `gain_init: 1.0` and therefore gives the guarantee up deliberately (see
    `PlanarSE2Encoder`'s docstring on why a zero vector is itself an input the backbone has
    never seen). The test exists so the mechanism is known to still honour it when asked.
    """
    model = _toy_flow(specs=(_spec(gain_init=0.0),), seed=4, randomize_encoder=False)
    tgt = _targets()
    a, oa = _context(model, input_ids=_IDS, targets=tgt.clone(),
                     **_kw(relative_se2(_poses(seed=3))))
    b, ob = _context(model, input_ids=_IDS, targets=tgt.clone(),
                     **_kw(relative_se2(_poses(seed=9))))
    assert torch.equal(a, b)
    assert torch.equal(oa["loss"], ob["loss"])


def test_the_planar_spec_this_run_uses_is_live_at_step_zero():
    """The counterpart, pinned because it changes how v8's step 0 may be read: with
    `gain_init=1.0` a FRESH encoder already injects a 0.4-norm pose-dependent vector, so
    step 0 is not invariant to the pose whether or not `--init-modality-from` is passed."""
    model = _toy_flow(specs=(POSE_SPEC,), seed=4, randomize_encoder=False)
    tgt = _targets()
    a, _ = _context(model, input_ids=_IDS, targets=tgt.clone(),
                    **_kw(relative_se2(_poses(seed=3))))
    b, _ = _context(model, input_ids=_IDS, targets=tgt.clone(),
                    **_kw(relative_se2(_poses(seed=9))))
    assert _rel_diff(a, b) > 1e-3


def test_gradient_reaches_the_pose_encoder():
    """A parameter left out of the backward graph is silently frozen, and "the injection did
    not help" is exactly what that looks like from the loss curve."""
    model = _toy_flow(specs=(POSE_SPEC,), seed=1)
    _run(model, input_ids=_IDS, targets=_targets(),
         **_kw(relative_se2(_poses())))["loss"].backward()
    grads = [p.grad for p in model.modality_embedder.parameters() if p.grad is not None]
    assert grads, "no gradient reached the modality encoder at all"
    assert any(float(g.abs().sum()) > 0 for g in grads)


def test_zero_touch_reaches_the_encoder_when_the_window_has_no_markers():
    """A window with no occurrences leaves the encoder out of the graph, which under
    `ddp_find_unused_parameters=False` hangs. The keep-alive term must be in this forward
    too, and must not perturb the loss."""
    model = _toy_flow(specs=(POSE_SPEC,), seed=1)
    out = _run(model, input_ids=_BARE_IDS, targets=_targets(),
               modality_pose_values=torch.zeros(0, POSE_DIM),
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
        ref = _run(model, input_ids=_BARE_IDS, targets=_targets())
    finally:
        model.modality_embedder = encoders
        encoders.attach(hooked)
    assert torch.equal(out["loss"], ref["loss"]), "the touch term moved the loss"


def test_an_undeclared_modality_key_is_rejected():
    """`pop_from` owns the whole `modality_*` prefix. A key it does not recognise must raise
    rather than being forwarded to a backbone that will not accept it."""
    model = _toy_flow(specs=(POSE_SPEC,), seed=1)
    with pytest.raises(Exception):
        _run(model, input_ids=_IDS, targets=_targets(),
             modality_elevation_values=torch.zeros(3, 1),
             modality_elevation_counts=torch.tensor([3]))


# ======================================================================================
# Train / rollout frame agreement, through this head's forward
# ======================================================================================
def test_rollout_prefix_values_reproduce_the_collator_window_through_the_forward():
    """The collator transforms the whole window at once; the rollout accumulates raw poses
    and keeps `relative_se2(prefix)[-1]` per step. Fed to THIS model they must produce
    bit-identical outputs -- if they do not, the policy is deployed on a coordinate
    convention it never trained on and nothing anywhere errors."""
    from longnav.utils.vector_rollout import VectorRolloutPolicy

    poses = _poses(n=3, seed=17)
    collated = POSE_SPEC.apply_transform(poses.float())

    policy = object.__new__(VectorRolloutPolicy)
    policy._raw_poses = []
    stepwise = torch.cat([policy.pose_values(p) for p in poses], dim=0)
    assert torch.allclose(stepwise, collated, atol=1e-6)

    model = _toy_flow(specs=(POSE_SPEC,), seed=1)
    tgt = _targets()
    ca, a = _context(model, input_ids=_IDS, targets=tgt.clone(), **_kw(collated))
    cb, b = _context(model, input_ids=_IDS, targets=tgt.clone(), **_kw(stepwise))
    assert torch.equal(ca, cb)
    for k in a:
        assert torch.equal(torch.as_tensor(a[k]), torch.as_tensor(b[k])), k


def test_a_window_is_anchored_on_its_own_first_row_through_the_forward():
    """A training window is a fresh episode: the same physical trajectory, arriving as a
    mid-episode window, must give what a rollout started at that frame gives."""
    poses = _poses(n=8, seed=21)
    start = 4
    window = poses[start:start + 3]
    assert torch.equal(relative_se2(window)[0], torch.zeros(POSE_DIM))

    model = _toy_flow(specs=(POSE_SPEC,), seed=1)
    tgt = _targets()
    ca, a = _context(model, input_ids=_IDS, targets=tgt.clone(),
                     **_kw(relative_se2(window)))
    incremental = torch.cat(
        [relative_se2_last(window[: i + 1]) for i in range(window.shape[0])], dim=0
    )
    cb, b = _context(model, input_ids=_IDS, targets=tgt.clone(), **_kw(incremental))
    assert torch.equal(ca, cb)
    assert torch.equal(a["loss"], b["loss"])
    # And emphatically not the same as the episode-anchored values for those rows.
    cc, _ = _context(model, input_ids=_IDS, targets=tgt.clone(),
                     **_kw(relative_se2(poses)[start:start + 3]))
    assert _rel_diff(ca, cc) > 1e-3


# ======================================================================================
# Inertness, in-tree companion to dump/flow_pose/flow_inertness_check.py
# ======================================================================================
def test_no_specs_means_no_extra_keys_at_all():
    """A run without the flag produces exactly the outputs it always did. `free_*` is absent
    for this head's own reason (one generation mode); nothing modality-shaped is added."""
    out = _run(_toy_flow(seed=5), input_ids=_BARE_IDS, targets=_targets())
    assert not any("modality" in k for k in out)
    assert set(out) == {
        "loss", "loss_sum", "n_turns", "n_tokens", "n_dense_tokens", "n_steps",
        "sum_sq_err", "sum_abs_err", "n_rows", "sum_pose_sq_err", "sum_pose_abs_err",
        "sum_stop_pred", "sum_stop_gt", "sum_creep_pred", "sum_creep_gt",
        "sum_flips", "n_flip_pairs",
    }


def test_declaring_nothing_leaves_the_module_list_unchanged():
    model = _toy_flow(seed=5)
    assert not model.modality_embedder
    assert list(model.modality_embedder.parameters()) == []


def test_a_flow_checkpoint_with_no_specs_omits_the_field(tmp_path):
    """Byte-compatible with every checkpoint written before this port."""
    import json

    _toy_flow(seed=1).save_pretrained(tmp_path)
    meta = json.loads((tmp_path / HEAD_CONFIG_FILE).read_text())
    assert "modality_specs" not in meta["model"]
    assert set(torch.load(tmp_path / HEAD_WEIGHTS_FILE, weights_only=False)) == {
        "head", "normalizer"
    }


def test_flow_checkpoint_round_trips_the_specs_alongside_the_fm_block(tmp_path):
    """`TurnFlowActionRegressor.save_pretrained` rewrites the config JSON to add its `fm_*`
    keys. That rewrite must not drop what the base wrote into it."""
    import json

    src = _toy_flow(specs=(POSE_SPEC,), seed=1)
    src.save_pretrained(tmp_path)

    meta = json.loads((tmp_path / HEAD_CONFIG_FILE).read_text())
    assert meta["model"]["modality_specs"][0]["transform"] == "pose_relative_first"
    assert meta["model"]["modality_specs"][0]["encoder"] == "planar_se2"
    assert meta["fm_n_ticks"] == N_TICKS and meta["fm_context_dim"] == CTX
    blob = torch.load(tmp_path / HEAD_WEIGHTS_FILE, weights_only=False)
    assert {"head", "normalizer", "modality"} == set(blob)

    dst = _toy_flow(specs=(POSE_SPEC,), seed=2)
    dst.load_trainable(tmp_path, adapter=False)
    a, b = dict(src.named_parameters()), dict(dst.named_parameters())
    names = [n for n in a if n.startswith(("head.", "normalizer.", "modality_embedder."))]
    assert names
    for n in names:
        assert torch.equal(a[n], b[n]), n


# ======================================================================================
# The encoder-only warm start (`--init-modality-from`)
# ======================================================================================
def _foreign_checkpoint(path, encoders_from):
    """A checkpoint shaped like the REGRESSION head's, carrying `encoders_from`'s weights.

    `head` and `normalizer` deliberately hold tensors that could never load into a flow
    model -- v7's readout emits 30 numbers and its normalizer is two 3-vectors, while this
    head's readout emits a context vector and its normalizer holds the velocity field. If
    `init_modality_from` ever reached for either, it would raise here instead of quietly
    doing the wrong thing.
    """
    path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "head": {"proj.weight": torch.zeros(30, 7), "proj.bias": torch.zeros(30)},
            "normalizer": {"mean": torch.zeros(3), "std": torch.ones(3)},
            "modality": {"encoders": encoders_from.modality_embedder.encoders.state_dict()},
        },
        path / HEAD_WEIGHTS_FILE,
    )
    return path


def test_warm_start_loads_the_encoders_from_a_foreign_head_and_nothing_else(tmp_path):
    """The `--init-modality-from` contract, in the shape it will actually be used: the
    source is a REGRESSION checkpoint whose head and normalizer this model could not load
    even if it tried. Exactly the encoders come across; every other module stays fresh."""
    src = _toy_flow(specs=(POSE_SPEC,), seed=1)
    with torch.no_grad():  # move the encoders off init so equality means something
        for p in src.modality_embedder.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    _foreign_checkpoint(tmp_path / "ckpt", src)

    dst = _toy_flow(specs=(POSE_SPEC,), seed=2)
    before = {n: p.clone() for n, p in dst.named_parameters()}
    assert dst.init_modality_from(tmp_path / "ckpt") == ["pose"]

    want = dict(src.modality_embedder.named_parameters())
    got = dict(dst.modality_embedder.named_parameters())
    assert set(want) == set(got) and want
    for n in want:
        assert torch.equal(want[n], got[n]), n
    # Buffers travel with the encoders too -- `pos_scale` and `out_norm` define what a metre
    # means and where the injected vector sits, and a run that silently kept its own would
    # be injecting at a different scale than the encoder was trained at.
    for n, b in src.modality_embedder.named_buffers():
        assert torch.equal(b, dict(dst.modality_embedder.named_buffers())[n]), n
    # ...and NOTHING else moved: not the readout MLP, not the velocity field.
    for n, p in dst.named_parameters():
        if n.startswith("modality_embedder."):
            continue
        assert torch.equal(before[n], p), f"{n} was modified by the encoder warm start"
    assert any(n.startswith("normalizer.decoder.") for n in before), "no field params checked"


def test_warm_start_leaves_the_head_and_field_at_fresh_init(tmp_path):
    """Stated as an equality against a second model built the same way, which is what "fresh
    init" has to mean: the warm start must be indistinguishable from not having run, outside
    the encoders."""
    _foreign_checkpoint(tmp_path / "ckpt", _toy_flow(specs=(POSE_SPEC,), seed=1))
    dst = _toy_flow(specs=(POSE_SPEC,), seed=2)
    dst.init_modality_from(tmp_path / "ckpt")
    ref = _toy_flow(specs=(POSE_SPEC,), seed=2)

    a, b = dict(dst.named_parameters()), dict(ref.named_parameters())
    shared = [n for n in a if not n.startswith("modality_embedder.")]
    assert shared
    for n in shared:
        assert torch.equal(a[n], b[n]), n
    assert any(not torch.equal(a[n], b[n]) for n in a if n.startswith("modality_embedder."))


def test_warm_start_refuses_a_checkpoint_with_no_encoders(tmp_path):
    """A no-pose checkpoint has nothing to warm start FROM. Silently proceeding would give a
    run that reports a warm start and trains a randomly-initialised encoder."""
    _toy_flow(seed=1).save_pretrained(tmp_path)
    with pytest.raises(RuntimeError, match="no modality encoder weights"):
        _toy_flow(specs=(POSE_SPEC,), seed=2).init_modality_from(tmp_path)


def test_warm_start_refuses_when_this_run_declares_no_specs(tmp_path):
    """The flag without `--modality-specs` is a mis-specified run, not a no-op."""
    _foreign_checkpoint(tmp_path / "ckpt", _toy_flow(specs=(POSE_SPEC,), seed=1))
    with pytest.raises(RuntimeError, match="declares no modality specs"):
        _toy_flow(seed=2).init_modality_from(tmp_path / "ckpt")


def test_warm_start_refuses_shape_drift(tmp_path):
    """Both runs use the same `pose_spec_planar.json`, so the shapes agree -- but that is
    verified, not assumed. A differently-shaped encoder raises rather than being skipped."""
    _foreign_checkpoint(tmp_path / "ckpt", _toy_flow(specs=(POSE_SPEC,), seed=1))
    wide = ModalityEmbedSpec(
        token="<pose>", n_features=POSE_DIM, encoder="planar_se2", column="obs_poses",
        transform="pose_relative_first",
        encoder_kwargs={"pos_scale": 4.0, "hidden_dims": [64], "out_norm": 0.4,
                        "gain_init": 1.0},
    )
    with pytest.raises(RuntimeError):
        _toy_flow(specs=(wide,), seed=2).init_modality_from(tmp_path / "ckpt")


def test_warm_start_refuses_an_undeclared_encoder(tmp_path):
    """Weights for a spec this run does not declare are a mismatched pair of runs."""
    two = (POSE_SPEC, ModalityEmbedSpec(token="<alt>", n_features=POSE_DIM,
                                        encoder="planar_se2", column="obs_poses",
                                        encoder_kwargs={"hidden_dims": [16]}))
    _foreign_checkpoint(tmp_path / "ckpt", _toy_flow(specs=two, seed=1))
    with pytest.raises(RuntimeError, match="does not declare"):
        _toy_flow(specs=(POSE_SPEC,), seed=2).init_modality_from(tmp_path / "ckpt")


def test_warm_start_refuses_a_missing_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError):
        _toy_flow(specs=(POSE_SPEC,), seed=2).init_modality_from(tmp_path / "nope")


def test_a_warm_started_encoder_changes_step_zero(tmp_path):
    """What the warm start costs, stated as a test rather than as a comment: after it, step
    0 is a different forward from the fresh run's. With `gain_init=1.0` the fresh run was
    already not invariant to the pose values (see
    `test_the_planar_spec_this_run_uses_is_live_at_step_zero`), so what is given up here is
    agreement with the FRESH encoder, not the zero-init guarantee -- which that spec had
    already traded away."""
    src = _toy_flow(specs=(POSE_SPEC,), seed=1)
    with torch.no_grad():
        for p in src.modality_embedder.parameters():
            p.add_(torch.randn_like(p) * 0.5)
    _foreign_checkpoint(tmp_path / "ckpt", src)

    poses = _kw(relative_se2(_poses(seed=3)))
    fresh = _toy_flow(specs=(POSE_SPEC,), seed=2)
    before, _ = _context(fresh, input_ids=_IDS, targets=_targets(), **poses)
    fresh.init_modality_from(tmp_path / "ckpt")
    after, _ = _context(fresh, input_ids=_IDS, targets=_targets(), **poses)
    assert _rel_diff(before, after) > 1e-3
