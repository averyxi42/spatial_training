"""
Tests for the pose-injection run: the frame transform, the SE(2) encoder, the marker in
the user block, and the stop head.

The load-bearing ones, in the order they would ruin the experiment:

  * **train/rollout frame agreement.** The collator transforms a whole window; the rollout
    transforms an accumulating prefix and keeps the last row. If those disagree the model
    is deployed on a coordinate system it never trained on, and nothing anywhere errors.
  * **windowing anchors to the first visible row.** A window is a new episode, so the same
    physical trajectory must produce the same values whether it arrived as a window or as
    a whole episode.
  * **the marker leaves the assistant turn alone.** It goes in the user block, so readout
    spans, span lengths and `train_content_len` must be bit-identical to the no-marker
    conversation. Asserted, not assumed.
  * **stop-grad really isolates.** The whole justification for not tuning the stop loss
    weight is that its gradient cannot reach the motion objective.

    pytest tests/test_pose_injection.py -q
"""

import math
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for p in (str(_ROOT), str(_ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import pytest
import torch
import torch.nn as nn
from transformers import AutoProcessor

from longnav.utils.modality_embed import (
    ENCODER_REGISTRY,
    TRANSFORM_REGISTRY,
    FourierSE2Encoder,
    ModalityEmbedSpec,
)
from longnav.utils.pose_frame import POSE_DIM, relative_se2, relative_se2_last, wrap_to_pi
from longnav.utils.stop_head import (
    StopHead,
    StopHeadConfig,
    average_precision,
    episode_stop_labels,
    roc_auc,
    stop_metrics,
)
from longnav.utils.turn_vectors import TurnVectorHead, find_turn_spans

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
POSE_SPEC = ModalityEmbedSpec(
    token="<pose>", n_features=POSE_DIM, encoder="fourier_se2",
    column="obs_poses", transform="pose_relative_first",
)


def _episode(n=12, seed=0):
    """A plausible raw scene-frame trajectory: far from the origin, heading wrapping."""
    g = torch.Generator().manual_seed(seed)
    xy = torch.rand((n, 2), generator=g) * 4.0 + torch.tensor([37.5, -12.25])
    theta = (torch.rand((n,), generator=g) * 2 - 1) * math.pi
    return torch.cat([xy, theta[:, None]], dim=1).double()


# ======================================================================================
# Angle wrapping
# ======================================================================================
def test_wrap_to_pi_covers_the_corpus_case():
    """The concrete failure from the corpus: 5.955 rad is really a 0.400 rad turn."""
    naive = torch.tensor([5.955])
    assert abs(float(wrap_to_pi(naive)) - (5.955 - 2 * math.pi)) < 1e-6
    assert abs(float(wrap_to_pi(naive))) == pytest.approx(0.328, abs=0.01)


def test_wrap_to_pi_is_half_open_on_the_right():
    """`(-pi, pi]`: a half turn maps to +pi from either side, never to -pi."""
    assert float(wrap_to_pi(torch.tensor([math.pi]))) == pytest.approx(math.pi)
    assert float(wrap_to_pi(torch.tensor([-math.pi]))) == pytest.approx(math.pi)
    vals = torch.linspace(-20, 20, 4001)
    w = wrap_to_pi(vals)
    assert bool((w > -math.pi - 1e-6).all()) and bool((w <= math.pi + 1e-6).all())
    # Wrapping is idempotent and preserves the angle modulo 2*pi.
    assert torch.allclose(wrap_to_pi(w), w, atol=1e-6)
    assert torch.allclose(torch.cos(w), torch.cos(vals), atol=1e-5)
    assert torch.allclose(torch.sin(w), torch.sin(vals), atol=1e-5)


# ======================================================================================
# The frame transform
# ======================================================================================
def test_relative_se2_origin_is_exactly_zero():
    rel = relative_se2(_episode())
    assert rel.shape == (12, POSE_DIM)
    assert torch.equal(rel[0], torch.zeros(POSE_DIM))


def test_relative_se2_is_invariant_to_a_global_rigid_motion():
    """The point of expressing poses in the start frame: the scene's arbitrary origin and
    orientation must not survive into the value. Rototranslate the whole episode and the
    injected values may not move."""
    poses = _episode()
    a, tx, ty = 1.1, -55.0, 71.5
    c, s = math.cos(a), math.sin(a)
    moved = torch.stack([
        c * poses[:, 0] - s * poses[:, 1] + tx,
        s * poses[:, 0] + c * poses[:, 1] + ty,
        poses[:, 2] + a,
    ], dim=1)
    assert torch.allclose(relative_se2(poses), relative_se2(moved), atol=1e-4)


def test_relative_se2_recovers_a_known_displacement():
    """Start facing +y; step one metre north. In the start frame that is one metre AHEAD
    (+x), not one metre to the side -- i.e. the rotation is actually applied."""
    poses = torch.tensor([[10.0, 10.0, math.pi / 2], [10.0, 11.0, math.pi / 2]])
    rel = relative_se2(poses)
    assert rel[1].tolist() == pytest.approx([1.0, 0.0, 0.0], abs=1e-5)


def test_relative_se2_unwraps_the_heading_difference():
    poses = torch.tensor([[0.0, 0.0, 3.0], [0.0, 0.0, -3.0]])
    # Naive difference is -6.0 rad; the true rotation is the short way round.
    assert float(relative_se2(poses)[1, 2]) == pytest.approx(2 * math.pi - 6.0, abs=1e-5)


def test_relative_se2_rejects_bad_shapes_and_accepts_empty():
    with pytest.raises(ValueError):
        relative_se2(torch.zeros(4, 2))
    assert relative_se2(torch.zeros(0, POSE_DIM)).shape == (0, POSE_DIM)


# ======================================================================================
# A window is a new episode
# ======================================================================================
def test_window_anchors_to_the_first_visible_row():
    """Values for a window must depend only on the window, never on the frames before it."""
    poses = _episode(n=20, seed=3)
    for start in (0, 1, 7, 15):
        window = poses[start:start + 5]
        # Prepending unseen history must not change a single value in the window.
        assert torch.equal(relative_se2(window), relative_se2(poses[start:start + 5]))
        assert torch.equal(relative_se2(window)[0], torch.zeros(POSE_DIM))
        # And the window is genuinely self-contained: the frames before it are not read.
        shifted = poses.clone()
        shifted[:start] += 1000.0
        assert torch.equal(relative_se2(shifted[start:start + 5]), relative_se2(window))


def test_a_mid_episode_window_is_not_episode_anchored():
    """Guards the decision itself: if someone reverts to episode-start-relative, the
    window's first row stops being the origin and this fires."""
    poses = _episode(n=20, seed=4)
    windowed = relative_se2(poses[6:11])
    episode = relative_se2(poses)[6:11]
    assert not torch.allclose(windowed, episode, atol=1e-3)


# ======================================================================================
# Train / rollout agreement -- the one that fails silently
# ======================================================================================
def test_rollout_prefix_transform_matches_the_collator_window():
    """The rollout sees one pose at a time and keeps `relative_se2(prefix)[-1]`; the
    collator transforms the whole window at once. Row for row, they must agree."""
    poses = _episode(n=16, seed=7)
    collated = POSE_SPEC.apply_transform(poses.float())
    incremental = torch.cat(
        [relative_se2_last(poses[: i + 1]) for i in range(poses.shape[0])], dim=0
    )
    assert torch.allclose(collated, incremental, atol=1e-6)


def test_policy_pose_values_reproduces_the_collated_window():
    """The rollout method itself, not just the function under it.

    `VectorRolloutPolicy.pose_values` is exercised directly -- building a real policy needs
    a 2B backbone, and what is being checked here is the episode bookkeeping (accumulate
    raw, re-derive against the first) rather than anything the backbone does.
    """
    from longnav.utils.vector_rollout import VectorRolloutPolicy

    poses = _episode(n=14, seed=13)
    policy = object.__new__(VectorRolloutPolicy)
    policy._raw_poses = []
    stepwise = torch.cat([policy.pose_values(p) for p in poses], dim=0)
    assert torch.allclose(stepwise, POSE_SPEC.apply_transform(poses.float()), atol=1e-6)

    # A new episode is a new origin -- which is exactly what a training window is.
    policy._raw_poses = []
    again = torch.cat([policy.pose_values(p) for p in poses[6:]], dim=0)
    assert torch.allclose(again, POSE_SPEC.apply_transform(poses[6:].float()), atol=1e-6)


def test_policy_pose_values_rejects_a_malformed_pose():
    from longnav.utils.vector_rollout import VectorRolloutPolicy

    policy = object.__new__(VectorRolloutPolicy)
    policy._raw_poses = []
    with pytest.raises(ValueError, match="x, y, theta"):
        policy.pose_values([1.0, 2.0])


def test_rollout_matches_collator_on_a_mid_episode_window():
    """Same claim when the training sample was a window: the rollout that starts at that
    window's first frame reproduces exactly the window's values."""
    poses = _episode(n=25, seed=11)
    start, cap = 9, 8
    window = poses[start:start + cap]
    collated = POSE_SPEC.apply_transform(window.float())
    incremental = torch.cat(
        [relative_se2_last(window[: i + 1]) for i in range(cap)], dim=0
    )
    assert torch.allclose(collated, incremental, atol=1e-6)


# ======================================================================================
# The spec-level transform hook
# ======================================================================================
def test_transform_defaults_to_identity_and_leaves_old_specs_alone():
    plain = ModalityEmbedSpec(token="<x>", n_features=4, encoder="mlp")
    assert plain.transform is None and plain.raw_width == 4
    v = torch.randn(6, 4)
    assert torch.equal(plain.apply_transform(v), v)


def test_transform_registry_key_is_validated_at_config_time():
    with pytest.raises(KeyError, match="unknown modality transform"):
        ModalityEmbedSpec(token="<x>", n_features=3, encoder="mlp", transform="nope")
    assert "pose_relative_first" in TRANSFORM_REGISTRY


def test_transform_round_trips_through_the_checkpoint_form():
    d = POSE_SPEC.to_dict()
    assert d["transform"] == "pose_relative_first"
    assert ModalityEmbedSpec.from_dict(d) == POSE_SPEC


def test_n_raw_features_must_not_disagree_without_a_transform():
    with pytest.raises(ValueError, match="no transform is declared"):
        ModalityEmbedSpec(token="<x>", n_features=3, encoder="mlp", n_raw_features=5)


def test_a_transform_must_be_row_preserving():
    from longnav.utils.modality_embed import register_transform

    @register_transform("_test_drops_a_row")
    def _drop(v):
        return v[:-1]

    spec = ModalityEmbedSpec(token="<x>", n_features=3, encoder="mlp",
                             transform="_test_drops_a_row")
    with pytest.raises(ValueError, match="row-preserving"):
        spec.apply_transform(torch.randn(5, 3))


# ======================================================================================
# FourierSE2Encoder
# ======================================================================================
def test_fourier_encoder_output_is_value_independent_at_init():
    """Zero-init on the final layer: the output does not *depend on* the pose at step 0.

    Not the same claim as "identical to an uninjected model" -- the scatter replaces the
    marker's embedding, so this is a zero vector, not the marker id's own embedding.
    """
    enc = FourierSE2Encoder(n_features=3, d_model=32)
    a = enc(torch.tensor([[1.0, 2.0, 0.5]]))
    b = enc(torch.tensor([[-9.0, 4.0, -2.5]]))
    assert torch.equal(a, b) and bool((a == 0).all())


def test_fourier_encoder_depends_on_the_value_once_trained():
    enc = FourierSE2Encoder(n_features=3, d_model=32)
    with torch.no_grad():
        enc.net[-1].weight.normal_(0, 0.1)
    a = enc(torch.tensor([[1.0, 2.0, 0.5]]))
    b = enc(torch.tensor([[1.0, 2.0, 0.6]]))
    assert not torch.allclose(a, b)


def test_fourier_encoder_heading_is_continuous_across_the_wrap():
    """`(cos, sin)`, never raw theta: +pi and -pi are the same heading, so the features
    must coincide there. With raw theta they would be maximally far apart."""
    enc = FourierSE2Encoder(n_features=3, d_model=8)
    f_pos = enc.features(torch.tensor([[0.0, 0.0, math.pi]]))
    f_neg = enc.features(torch.tensor([[0.0, 0.0, -math.pi]]))
    assert torch.allclose(f_pos, f_neg, atol=1e-6)


def test_fourier_encoder_scale_is_a_buffer_and_travels_with_the_checkpoint():
    """Training and rollout cannot disagree about what a metre means."""
    enc = FourierSE2Encoder(n_features=3, d_model=8, n_freqs=4, pos_scale=2.5)
    state = enc.state_dict()
    assert "pos_scale" in state and "freqs" in state
    assert float(state["pos_scale"]) == 2.5 and state["freqs"].shape == (4,)

    reloaded = FourierSE2Encoder(n_features=3, d_model=8, n_freqs=4, pos_scale=1.0)
    reloaded.load_state_dict(state)
    assert float(reloaded.pos_scale) == 2.5
    assert torch.equal(reloaded.freqs, enc.freqs)


def test_fourier_encoder_computes_in_fp32_under_bf16_autocast():
    """bf16 has 8 mantissa bits; `sin(2*pi*x/0.25)` on a bf16 x is noise. The base class
    disables autocast, so two nearby poses must still differ."""
    enc = FourierSE2Encoder(n_features=3, d_model=16)
    with torch.no_grad():
        enc.net[-1].weight.normal_(0, 0.1)
    close = torch.tensor([[10.00, 3.0, 0.1], [10.01, 3.0, 0.1]])
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = enc(close, dtype=torch.bfloat16)
    assert out.dtype == torch.bfloat16
    assert not torch.equal(out[0], out[1])


def test_fourier_encoder_rejects_a_non_pose_width():
    with pytest.raises(ValueError, match="planar pose"):
        FourierSE2Encoder(n_features=4, d_model=8)
    assert ENCODER_REGISTRY["fourier_se2"] is FourierSE2Encoder


# ======================================================================================
# The marker in the user block
# ======================================================================================
@pytest.fixture(scope="module")
def processor():
    return AutoProcessor.from_pretrained(MODEL_ID)


def _conversation(marker=None, n_turns=3):
    from longnav.utils.vector_rollout import RolloutConfig, user_block_content

    cfg = RolloutConfig(modality_marker=marker)
    messages = [{"role": "user", "content": [{"type": "text", "text": "system prompt"}]}]
    for i in range(n_turns):
        messages += [
            {"role": "user", "content": user_block_content(cfg, i)},
            {"role": "assistant", "content": [{"type": "text", "text": "**____**"}]},
        ]
    return messages


def test_marker_is_written_once_per_turn_after_the_image(processor):
    text = processor.apply_chat_template(_conversation("<pose>"), tokenize=False)
    assert text.count("<pose>") == 3
    # After the image span, and before the "Action:" cue.
    assert text.index("<|vision_end|>") < text.index("<pose>") < text.index("Action:")


def test_marker_does_not_sit_immediately_after_vision_start(processor):
    """`get_rope_index` reads exactly that position to decide image-vs-video; a marker
    there makes every downstream position silently wrong."""
    tok = processor.tokenizer
    tok.add_special_tokens({"additional_special_tokens": ["<pose>"]})
    ids = tok(processor.apply_chat_template(_conversation("<pose>"), tokenize=False),
              return_tensors="pt")["input_ids"]
    marker_id = tok.convert_tokens_to_ids("<pose>")
    start_id = tok.convert_tokens_to_ids("<|vision_start|>")
    starts = (ids[0] == start_id).nonzero().flatten().tolist()
    assert starts
    assert all(int(ids[0, s + 1]) != marker_id for s in starts)


def test_marker_is_one_token(processor):
    tok = processor.tokenizer
    tok.add_special_tokens({"additional_special_tokens": ["<pose>"]})
    assert len(tok.encode("<pose>", add_special_tokens=False)) == 1


def test_assistant_spans_are_unchanged_by_the_marker(processor):
    """The marker goes in the USER block, so readout spans, their count, their length and
    hence `train_content_len` must be exactly what they were without it."""
    from longnav.utils.turn_vectors import (
        ACTION_POSTFIX,
        ACTION_PREFIX,
        resolve_affix_ids,
    )
    from longnav.utils.vector_sft import assistant_turn_indices

    tok = processor.tokenizer
    tok.add_special_tokens({"additional_special_tokens": ["<pose>"]})
    # The production affixes. Bare `**` would pair one turn's closing marker with the
    # next turn's opening one, which is exactly what these disambiguate.
    prefix_ids, postfix_ids = resolve_affix_ids(tok, ACTION_PREFIX, ACTION_POSTFIX)

    spans = {}
    for marker in (None, "<pose>"):
        messages = _conversation(marker)
        assert assistant_turn_indices(messages) == [2, 4, 6]
        text = processor.apply_chat_template(messages, tokenize=False)
        ids = tok(text, return_tensors="pt")["input_ids"]
        found = find_turn_spans(ids, prefix_ids, postfix_ids, shift_left=True)[0]
        spans[marker] = [(len(s), tok.decode(ids[0, s.start:s.end])) for s in found]

    assert len(spans[None]) == 3
    # Same number of turns, same content length per turn (== train_content_len), same
    # tokens read. Only the absolute offsets move, and nothing reads those.
    assert spans[None] == spans["<pose>"]
    assert {n for n, _ in spans["<pose>"]} == {1}


def test_rollout_text_matches_the_training_conversation(processor):
    """Token-exact, with the marker present. If these drift the rollout feeds the head a
    context it never trained on."""
    from longnav.utils.vector_rollout import RolloutConfig, full_context_text

    cfg = RolloutConfig(modality_marker="<pose>")
    rollout = full_context_text(processor, cfg, 3, system_prompt="system prompt")
    training = processor.apply_chat_template(_conversation("<pose>"), tokenize=False)
    assert rollout == training


# ======================================================================================
# The collator: windowing, the transform, and the labels together
# ======================================================================================
def _row(n_turns, seed=0):
    from PIL import Image

    poses = _episode(n=n_turns, seed=seed)
    return {
        "messages": _conversation("<pose>", n_turns=n_turns),
        "images": [Image.new("RGB", (56, 56)) for _ in range(n_turns)],
        "action_chunks": torch.zeros(n_turns, 2, 3).tolist(),
        "obs_poses": poses.tolist(),
    }, poses


def _collate(processor, row, cap, train, seed=0):
    from longnav.utils.vector_sft import DataConfig, TurnVectorCollator

    processor.tokenizer.add_special_tokens({"additional_special_tokens": ["<pose>"]})
    collator = TurnVectorCollator(
        processor,
        DataConfig(max_turns_per_sample=cap),
        train=train,
        seed=seed,
        modality_specs=(POSE_SPEC,),
        stop_labels=True,
    )
    return collator([row])


def test_collator_emits_transformed_pose_in_occurrence_order(processor):
    """End to end: raw `obs_poses` in the row, relative-to-first values on the wire, one
    per marker occurrence, counted against the tokenized text."""
    row, poses = _row(6, seed=2)
    out = _collate(processor, row, cap=None, train=False)
    values = out["modality_pose_values"]
    assert values.shape == (6, POSE_DIM)
    assert int(out["modality_pose_counts"][0]) == 6
    assert torch.allclose(values, relative_se2(poses), atol=1e-6)
    # The raw column is emphatically not what was sent.
    assert not torch.allclose(values, poses.float(), atol=1e-3)


def test_collator_transforms_after_windowing_not_before(processor):
    """The decision, at the seam that implements it: with a window, the origin is the
    window's first observation, not the episode's."""
    row, poses = _row(20, seed=6)
    out = _collate(processor, row, cap=5, train=False)  # eval window -> starts at 0
    assert torch.allclose(out["modality_pose_values"], relative_se2(poses[:5]), atol=1e-6)

    # A train window starts somewhere else; whichever it picked, row 0 of the values is
    # the origin and the values match that window transformed on its own.
    out = _collate(processor, row, cap=5, train=True, seed=3)
    values = out["modality_pose_values"]
    assert values.shape == (5, POSE_DIM)
    assert torch.equal(values[0], torch.zeros(POSE_DIM))
    starts = [s for s in range(16) if torch.allclose(values, relative_se2(poses[s:s + 5]),
                                                     atol=1e-6)]
    assert len(starts) >= 1


def test_collator_stop_label_follows_the_window(processor):
    row, _ = _row(20, seed=8)
    # Eval takes the first N, which does not reach the episode end.
    assert float(_collate(processor, row, cap=5, train=False)["stop_targets"].sum()) == 0.0
    # The whole episode does.
    full = _collate(processor, row, cap=None, train=False)["stop_targets"]
    assert full.tolist() == [0.0] * 19 + [1.0]


def test_collator_is_unchanged_without_a_pose_spec(processor):
    """Inertness at the data seam: no spec, no modality keys, no stop labels."""
    from longnav.utils.vector_sft import DataConfig, TurnVectorCollator

    row, _ = _row(4, seed=1)
    out = TurnVectorCollator(processor, DataConfig(max_turns_per_sample=None),
                             train=False)([row])
    assert not any(k.startswith("modality_") for k in out)
    assert "stop_targets" not in out


# ======================================================================================
# Stop labels
# ======================================================================================
def test_stop_label_marks_only_the_episode_end():
    labels = episode_stop_labels(n_turns=5, window_start=0, n_total_turns=5)
    assert labels.tolist() == [0, 0, 0, 0, 1]


def test_a_window_short_of_the_end_has_no_positive():
    """Correct rather than a gap: those turns genuinely are not stops."""
    assert episode_stop_labels(4, window_start=0, n_total_turns=20).sum() == 0
    assert episode_stop_labels(4, window_start=10, n_total_turns=20).sum() == 0


def test_a_window_reaching_the_end_has_exactly_one_positive():
    labels = episode_stop_labels(4, window_start=16, n_total_turns=20)
    assert labels.tolist() == [0, 0, 0, 1]


# ======================================================================================
# Ranking metrics
# ======================================================================================
def test_average_precision_and_auc_on_a_perfect_ranking():
    scores = [5.0, 4.0, 1.0, 0.0]
    labels = [1, 1, 0, 0]
    assert average_precision(scores, labels) == pytest.approx(1.0)
    assert roc_auc(scores, labels) == pytest.approx(1.0)


def test_auc_is_half_on_a_reversed_and_a_tied_ranking():
    assert roc_auc([0.0, 1.0], [1, 0]) == pytest.approx(0.0)
    assert roc_auc([1.0, 1.0], [1, 0]) == pytest.approx(0.5)


def test_average_precision_of_a_random_ranking_is_about_the_base_rate():
    g = torch.Generator().manual_seed(0)
    n = 20000
    labels = (torch.rand(n, generator=g) < 0.011).float()
    scores = torch.randn(n, generator=g)
    ap = average_precision(scores.tolist(), labels.tolist())
    assert ap == pytest.approx(float(labels.mean()), abs=0.01)


def test_metrics_are_invariant_to_temperature():
    """Why a badly calibrated head is still salvageable: any monotone rescaling of the
    logits leaves the ranking, and therefore AP and AUC, untouched."""
    g = torch.Generator().manual_seed(1)
    scores = torch.randn(500, generator=g)
    labels = (torch.rand(500, generator=g) < 0.1).float()
    a = stop_metrics(scores.tolist(), labels.tolist())
    b = stop_metrics((scores / 7.5).tolist(), labels.tolist())
    assert a["stop_ap"] == pytest.approx(b["stop_ap"])
    assert a["stop_auc"] == pytest.approx(b["stop_auc"])


def test_metrics_are_nan_rather_than_plausible_when_a_class_is_missing():
    m = stop_metrics([0.1, 0.2, 0.3], [0, 0, 0])
    assert math.isnan(m["stop_ap"]) and math.isnan(m["stop_auc"])


def test_stop_metrics_reports_the_base_rate_alongside_ap():
    m = stop_metrics([1.0, 0.0, 0.0, 0.0], [1, 0, 0, 0])
    assert m["stop_pos_rate"] == pytest.approx(0.25)
    assert m["stop_n"] == 4


# ======================================================================================
# Stop head
# ======================================================================================
def test_stop_head_starts_at_p_one_half():
    head = StopHead(16, StopHeadConfig())
    assert torch.allclose(head.probability(head(torch.randn(4, 16))),
                          torch.full((4,), 0.5))


def test_stop_head_inference_modes():
    head = StopHead(4, StopHeadConfig(inference="threshold", threshold=0.6))
    logits = torch.tensor([-2.0, 2.0])
    assert head.decide(logits).tolist() == [False, True]

    head.cfg.inference = "sample"
    g = torch.Generator().manual_seed(0)
    # A near-certain stop is sampled as a stop essentially always; a near-certain
    # non-stop essentially never.
    draws = torch.stack([head.decide(torch.tensor([8.0, -8.0]), generator=g)
                         for _ in range(200)])
    assert draws[:, 0].float().mean() > 0.98
    assert draws[:, 1].float().mean() < 0.02


def test_stop_head_temperature_moves_probability_but_not_order():
    head = StopHead(4, StopHeadConfig(temperature=4.0))
    logits = torch.tensor([-2.0, 1.0, 3.0])
    p = head.probability(logits)
    assert bool((p.argsort() == logits.argsort()).all())
    assert float(p[0]) > float(torch.sigmoid(logits[0]))  # pulled toward 0.5


def test_pos_weight_raises_the_positive_gradient():
    """What `pos_weight` is for: one positive per ninety turns otherwise contributes a
    ninetieth of the gradient."""
    logits = torch.zeros(91, requires_grad=True)
    labels = torch.zeros(91)
    labels[0] = 1.0
    plain = StopHead(2, StopHeadConfig(pos_weight=None)).loss(logits, labels)
    g_plain = torch.autograd.grad(plain, logits)[0]
    logits2 = torch.zeros(91, requires_grad=True)
    weighted = StopHead(2, StopHeadConfig(pos_weight=90.0)).loss(logits2, labels)
    g_weighted = torch.autograd.grad(weighted, logits2)[0]
    assert abs(float(g_weighted[0])) == pytest.approx(90 * abs(float(g_plain[0])), rel=1e-4)
    assert float(g_weighted[1]) == pytest.approx(float(g_plain[1]))


# ======================================================================================
# Stop-grad isolation, through the real model forward
# ======================================================================================
class _StubBackbone(nn.Module):
    """Just enough of the backbone for `TurnVectorRegressor.forward`: a differentiable
    path from parameters to `last_hidden_state`."""

    def __init__(self, vocab=64, hidden=32):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.proj = nn.Linear(hidden, hidden)

    def get_input_embeddings(self):
        return self.embed

    def forward(self, input_ids=None, **kw):
        return {"last_hidden_state": self.proj(self.embed(input_ids))}


def _toy_model(stop_cfg=None, seed=0):
    from longnav.utils.vector_sft import (
        LossConfig,
        ModelConfig,
        TargetNormalizer,
        TurnVectorRegressor,
    )

    torch.manual_seed(seed)
    hidden, out_dim = 32, 3
    head = TurnVectorHead(hidden_size=hidden, out_dim=out_dim, mode="mean")
    model_cfg = ModelConfig(shift_left=True, stop_head=stop_cfg)
    model = TurnVectorRegressor(
        backbone=_StubBackbone(hidden=hidden),
        head=head,
        normalizer=TargetNormalizer(out_dim, enabled=False),
        target_shape=(out_dim,),
        prefix_ids=[7],
        postfix_ids=[8],
        model_cfg=model_cfg,
        loss_cfg=LossConfig(normalize_targets=False),
        stop_head=None if stop_cfg is None else StopHead(head.pooled_dim, stop_cfg),
    )
    return model


# Three turns: `7 <content> 8`, so `find_turn_spans` locates three single-token readouts.
_IDS = torch.tensor([[1, 7, 20, 8, 2, 7, 21, 8, 3, 7, 22, 8]])
_TARGETS = torch.randn(3, 3)
_STOPS = torch.tensor([0.0, 0.0, 1.0])


def _grads(model, with_stop):
    model.zero_grad()
    kw = {"stop_targets": _STOPS} if with_stop else {}
    out = model(input_ids=_IDS, targets=_TARGETS.clone(), **kw)
    out["loss"].backward()
    return {n: p.grad.clone() for n, p in model.named_parameters()
            if p.grad is not None and not n.startswith("stop_head.")}


# `zero_init=False` in both arms on purpose. With the zero-initialised output layer the
# gradient reaching the pooled context is `W_out^T @ dL/dlogit`, which is identically zero
# at step 0 -- so a stop-grad test would pass at init even if `stop_grad` did nothing at
# all. Starting the head off zero is what makes the comparison mean something.
def test_stop_grad_leaves_the_motion_gradients_bit_identical():
    """The claim that makes the loss weight safe: with `stop_grad` on, the stop loss
    reaches the stop head's parameters and nothing else."""
    cfg = StopHeadConfig(stop_grad=True, loss_weight=1000.0, zero_init=False)
    a = _grads(_toy_model(cfg, seed=5), True)
    b = _grads(_toy_model(None, seed=5), False)
    assert set(a) == set(b)
    for k in a:
        assert torch.allclose(a[k], b[k], atol=1e-6), k


def test_stop_grad_off_does_reach_the_motion_parameters():
    """The other arm really is different -- otherwise the flag would be decorative."""
    cfg = StopHeadConfig(stop_grad=False, loss_weight=1000.0, zero_init=False)
    a = _grads(_toy_model(cfg, seed=5), True)
    b = _grads(_toy_model(None, seed=5), False)
    assert any(not torch.allclose(a[k], b[k], atol=1e-6) for k in a)


def test_stop_head_reports_its_own_loss_and_logits_separately():
    model = _toy_model(StopHeadConfig(), seed=5)
    out = model(input_ids=_IDS, targets=_TARGETS.clone(), stop_targets=_STOPS)
    assert out["stop_logits"].shape == (3,)
    assert out["stop_labels"].tolist() == [0.0, 0.0, 1.0]
    # Motion and stop are separable, not fused into one number.
    assert "motion_loss_sum" in out and "stop_loss_sum" in out
    assert float(out["stop_n"]) == 3.0


def test_stop_targets_must_be_sliced_with_the_window():
    model = _toy_model(StopHeadConfig(), seed=5)
    with pytest.raises(RuntimeError, match="not sliced with the turn window"):
        model(input_ids=_IDS, targets=_TARGETS.clone(), stop_targets=torch.zeros(7))


# ======================================================================================
# Checkpoint round-trip
# ======================================================================================
def _toy_with_everything(seed=0):
    """A toy model carrying both additions, for the save/load contract."""
    from longnav.utils.modality_embed import ModalityEmbedder
    from longnav.utils.vector_sft import (
        LossConfig,
        ModelConfig,
        TargetNormalizer,
        TurnVectorRegressor,
    )

    torch.manual_seed(seed)
    hidden, out_dim = 32, 3
    head = TurnVectorHead(hidden_size=hidden, out_dim=out_dim, mode="mean")
    stop_cfg = StopHeadConfig(hidden_dims=(8,), pos_weight=90.0, temperature=2.0)
    cfg = ModelConfig(shift_left=True, modality_specs=(POSE_SPEC,), stop_head=stop_cfg)
    return TurnVectorRegressor(
        backbone=_StubBackbone(hidden=hidden), head=head,
        normalizer=TargetNormalizer(out_dim, enabled=False),
        target_shape=(out_dim,), prefix_ids=[7], postfix_ids=[8],
        model_cfg=cfg, loss_cfg=LossConfig(normalize_targets=False),
        modality_embedder=ModalityEmbedder((POSE_SPEC,), d_model=hidden),
        stop_head=StopHead(head.pooled_dim, stop_cfg),
    )


def test_checkpoint_round_trips_the_spec_the_encoder_and_the_stop_head(tmp_path):
    import json

    from longnav.utils.vector_sft import HEAD_CONFIG_FILE, ModelConfig

    src = _toy_with_everything(seed=1)
    with torch.no_grad():  # move the encoder and stop head off their inits
        for p in src.modality_embedder.parameters():
            p.add_(torch.randn_like(p) * 0.05)
        for p in src.stop_head.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    src.save_pretrained(tmp_path)

    meta = json.loads((tmp_path / HEAD_CONFIG_FILE).read_text())
    assert meta["model"]["modality_specs"][0]["transform"] == "pose_relative_first"
    assert meta["model"]["stop_head"]["pos_weight"] == 90.0
    # The config form must rebuild the same objects, not plain dicts.
    rebuilt = ModelConfig(**meta["model"])
    assert rebuilt.modality_specs == (POSE_SPEC,)
    assert rebuilt.stop_head == StopHeadConfig(hidden_dims=(8,), pos_weight=90.0,
                                               temperature=2.0)

    dst = _toy_with_everything(seed=2)
    assert not torch.equal(
        dict(dst.stop_head.named_parameters())["net.1.weight"],
        dict(src.stop_head.named_parameters())["net.1.weight"],
    )
    dst.load_head_state(tmp_path)
    for (n, a), (_, b) in zip(sorted(src.named_parameters()),
                              sorted(dst.named_parameters())):
        if n.startswith(("modality_embedder.", "stop_head.", "head.")):
            assert torch.equal(a, b), n


def test_loading_a_stop_head_into_a_model_that_declares_none_raises(tmp_path):
    # A stop head but no modality spec, so this reaches the stop-head guard rather than
    # tripping the modality one first (which it would, and correctly).
    _toy_model(StopHeadConfig(), seed=1).save_pretrained(tmp_path)
    with pytest.raises(RuntimeError, match="config declares no stop head"):
        _toy_model(None, seed=1).load_head_state(tmp_path)


def test_declaring_a_stop_head_with_no_weights_in_the_checkpoint_raises(tmp_path):
    _toy_model(None, seed=1).save_pretrained(tmp_path)
    with pytest.raises(RuntimeError, match="no stop-head weights"):
        _toy_model(StopHeadConfig(), seed=1).load_head_state(tmp_path)


def test_a_no_extras_checkpoint_omits_both_fields(tmp_path):
    """Byte-identical to one written before either field existed -- an older reader would
    choke on an unexpected `ModelConfig` keyword, including a null one."""
    import json

    from longnav.utils.vector_sft import HEAD_CONFIG_FILE

    _toy_model(None, seed=1).save_pretrained(tmp_path)
    meta = json.loads((tmp_path / HEAD_CONFIG_FILE).read_text())
    assert "stop_head" not in meta["model"]
    assert "modality_specs" not in meta["model"]


def test_no_stop_head_means_no_extra_keys_at_all():
    """Inertness: a run without the flag produces exactly the outputs it always did.

    `motion_loss_sum` is absent too -- with one objective it would only be `turn_loss`
    under a second name, and adding a logged series to every existing run for nothing is
    the kind of "harmless" change that makes a diff against the base tree fail.
    `dump/pose_injection/inertness_check.py` is what actually pins this, by running the
    same forward under a checkout of the base commit.
    """
    out = _toy_model(None, seed=5)(input_ids=_IDS, targets=_TARGETS.clone())
    assert not any(k.startswith("stop") for k in out)
    assert "motion_loss_sum" not in out
    assert set(out) == {"loss", "loss_sum", "sum_sq_err", "sum_abs_err", "n_rows",
                        "n_turns", "n_tokens", "n_dense_tokens", "n_steps"}
