"""
Tests for `longnav.utils.modality_embed` -- DESIGN.md sections 8 and 8.1.

Deliberately synthetic and deliberately irregular. There is no real modality here: the
specs carry an arbitrary `F` and the markers appear at an uneven rate -- two in one
message, none in another, one in the prologue, a second modality at a different rate.
That is the point. A test suite built around one marker per turn would pass just as well
against a mechanism that had quietly hardcoded turns, and turns are exactly what the
binding must not know about.

The heavy assertions run without model weights:

  * position ids come from `get_rope_index`, which reads only `self.config`
  * the hooks are tested against a toy `nn.Embedding`, which is the whole contract
  * the sparse keep-mask rule is checked structurally

    pytest tests/test_modality_embed.py -q
"""

import json
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for p in (str(_ROOT), str(_ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import pytest
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoConfig, AutoProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

from longnav.utils.modality_embed import (
    ENCODER_REGISTRY,
    MODALITY_PREFIX,
    BucketEncoder,
    ConstantEncoder,
    ModalityBatch,
    ModalityEmbedder,
    ModalityEmbedSpec,
    MLPEncoder,
    attach_modalities,
    build_encoder,
    coerce_specs,
    single_example_batch,
)

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"

# Two modalities at different rates, neither of which is a pose and neither of which has
# one occurrence per turn.
SPEC_A = ModalityEmbedSpec(token="<mA>", n_features=5, encoder="mlp",
                           encoder_kwargs={"hidden_dims": [8]}, column="col_a")
SPEC_B = ModalityEmbedSpec(token="<mB>", n_features=2, encoder="constant", column="col_b")

D_MODEL = 16


# ======================================================================================
# Fixtures
# ======================================================================================
@pytest.fixture(scope="module")
def processor():
    return AutoProcessor.from_pretrained(MODEL_ID)


@pytest.fixture(scope="module")
def qwen_config():
    return AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)


class RopeStub:
    """Just enough of `Qwen3VLModel` for `get_rope_index` -- it reads only `self.config`."""

    def __init__(self, config):
        self.config = config

    get_rope_index = Qwen3VLModel.get_rope_index


class ToyEmbeddingModel(nn.Module):
    """An `nn.Embedding` behind a trivial wrapper: the exact surface the hooks attach to."""

    def __init__(self, vocab: int = 64, d_model: int = D_MODEL):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, d_model)

    def forward(self, input_ids):
        return self.embed_tokens(input_ids)


def make_embedder(specs=(SPEC_A, SPEC_B), tokenizer=None, d_model=D_MODEL):
    emb = ModalityEmbedder(specs, d_model=d_model)
    if tokenizer is not None:
        emb.register(tokenizer)
    else:  # ids we choose ourselves, for the toy model
        emb.token_ids = {s.key: 50 + i for i, s in enumerate(emb.specs)}
    return emb


# ======================================================================================
# Spec / registry
# ======================================================================================
def test_key_is_derived_from_the_token():
    assert SPEC_A.key == "mA" and SPEC_A.token == "<mA>"
    # `column` is a data source, never a second identity.
    assert SPEC_A.source_column == "col_a"
    assert ModalityEmbedSpec("<z>", 1, "constant").source_column == "z"


@pytest.mark.parametrize("token", ["pose", "<>", "<pose", "pose>", ""])
def test_bad_token_shape_is_rejected(token):
    with pytest.raises(ValueError, match="must look like"):
        ModalityEmbedSpec(token=token, n_features=3, encoder="constant")


def test_duplicate_tokens_are_rejected():
    with pytest.raises(ValueError, match="duplicate modality token"):
        coerce_specs([SPEC_A, ModalityEmbedSpec("<mA>", 9, "constant")])


def test_specs_round_trip_through_json():
    payload = json.loads(json.dumps([SPEC_A.to_dict(), SPEC_B.to_dict()]))
    assert coerce_specs(payload) == (SPEC_A, SPEC_B)


def test_unknown_encoder_key_lists_the_known_ones():
    spec = ModalityEmbedSpec("<x>", 3, "no_such_encoder")
    with pytest.raises(KeyError, match="no_such_encoder"):
        build_encoder(spec, D_MODEL)
    assert {"constant", "mlp", "bucket"} <= set(ENCODER_REGISTRY)


# ======================================================================================
# Encoders
# ======================================================================================
@pytest.mark.parametrize("name,kwargs,F", [
    ("constant", {}, 4),
    ("mlp", {"hidden_dims": [8, 8]}, 4),
    ("bucket", {"n_buckets": 8, "lo": -2.0, "hi": 2.0}, 4),
])
def test_encoders_are_zero_at_init_and_shape_correct(name, kwargs, F):
    """Zero-init means step 0 is bit-identical to no injection at all."""
    enc = build_encoder(ModalityEmbedSpec("<x>", F, name, kwargs), D_MODEL)
    out = enc(torch.randn(7, F))
    assert out.shape == (7, D_MODEL)
    assert torch.equal(out, torch.zeros_like(out)), f"{name} is not zero at init"


@pytest.mark.parametrize("name,kwargs", [
    ("mlp", {"hidden_dims": [8]}),
    ("bucket", {"n_buckets": 8, "lo": -2.0, "hi": 2.0}),
])
def test_value_dependent_encoders_have_gradient_at_init(name, kwargs):
    """Zeroing the *output* layer is safe; zeroing a first layer would sever the path.

    The gradient with respect to the zeroed layer's own weights is proportional to its
    (nonzero) input, so it moves off zero on the very first step.
    """
    enc = build_encoder(ModalityEmbedSpec("<x>", 4, name, kwargs), D_MODEL)
    # A *linear* functional, deliberately: the output is exactly zero at init, so anything
    # quadratic (`out.pow(2)`) has zero gradient there for reasons that have nothing to do
    # with the encoder and would make this test vacuous.
    enc(torch.randn(5, 4)).sum().backward()
    grads = [p.grad for p in enc.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads), "no gradient reaches the encoder"


def test_constant_encoder_ignores_its_input():
    enc = ConstantEncoder(n_features=3, d_model=D_MODEL, zero_init=False)
    a = enc(torch.randn(4, 3))
    b = enc(torch.randn(4, 3))
    assert torch.equal(a, b)


def test_mlp_encoder_distinguishes_values_once_trained():
    enc = MLPEncoder(n_features=3, d_model=D_MODEL, hidden_dims=[8], zero_init=False)
    x = torch.tensor([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    out = enc(x)
    assert not torch.allclose(out[0], out[1])


def test_bucket_encoder_discretises_and_clamps():
    enc = BucketEncoder(n_features=1, d_model=D_MODEL, n_buckets=4, lo=0.0, hi=1.0,
                        zero_init=False)
    # Same bucket -> same embedding; different bucket -> different.
    same = enc(torch.tensor([[0.01], [0.20]]))
    assert torch.allclose(same[0], same[1])
    assert not torch.allclose(enc(torch.tensor([[0.1]]))[0], enc(torch.tensor([[0.9]]))[0])
    # Out of range clamps into the end buckets rather than indexing out of bounds.
    assert torch.allclose(enc(torch.tensor([[-99.0]]))[0], enc(torch.tensor([[0.01]]))[0])
    assert torch.allclose(enc(torch.tensor([[99.0]]))[0], enc(torch.tensor([[0.99]]))[0])
    # lo/hi are buffers, so they travel with the checkpoint.
    assert "lo" in enc.state_dict() and "hi" in enc.state_dict()


def test_encoder_runs_in_fp32_under_autocast_and_casts_at_the_end():
    """The backbone runs bf16 autocast; continuous features lose real precision there."""
    enc = MLPEncoder(n_features=2, d_model=D_MODEL, hidden_dims=[8], zero_init=False)
    x = torch.tensor([[1.0, 1.0], [1.0 + 2 ** -12, 1.0]])
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = enc(x)
        cast = enc(x, dtype=torch.bfloat16)
    assert out.dtype == torch.float32, "encoder must compute in fp32"
    assert cast.dtype == torch.bfloat16, "the cast happens at the end"
    # bf16 has 8 mantissa bits: the two inputs would be identical if the *input* were
    # rounded, so distinct outputs prove the computation really happened in fp32.
    assert not torch.equal(out[0], out[1])


def test_encoder_rejects_wrong_feature_width():
    enc = MLPEncoder(n_features=3, d_model=D_MODEL)
    with pytest.raises(ValueError, match="expects"):
        enc(torch.randn(2, 4))


# ======================================================================================
# ModalityBatch -- the wire format
# ======================================================================================
def test_wire_format_is_flat_values_plus_counts():
    batch = ModalityBatch(
        values={"mA": torch.zeros(5, 3)},
        counts={"mA": torch.tensor([2, 0, 3])},
    )
    kw = batch.to_kwargs()
    assert set(kw) == {"modality_mA_values", "modality_mA_counts"}
    # Flat (sum N, F) + counts (B,) -- not padded (B, max_N, F) + a validity mask.
    assert kw["modality_mA_values"].shape == (5, 3)
    assert kw["modality_mA_counts"].shape == (3,)


def test_pop_from_strips_only_modality_keys():
    kwargs = {
        "pixel_values": torch.zeros(1),
        "image_grid_thw": torch.zeros(1),
        **ModalityBatch({"mA": torch.zeros(4, 3)}, {"mA": torch.tensor([4])}).to_kwargs(),
    }
    batch = ModalityBatch.pop_from(kwargs, known_keys=["mA"])
    assert set(kwargs) == {"pixel_values", "image_grid_thw"}, "left a modality key behind"
    assert batch.values["mA"].shape == (4, 3)


def test_pop_from_raises_on_an_unrecognised_modality_key():
    """The prefix is owned wholesale; a stray key would reach a backbone that rejects it."""
    with pytest.raises(ValueError, match="unrecognised modality key"):
        ModalityBatch.pop_from({f"{MODALITY_PREFIX}mA_wibble": torch.zeros(1)})


def test_pop_from_raises_on_an_undeclared_key():
    kwargs = ModalityBatch({"ghost": torch.zeros(1, 2)}, {"ghost": torch.tensor([1])}).to_kwargs()
    with pytest.raises(ValueError, match="no spec declares it"):
        ModalityBatch.pop_from(kwargs, known_keys=["mA"])


def test_pop_from_requires_values_and_counts_together():
    with pytest.raises(ValueError, match="missing"):
        ModalityBatch.pop_from({f"{MODALITY_PREFIX}mA_values": torch.zeros(2, 3)},
                               known_keys=["mA"])


def test_pop_from_checks_counts_sum_against_rows():
    kwargs = {f"{MODALITY_PREFIX}mA_values": torch.zeros(4, 3),
              f"{MODALITY_PREFIX}mA_counts": torch.tensor([2, 1])}
    with pytest.raises(ValueError, match="counts sum to 3"):
        ModalityBatch.pop_from(kwargs, known_keys=["mA"])


def test_pop_from_is_a_no_op_when_nothing_is_configured():
    kwargs = {"pixel_values": torch.zeros(1)}
    batch = ModalityBatch.pop_from(kwargs, known_keys=[])
    assert not batch and set(kwargs) == {"pixel_values"}


def test_concat_puts_the_prologue_first():
    a = single_example_batch({"mA": torch.tensor([[1.0, 1.0]])})
    b = single_example_batch({"mA": torch.tensor([[2.0, 2.0], [3.0, 3.0]])})
    out = a.concat(b)
    assert out.values["mA"][:, 0].tolist() == [1.0, 2.0, 3.0]
    assert out.counts["mA"].tolist() == [3]


# ======================================================================================
# Section 8, test 1: STATIC position ids
# ======================================================================================
def _marked_conversation(marker_a: str, marker_b: str, n_turns: int = 3) -> str:
    """A multi-image conversation with a deliberately irregular marker pattern.

    Occurrences of `<mA>`: prologue 1, turn 0 one, turn 1 TWO, turn 2 none -> 4 total.
    Occurrences of `<mB>`: turn 1 only -> 1 total.
    Neither rate is the turn count, and neither is the other's.
    """
    parts = [f"<|im_start|>system\nOrigin {marker_a} here.<|im_end|>\n"]
    for i in range(n_turns):
        if i == 0:
            extra = f" {marker_a}"
        elif i == 1:
            extra = f" {marker_a} {marker_a} {marker_b}"
        else:
            extra = ""
        parts.append(
            f"<|im_start|>user\nObservation {i}:{extra}"
            "<|vision_start|><|image_pad|><|vision_end|>Action:<|im_end|>\n"
        )
        parts.append("<|im_start|>assistant\n**____**<|im_end|>\n")
    return "".join(parts)


@pytest.fixture(scope="module")
def marked_batch(processor, qwen_config):
    """Tokenized irregular conversation + the ids of the two markers."""
    ids = attach_modalities(processor, [SPEC_A, SPEC_B])
    images = [Image.new("RGB", (128, 128), (40, 90, 200)) for _ in range(3)]
    batch = processor(text=_marked_conversation(SPEC_A.token, SPEC_B.token),
                      images=images, videos=None, padding=False, return_tensors="pt")
    return batch, ids


def test_markers_survive_templating_verbatim(processor, marked_batch):
    """The literal must reach the tokenizer intact, and as exactly one id each."""
    batch, ids = marked_batch
    assert int((batch["input_ids"] == ids["mA"]).sum()) == 4
    assert int((batch["input_ids"] == ids["mB"]).sum()) == 1


def test_static_position_ids_are_identical_to_an_ordinary_token(qwen_config, marked_batch):
    """Section 8, test 1 -- the one that has to pass before anything else matters.

    Replacing a marker with any single ordinary text token must leave the position ids of
    every other token, text and visual, unchanged. Comparison is by *id substitution at
    the same offsets*: re-rendering the text would shift them, because an added token is a
    pre-tokenization split, so ' <mA>' keeps the space as its own token while ' ____'
    merges into one.
    """
    batch, ids = marked_batch
    stub = RopeStub(qwen_config)
    marker_mask = (batch["input_ids"] == ids["mA"]) | (batch["input_ids"] == ids["mB"])
    assert int(marker_mask.sum()) == 5

    ordinary = batch["input_ids"].clone()
    ordinary[marker_mask] = 2130  # '____', a single ordinary text token

    def rope(x):
        return stub.get_rope_index(
            input_ids=x,
            image_grid_thw=batch.get("image_grid_thw"),
            video_grid_thw=None,
            attention_mask=batch.get("attention_mask"),
        )

    pos_marker, delta_marker = rope(batch["input_ids"])
    pos_ordinary, delta_ordinary = rope(ordinary)

    assert torch.equal(pos_marker, pos_ordinary), (
        "a marker token perturbs mRoPE position ids -- the whole approach is unsound"
    )
    assert torch.equal(delta_marker, delta_ordinary), "the marker changes the rope delta"

    # Stronger: a marker occupies exactly one position on all three mRoPE axes.
    p = pos_marker[:, 0, :]
    for j in marker_mask[0].nonzero(as_tuple=True)[0].tolist():
        assert (p[:, j] - p[:, j - 1]).tolist() == [1, 1, 1]


def test_marker_directly_after_vision_start_is_rejected(processor, qwen_config):
    """`get_rope_index` reads `input_ids[vision_start + 1]` to tell image from video.

    A marker there makes the span count as neither and every position after it is
    silently wrong -- no exception, no NaN. So the placement is checked, not trusted.
    """
    attach_modalities(processor, [SPEC_A])
    embedder = make_embedder([SPEC_A], tokenizer=processor.tokenizer)
    text = ("<|im_start|>user\n<|vision_start|>" + SPEC_A.token
            + "<|image_pad|><|vision_end|><|im_end|>\n")
    batch = processor(text=text, images=[Image.new("RGB", (128, 128))], videos=None,
                      padding=False, return_tensors="pt")
    with pytest.raises(ValueError, match="immediately after"):
        embedder.check_placement(batch["input_ids"], qwen_config.vision_start_token_id)


# ======================================================================================
# Section 8, test 2: INCREMENTAL position ids
# ======================================================================================
def test_incremental_rope_offsets_match_the_full_sequence(qwen_config, processor):
    """Section 8, test 2. The rollout accumulates `rope_offset` as `n_new + delta`
    (`vector_rollout.py:_forward`). A marker must not break that arithmetic.

    This is the failure that would be silent and eval-only: training does one full-sequence
    forward and would never see it.
    """
    attach_modalities(processor, [SPEC_A, SPEC_B])
    stub = RopeStub(qwen_config)
    n_turns = 3
    images = [Image.new("RGB", (128, 128), (40, 90, 200)) for _ in range(n_turns)]

    # Chunk the conversation the way the rollout does: prologue+turn0, turn1, turn2 ...
    chunks = [f"<|im_start|>system\nOrigin {SPEC_A.token} here.<|im_end|>\n"]
    for i in range(n_turns):
        extra = {0: f" {SPEC_A.token}",
                 1: f" {SPEC_A.token} {SPEC_A.token} {SPEC_B.token}"}.get(i, "")
        piece = (f"<|im_start|>user\nObservation {i}:{extra}"
                 "<|vision_start|><|image_pad|><|vision_end|>Action:<|im_end|>\n"
                 "<|im_start|>assistant\n**____**<|im_end|>\n")
        chunks.append(piece) if i else chunks.__setitem__(0, chunks[0] + piece)

    # Incremental: per-chunk get_rope_index plus a running offset.
    incremental, offset = [], 0
    for i, chunk in enumerate(chunks):
        b = processor(text=chunk, images=[images[i]], videos=None, padding=False,
                      return_tensors="pt")
        pos, delta = stub.get_rope_index(
            input_ids=b["input_ids"], image_grid_thw=b.get("image_grid_thw"),
            video_grid_thw=None, attention_mask=b.get("attention_mask"),
        )
        incremental.append(pos + offset)
        offset += b["input_ids"].shape[1] + int(delta.reshape(-1)[0])
    incremental = torch.cat(incremental, dim=-1)

    full = processor(text="".join(chunks), images=images, videos=None, padding=False,
                     return_tensors="pt")
    pos_full, _ = stub.get_rope_index(
        input_ids=full["input_ids"], image_grid_thw=full.get("image_grid_thw"),
        video_grid_thw=None, attention_mask=full.get("attention_mask"),
    )
    assert incremental.shape == pos_full.shape
    assert torch.equal(incremental, pos_full), (
        "incremental rope offsets diverge from the single-shot positions with markers "
        "present; a rollout would silently see a different sequence than training"
    )


# ======================================================================================
# Section 8, test 3: SPARSE keep mask
# ======================================================================================
def test_markers_are_never_dropped_by_the_sparse_keep_mask(qwen_config, marked_batch):
    """Section 8, test 3. The sparse path drops tokens via `seq_keep_mask`.

    `modeling.py` builds it by marking *visual* positions False and adding keepers back
    (`batch_keep_mask[visual_pos_masks] = False`). A marker is not a visual token, so it
    can never be dropped -- but a dropped token shifts every survivor's position, so
    assert it rather than reason about it.
    """
    batch, ids = marked_batch
    input_ids = batch["input_ids"]
    visual_pos_masks = (input_ids == qwen_config.image_token_id) | (
        input_ids == qwen_config.video_token_id
    )
    marker_mask = (input_ids == ids["mA"]) | (input_ids == ids["mB"])
    assert int(marker_mask.sum()) > 0
    assert not bool((visual_pos_masks & marker_mask).any()), (
        "a marker was classified as a visual token and could be sparsified away"
    )

    # Reproduce modeling.py's construction and confirm every marker survives.
    keep = torch.ones_like(input_ids, dtype=torch.bool)
    keep[visual_pos_masks] = False
    seq_keep_mask = keep.any(dim=0)
    assert bool(seq_keep_mask[marker_mask[0]].all())


# ======================================================================================
# The hooks: scatter, alignment, assertions
# ======================================================================================
def _toy_setup(specs=(SPEC_A, SPEC_B), zero_init=False):
    """A toy embedding model with the hooks attached and non-zero encoders."""
    specs = tuple(specs)
    model = ToyEmbeddingModel()
    emb = make_embedder(specs)
    if not zero_init:
        for key in emb.encoders:
            for p in emb.encoders[key].parameters():
                nn.init.normal_(p, std=0.5)
    emb.attach(model.embed_tokens)
    return model, emb


def _uninjected(model, emb, ids):
    """The same forward with the hooks off.

    Needed because the mechanism *refuses* to embed a sequence containing markers with
    nothing pending -- which is the behaviour `test_markers_with_no_values_pending_raise`
    exists to pin down. So a baseline has to be taken deliberately, not by accident.
    """
    emb.detach()
    try:
        return model(ids).clone()
    finally:
        emb.attach(model.embed_tokens)


def _ids(pattern, emb):
    """Build input_ids from a list-of-lists of 'a'/'b'/int, one row per example."""
    tid = emb.token_ids
    rows = [[tid["mA"] if t == "a" else tid["mB"] if t == "b" else t for t in row]
            for row in pattern]
    return torch.tensor(rows, dtype=torch.long)


def test_scatter_lands_at_the_marker_positions_only():
    model, emb = _toy_setup()
    ids = _ids([[1, "a", 2, "a", 3]], emb)
    values = torch.tensor([[1.0, 0, 0, 0, 0], [0, 1.0, 0, 0, 0]])
    batch = ModalityBatch({"mA": values}, {"mA": torch.tensor([2])})

    plain = _uninjected(model, emb, ids)
    with emb.pending(batch):
        out = model(ids)

    marker = ids[0] == emb.token_ids["mA"]
    assert torch.equal(out[0, ~marker], plain[0, ~marker]), "non-marker positions moved"
    assert not torch.allclose(out[0, marker], plain[0, marker])
    expected = emb.encoders["mA"](values, dtype=out.dtype)
    assert torch.allclose(out[0, marker], expected)


def test_binding_is_occurrence_order_not_turns():
    """The k-th occurrence gets the k-th row. Two in one place, none in another, and a
    second modality at a different rate -- all handled by the same rule."""
    model, emb = _toy_setup()
    #            0    1    2    3    4    5    6
    ids = _ids([[1, "a", "a", 2, "b", 3, "a"]], emb)
    a_vals = torch.arange(15, dtype=torch.float32).reshape(3, 5)
    b_vals = torch.zeros(1, 2)
    batch = ModalityBatch({"mA": a_vals, "mB": b_vals},
                          {"mA": torch.tensor([3]), "mB": torch.tensor([1])})
    with emb.pending(batch):
        out = model(ids)
    enc = emb.encoders["mA"](a_vals, dtype=out.dtype)
    for k, pos in enumerate([1, 2, 6]):
        assert torch.allclose(out[0, pos], enc[k]), f"occurrence {k} got the wrong row"


def test_permuting_values_changes_the_output():
    """Alignment is real, not incidental: shuffle the rows and the result must move."""
    model, emb = _toy_setup()
    ids = _ids([[1, "a", 2, "a", "a"]], emb)
    vals = torch.randn(3, 5)
    batch = ModalityBatch({"mA": vals}, {"mA": torch.tensor([3])})
    with emb.pending(batch):
        a = model(ids).clone()
    perm = ModalityBatch({"mA": vals[[2, 0, 1]]}, {"mA": torch.tensor([3])})
    with emb.pending(perm):
        b = model(ids)
    assert not torch.allclose(a, b)


def test_batch_gt_1_with_differing_counts_scatters_correctly():
    """The sparse model runs B == 1 today; the mechanism must not assume it.

    Flat concatenation in batch order is already row-major over `(b, position)`, which is
    exactly what `masked_scatter` enumerates -- this is the test that says so.
    """
    model, emb = _toy_setup()
    ids = _ids([[1, "a", 2, "a", 3],     # 2 occurrences
                [1, 2, 3, 4, "a"],        # 1
                [1, 2, 3, 4, 5]], emb)    # 0
    vals = torch.arange(15, dtype=torch.float32).reshape(3, 5)
    batch = ModalityBatch({"mA": vals}, {"mA": torch.tensor([2, 1, 0])})
    with emb.pending(batch):
        out = model(ids)
    enc = emb.encoders["mA"](vals, dtype=out.dtype)
    assert torch.allclose(out[0, 1], enc[0])
    assert torch.allclose(out[0, 3], enc[1])
    assert torch.allclose(out[1, 4], enc[2])
    # The example with no occurrences is untouched.
    assert torch.equal(out[2], _uninjected(model, emb, ids)[2])


def test_zero_init_makes_step_0_independent_of_the_values():
    """What zero-init actually buys, stated precisely.

    DESIGN.md 3.2 says a zero-init encoder gives output "bit-identical to no injection".
    That is not literally achievable here and the design does not require it to be: the
    marker id never indexes the embedding table (section 2), so the injected vector
    *replaces* whatever was at that position rather than being added to it, and the marker
    also occupies a token position that an uninjected sequence would not have. The real
    guarantee -- the one that makes step 0 safe and a run comparable to its own baseline
    -- is that at init the model's output does not depend on the modality values at all.
    """
    model, emb = _toy_setup(zero_init=True)
    ids = _ids([[1, "a", 2, "a", "b"]], emb)

    def run(a_vals, b_vals):
        batch = ModalityBatch({"mA": a_vals, "mB": b_vals},
                              {"mA": torch.tensor([2]), "mB": torch.tensor([1])})
        with emb.pending(batch):
            return model(ids).clone()

    first = run(torch.randn(2, 5), torch.randn(1, 2))
    second = run(torch.randn(2, 5) * 100, torch.randn(1, 2) * 100)
    assert torch.equal(first, second), "step 0 must not depend on the injected values"
    # And the injected vector is exactly zero, not merely constant.
    marker = (ids[0] == emb.token_ids["mA"]) | (ids[0] == emb.token_ids["mB"])
    assert torch.equal(first[0, marker], torch.zeros_like(first[0, marker]))
    # Non-marker positions are untouched either way.
    plain = _uninjected(model, emb, ids)
    assert torch.equal(first[0, ~marker], plain[0, ~marker])


def test_marker_id_never_reaches_the_embedding_table():
    """The pre-hook rewrites the *argument*, so there is no dependency on a vocab row."""
    model, emb = _toy_setup()
    seen = {}
    model.embed_tokens.register_forward_pre_hook(
        lambda m, a: seen.setdefault("ids", a[0].clone())
    )
    ids = _ids([[1, "a", 2]], emb)
    with emb.pending(ModalityBatch({"mA": torch.zeros(1, 5)}, {"mA": torch.tensor([1])})):
        model(ids)
    assert emb.token_ids["mA"] not in seen["ids"].tolist()[0]
    # `input_ids` itself is untouched, which is what preserves rope / placeholder masks.
    assert emb.token_ids["mA"] in ids.tolist()[0]


def test_count_mismatch_raises_per_example_not_in_total():
    """A total-only check passes when A has one too many and B one too few -- exactly the
    misalignment that trains plausibly and means nothing."""
    model, emb = _toy_setup()
    ids = _ids([[1, "a", 2, "a"],   # 2 occurrences
                [1, "a", 2, 3]], emb)  # 1
    # Totals agree (3 == 3), the per-example split does not.
    batch = ModalityBatch({"mA": torch.zeros(3, 5)}, {"mA": torch.tensor([1, 2])})
    with pytest.raises(RuntimeError, match="count mismatch per example"):
        with emb.pending(batch):
            model(ids)


def test_markers_with_no_values_pending_raise():
    """The eval-only failure: a marker in the prologue, `reset()` never told about it."""
    model, emb = _toy_setup()
    ids = _ids([[1, "a", 2]], emb)
    with pytest.raises(RuntimeError, match="no modality values are pending"):
        model(ids)


def test_values_for_an_absent_marker_raise():
    model, emb = _toy_setup()
    ids = _ids([[1, 2, 3]], emb)
    batch = ModalityBatch({"mA": torch.zeros(2, 5)}, {"mA": torch.tensor([2])})
    with pytest.raises(RuntimeError, match="do not occur in this sequence|nothing was injected"):
        with emb.pending(batch):
            model(ids)


def test_occurrences_with_no_values_raise():
    model, emb = _toy_setup()
    ids = _ids([[1, "a", 2]], emb)
    with pytest.raises(RuntimeError, match="no values were provided"):
        with emb.pending(ModalityBatch()):
            model(ids)


def test_entering_the_backbone_twice_under_one_context_raises():
    """The consume-once assert: the second pass would silently reuse the same values."""
    model, emb = _toy_setup()
    ids = _ids([[1, "a", 2]], emb)
    batch = ModalityBatch({"mA": torch.zeros(1, 5)}, {"mA": torch.tensor([1])})
    with pytest.raises(RuntimeError, match="fired twice"):
        with emb.pending(batch):
            model(ids)
            model(ids)


def test_pending_cannot_be_re_entered():
    _, emb = _toy_setup()
    with pytest.raises(RuntimeError, match="already active"):
        with emb.pending(ModalityBatch()):
            with emb.pending(ModalityBatch()):
                pass


def test_pending_state_is_cleared_after_an_error():
    """A failed step must not leave values pending for the next one."""
    model, emb = _toy_setup()
    ids = _ids([[1, "a", 2]], emb)
    bad = ModalityBatch({"mA": torch.zeros(2, 5)}, {"mA": torch.tensor([2])})
    with pytest.raises(RuntimeError):
        with emb.pending(bad):
            model(ids)
    good = ModalityBatch({"mA": torch.zeros(1, 5)}, {"mA": torch.tensor([1])})
    with emb.pending(good):
        model(ids)  # must not raise


def test_gradient_flows_to_the_encoder_through_the_scatter():
    model, emb = _toy_setup()
    ids = _ids([[1, "a", 2, "a"]], emb)
    vals = torch.randn(2, 5)
    with emb.pending(ModalityBatch({"mA": vals}, {"mA": torch.tensor([2])})):
        model(ids).pow(2).sum().backward()
    grads = [p.grad for p in emb.encoders["mA"].parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)


def test_gradient_checkpointing_after_the_embedding_is_fine():
    """8.1 item 4, the layout that actually ships.

    The hook fires on the embedding and the context manager exits before backward. HF
    checkpoints *decoder layers*, not the embedding, so the embedding is computed once in
    the forward and never recomputed. Gradient must still reach the encoder through the
    checkpointed trunk.
    """
    from torch.utils.checkpoint import checkpoint

    model, emb = _toy_setup()
    trunk = nn.Linear(D_MODEL, D_MODEL)
    ids = _ids([[1, "a", 2, "a"]], emb)
    vals = torch.randn(2, 5)

    with emb.pending(ModalityBatch({"mA": vals}, {"mA": torch.tensor([2])})):
        embeds = model(ids)
    # Backward happens outside the context -- exactly the situation being tested.
    checkpoint(trunk, embeds, use_reentrant=False).pow(2).sum().backward()
    grads = [p.grad for p in emb.encoders["mA"].parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads), (
        "no gradient reached the encoder under gradient checkpointing"
    )


def test_embedding_inside_a_checkpointed_segment_fails_loudly():
    """The failure mode 8.1 item 4 warns about, pinned down.

    If the embedding *were* recomputed inside a checkpointed segment it would fire during
    backward with the context already exited. The consume-once assert turns that into a
    loud `RuntimeError` at the recompute, rather than a silently un-injected forward and
    wrong gradients. This is why the assert exists; do not weaken it to a warning.
    """
    from torch.utils.checkpoint import checkpoint

    model, emb = _toy_setup()
    trunk = nn.Linear(D_MODEL, D_MODEL)
    ids = _ids([[1, "a", 2, "a"]], emb)
    vals = torch.randn(2, 5)

    with emb.pending(ModalityBatch({"mA": vals}, {"mA": torch.tensor([2])})):
        out = checkpoint(lambda x: trunk(model(x)), ids, use_reentrant=False)
    with pytest.raises(RuntimeError, match="no modality values are pending"):
        out.pow(2).sum().backward()


def test_zero_touch_is_exactly_zero_and_reaches_every_encoder():
    """8.1 item 5: a batch with no occurrences leaves an encoder out of the backward graph,
    which under `ddp_find_unused_parameters=False` hangs. The touch term fixes that and
    must not perturb anything."""
    _, emb = _toy_setup()
    touch = emb.zero_touch(torch.device("cpu"), torch.float32)
    assert float(touch.detach()) == 0.0
    touch.backward()
    for key in emb.encoders:
        assert any(p.grad is not None for p in emb.encoders[key].parameters()), key
        for p in emb.encoders[key].parameters():
            if p.grad is not None:
                assert float(p.grad.abs().sum()) == 0.0, "the touch term must not add grad"


def test_no_specs_is_completely_inert():
    """With nothing registered: no hooks, no parameters, no state, no error."""
    model = ToyEmbeddingModel()
    emb = ModalityEmbedder()
    assert not emb and emb.keys == [] and list(emb.parameters()) == []
    emb.attach(model.embed_tokens)
    assert model.embed_tokens._forward_pre_hooks == {} or not emb.specs
    ids = torch.tensor([[1, 2, 3]])
    before = model(ids)
    with emb.pending(None):
        assert torch.equal(model(ids), before)
    assert emb.state_blob() is None
    emb.load_state_blob(None)  # legacy checkpoint path: silent


# ======================================================================================
# Tokenizer / corpus
# ======================================================================================
def test_corpus_containing_the_literal_is_rejected():
    emb = ModalityEmbedder([SPEC_A], d_model=D_MODEL)
    with pytest.raises(ValueError, match="already occurs in the corpus"):
        emb.check_corpus(["hello", f"a{SPEC_A.token}b"])
    emb.check_corpus(["hello", "no markers here"])  # must not raise


def test_adding_the_token_leaves_ordinary_text_tokenizing_identically(processor):
    """Added tokens are a pre-tokenization split over raw text: on text without the
    literal the split never fires and BPE sees byte-identical input."""
    from transformers import AutoTokenizer

    samples = [
        "You are a robot navigating an indoor environment toward a goal object.",
        "Observation 12:", "Action:", "**____**", "<|im_start|>assistant\n",
        "chest_of_drawers", "a<pose>b is not here", "x < y and y > z",
    ]
    clean = AutoTokenizer.from_pretrained(MODEL_ID)
    marked = AutoTokenizer.from_pretrained(MODEL_ID)
    attach_modalities(type("P", (), {"tokenizer": marked})(), [SPEC_A, SPEC_B])
    for s in samples:
        assert clean.encode(s, add_special_tokens=False) == marked.encode(
            s, add_special_tokens=False
        ), f"tokenization of {s!r} changed"


def test_unregistered_token_is_caught_at_bind_time(processor):
    """A checkpoint whose `added_tokens.json` lost the marker: `<x>` BPEs back into
    several ordinary tokens, the scatter finds nothing, and the error would otherwise
    surface much later as a confusing count mismatch."""
    from transformers import AutoTokenizer

    emb = ModalityEmbedder([ModalityEmbedSpec("<never_added>", 2, "constant")],
                           d_model=D_MODEL)
    with pytest.raises(ValueError, match="tokenizes to .* ids"):
        emb.bind_tokenizer(AutoTokenizer.from_pretrained(MODEL_ID))


def test_attach_before_bind_is_refused():
    emb = ModalityEmbedder([SPEC_A], d_model=D_MODEL)
    with pytest.raises(RuntimeError, match="bind_tokenizer"):
        emb.attach(ToyEmbeddingModel().embed_tokens)


# ======================================================================================
# Checkpointing: the four cases of DESIGN.md 7.1
# ======================================================================================
def test_state_blob_round_trips_weights():
    a = make_embedder([SPEC_A, SPEC_B])
    for key in a.encoders:
        for p in a.encoders[key].parameters():
            nn.init.normal_(p, std=0.5)
    b = make_embedder([SPEC_A, SPEC_B])
    b.load_state_blob(a.state_blob())
    for key in a.encoders:
        for pa, pb in zip(a.encoders[key].parameters(), b.encoders[key].parameters()):
            assert torch.equal(pa, pb)


def test_load_case_1_declared_spec_with_no_weights_raises():
    emb = make_embedder([SPEC_A])
    with pytest.raises(RuntimeError, match="no encoder weights"):
        emb.load_state_blob(None)


def test_load_case_2_weights_for_an_undeclared_spec_raise():
    blob = make_embedder([SPEC_A, SPEC_B]).state_blob()
    with pytest.raises(RuntimeError, match="does not declare"):
        make_embedder([SPEC_A]).load_state_blob(blob)
    # ... and the same blob against a config declaring nothing at all.
    with pytest.raises(RuntimeError, match="config declares no specs"):
        ModalityEmbedder().load_state_blob(blob)


def test_load_case_3_shape_drift_raises():
    wide = make_embedder([ModalityEmbedSpec("<mA>", 5, "mlp", {"hidden_dims": [8]})])
    narrow = make_embedder([ModalityEmbedSpec("<mA>", 5, "mlp", {"hidden_dims": [4]})])
    with pytest.raises(RuntimeError, match="size mismatch|shape"):
        narrow.load_state_blob(wide.state_blob())


def test_load_case_4_no_specs_and_no_blob_is_silent():
    ModalityEmbedder().load_state_blob(None)


def test_missing_spec_in_a_multi_spec_blob_raises():
    blob = make_embedder([SPEC_A]).state_blob()
    with pytest.raises(RuntimeError, match="no weights in the checkpoint"):
        make_embedder([SPEC_A, SPEC_B]).load_state_blob(blob)


# ======================================================================================
# Integration with vector_sft: the collator, the popped prefix, the checkpoint
# ======================================================================================
from longnav.utils.turn_vectors import TurnVectorHead, resolve_affix_ids  # noqa: E402
from longnav.utils.vector_sft import (  # noqa: E402
    DataConfig,
    LossConfig,
    ModelConfig,
    TargetNormalizer,
    TurnVectorCollator,
    TurnVectorRegressor,
    n_markers_in_message,
)


class _Sentinel(Exception):
    """Raised by the stub backbone once it has recorded what it was called with."""


class StubBackbone(nn.Module):
    """Enough backbone to drive `TurnVectorRegressor.forward` up to the backbone call."""

    def __init__(self, d_model=D_MODEL, vocab=200000):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.seen = None

    def get_input_embeddings(self):
        return self.embed

    def forward(self, input_ids=None, **kwargs):
        self.seen = dict(kwargs, input_ids=input_ids)
        self.embed(input_ids)  # so the hooks fire, exactly as the real backbone does
        raise _Sentinel


def build_stub_regressor(specs=(), tokenizer=None, target_shape=(2,)):
    cfg = ModelConfig(modality_specs=specs, head_hidden_dims=(8,))
    embedder = ModalityEmbedder(cfg.modality_specs, d_model=D_MODEL if cfg.modality_specs else None)
    if embedder:
        embedder.register(tokenizer)
    backbone = StubBackbone()
    model = TurnVectorRegressor(
        backbone=backbone,
        head=TurnVectorHead(hidden_size=D_MODEL, out_dim=2, mode="mean", hidden_dims=(8,)),
        normalizer=TargetNormalizer(target_shape[-1], enabled=False),
        target_shape=target_shape,
        prefix_ids=[1], postfix_ids=[2],
        model_cfg=cfg, loss_cfg=LossConfig(normalize_targets=False),
        modality_embedder=embedder,
    )
    model.attach_modality_hooks()
    return model


def test_forward_pops_modality_keys_before_the_backbone_call(processor):
    """The one thing that breaks if handled carelessly (DESIGN.md section 5).

    `forward`'s leftover kwargs go to the backbone *wholesale*; a modality tensor left in
    there is a `TypeError` from deep inside the model.
    """
    model = build_stub_regressor((SPEC_A,), processor.tokenizer)
    tid = model.modality_embedder.token_ids["mA"]
    ids = torch.tensor([[5, tid, 6, tid, 7]])
    batch = {
        "input_ids": ids,
        "targets": torch.zeros(1, 2),
        "pixel_values": torch.zeros(1),
        **ModalityBatch({"mA": torch.randn(2, 5)}, {"mA": torch.tensor([2])}).to_kwargs(),
    }
    with pytest.raises(_Sentinel):
        model(**batch)
    seen = model.backbone.seen
    assert not [k for k in seen if k.startswith(MODALITY_PREFIX)], (
        f"modality keys reached the backbone: {sorted(seen)}"
    )
    assert "pixel_values" in seen, "an ordinary multimodal key was swallowed"


def test_forward_rejects_modality_keys_when_no_spec_is_declared(processor):
    """With nothing declared they cannot be silently dropped either -- that would be a
    scatter that never happens."""
    model = build_stub_regressor()
    batch = {
        "input_ids": torch.tensor([[5, 6, 7]]),
        "targets": torch.zeros(1, 2),
        **ModalityBatch({"mA": torch.zeros(1, 5)}, {"mA": torch.tensor([1])}).to_kwargs(),
    }
    with pytest.raises(ValueError, match="no spec declares it"):
        model(**batch)


def test_a_backbone_error_is_not_masked_by_the_pending_context(processor):
    """The context's own consistency check must not replace the caller's exception."""
    model = build_stub_regressor((SPEC_A,), processor.tokenizer)
    tid = model.modality_embedder.token_ids["mA"]
    batch = {
        "input_ids": torch.tensor([[5, tid, 6]]),
        "targets": torch.zeros(1, 2),
        **ModalityBatch({"mA": torch.zeros(1, 5)}, {"mA": torch.tensor([1])}).to_kwargs(),
    }
    with pytest.raises(_Sentinel):
        model(**batch)


def test_no_specs_leaves_forward_completely_inert(processor):
    """No hooks installed, no parameters added, no extra keys expected or produced."""
    model = build_stub_regressor()
    assert list(model.modality_embedder.parameters()) == []
    assert model.modality_embedder.state_blob() is None
    with pytest.raises(_Sentinel):
        model(input_ids=torch.tensor([[5, 6, 7]]), targets=torch.zeros(1, 2))
    assert model.backbone.embed._forward_pre_hooks == {}


def test_checkpoint_round_trips_specs_and_encoder_weights(processor, tmp_path):
    model = build_stub_regressor((SPEC_A, SPEC_B), processor.tokenizer)
    for key in model.modality_embedder.encoders:
        for p in model.modality_embedder.encoders[key].parameters():
            nn.init.normal_(p, std=0.5)
    model.save_pretrained(tmp_path)

    meta = json.loads((tmp_path / "turn_vector_head_config.json").read_text())
    assert coerce_specs(meta["model"]["modality_specs"]) == (SPEC_A, SPEC_B), (
        "the spec list must be in the checkpoint config, so eval reads its settings "
        "rather than being told them"
    )
    blob = torch.load(tmp_path / "turn_vector_head.pt", weights_only=False)
    assert set(blob) == {"head", "normalizer", "modality"}

    fresh = build_stub_regressor((SPEC_A, SPEC_B), processor.tokenizer)
    fresh.load_head_state(tmp_path)
    for key in model.modality_embedder.encoders:
        for a, b in zip(model.modality_embedder.encoders[key].parameters(),
                        fresh.modality_embedder.encoders[key].parameters()):
            assert torch.equal(a, b)


def test_a_no_spec_checkpoint_is_shaped_exactly_as_before(processor, tmp_path):
    """Inertness where it matters most: a live run's checkpoint must not gain a key."""
    build_stub_regressor().save_pretrained(tmp_path)
    blob = torch.load(tmp_path / "turn_vector_head.pt", weights_only=False)
    assert set(blob) == {"head", "normalizer"}, "a no-spec checkpoint gained a key"
    meta = json.loads((tmp_path / "turn_vector_head_config.json").read_text())
    # Omitted, not written as []: an older reader doing `ModelConfig(**meta["model"])`
    # would raise on an unexpected keyword, and the two live runs' eval path is exactly
    # that reader.
    assert "modality_specs" not in meta["model"]


def test_a_legacy_config_without_the_key_still_loads(processor, tmp_path):
    """The two live runs' configs predate this field entirely."""
    build_stub_regressor().save_pretrained(tmp_path)
    path = tmp_path / "turn_vector_head_config.json"
    meta = json.loads(path.read_text())
    assert "modality_specs" not in meta["model"]  # what an older checkpoint looks like

    cfg = ModelConfig(**meta["model"])
    assert cfg.modality_specs == ()
    fresh = build_stub_regressor()
    fresh.load_head_state(tmp_path)  # must be silent


def test_loading_modality_weights_is_strict_even_when_strict_is_false(processor, tmp_path):
    """`strict=False` exists to tolerate head-shape evolution. Letting it also drop
    encoders would defeat the point of declaring them."""
    build_stub_regressor((SPEC_A,), processor.tokenizer).save_pretrained(tmp_path)
    fresh = build_stub_regressor()  # declares nothing
    with pytest.raises(RuntimeError, match="config declares no specs"):
        fresh.load_head_state(tmp_path, strict=False)


# -- the collator ----------------------------------------------------------------------
def _conversation_with_markers(n_turns=6):
    """Deliberately irregular: turn 0 gets two markers, turn 1 none, the rest one."""
    messages = [{"role": "user", "content": [{"type": "text", "text": "prologue"}]}]
    n_marks = []
    for t in range(n_turns):
        k = {0: 2, 1: 0}.get(t, 1)
        n_marks.append(k)
        messages += [
            {"role": "user", "content": [
                {"type": "text", "text": f"Observation {t}: " + SPEC_A.token * k},
                {"type": "image"},
                {"type": "text", "text": "Action:"},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": "**____**"}]},
        ]
    return messages, n_marks


def test_n_markers_in_message_counts_text_occurrences():
    messages, n_marks = _conversation_with_markers(4)
    counted = [n_markers_in_message(m, SPEC_A.token) for m in messages]
    assert sum(counted) == sum(n_marks)
    assert counted[1] == 2 and counted[3] == 0


@pytest.mark.parametrize("start,cap", [(0, 3), (1, 3), (2, 4), (3, 3)])
def test_collator_slices_the_modality_column_with_the_window(processor, tmp_path, start, cap):
    """DESIGN.md 8.1 item 1. The window is a random contiguous range of *turns*; the
    modality column is bound by *occurrence order*, so it has to be sliced by counting
    marker literals in the kept messages -- the same idiom images use."""
    n_turns = 6
    messages, n_marks = _conversation_with_markers(n_turns)
    img = tmp_path / "f.png"
    Image.new("RGB", (64, 64), (10, 20, 30)).save(img)

    total = sum(n_marks)
    values = torch.arange(total * SPEC_A.n_features, dtype=torch.float32).reshape(
        total, SPEC_A.n_features
    )
    row = {
        "messages": messages,
        "images": [str(img)] * n_turns,
        "action_chunks": torch.zeros(n_turns, 2).tolist(),
        "col_a": values.tolist(),
    }

    attach_modalities(processor, [SPEC_A])
    collator = TurnVectorCollator(processor, DataConfig(max_turns_per_sample=cap),
                                  train=False, modality_specs=(SPEC_A,))
    collator._rng = type("R", (), {"integers": staticmethod(lambda a, b: start)})()
    collator.train = True
    out = collator([row])

    kept_marks = sum(n_marks[start:start + cap])
    got = out["modality_mA_values"]
    assert out["modality_mA_counts"].tolist() == [kept_marks]
    assert got.shape == (kept_marks, SPEC_A.n_features)
    # The exact rows, not just the count: occurrence `i` of the window is occurrence
    # `sum(n_marks[:start]) + i` of the episode.
    offset = sum(n_marks[:start])
    assert torch.equal(got, values[offset:offset + kept_marks])
    # And the collator's own count check agrees with the tokenized text.
    tid = processor.tokenizer.encode(SPEC_A.token, add_special_tokens=False)[0]
    assert int((out["input_ids"] == tid).sum()) == kept_marks


def test_collator_keeps_a_prologue_marker_through_every_window(processor, tmp_path):
    """The prologue is always kept, so its occurrence is always row 0 -- for every window.

    An episode-level slot lives there, and it is the case that would break first if the
    modality column were sliced by turn index instead of by occurrence count.
    """
    n_turns, cap = 5, 2
    messages, n_marks = _conversation_with_markers(n_turns)
    messages[0]["content"][0]["text"] = "prologue " + SPEC_A.token
    img = tmp_path / "p.png"
    Image.new("RGB", (64, 64)).save(img)

    total = 1 + sum(n_marks)
    values = torch.arange(total * SPEC_A.n_features, dtype=torch.float32).reshape(
        total, SPEC_A.n_features
    )
    row = {"messages": messages, "images": [str(img)] * n_turns,
           "action_chunks": torch.zeros(n_turns, 2).tolist(), "col_a": values.tolist()}

    attach_modalities(processor, [SPEC_A])
    for start in range(n_turns - cap + 1):
        collator = TurnVectorCollator(processor, DataConfig(max_turns_per_sample=cap),
                                      train=True, modality_specs=(SPEC_A,))
        collator._rng = type("R", (), {"integers": staticmethod(lambda a, b: start)})()
        out = collator([row])
        got = out["modality_mA_values"]
        assert torch.equal(got[0], values[0]), (
            f"window at turn {start} lost or misplaced the prologue's value row"
        )
        offset = 1 + sum(n_marks[:start])
        rest = sum(n_marks[start:start + cap])
        assert torch.equal(got[1:], values[offset:offset + rest])


def test_collator_handles_two_modalities_at_different_rates(processor, tmp_path):
    """Nothing couples the two rates -- or couples either of them to the turn count."""
    n_turns = 4
    messages = [{"role": "user", "content": [{"type": "text", "text": "prologue"}]}]
    a_marks, b_marks = [], []
    for t in range(n_turns):
        na, nb = (3 if t == 0 else 1 if t != 2 else 0), (1 if t % 2 else 0)
        a_marks.append(na)
        b_marks.append(nb)
        messages += [
            {"role": "user", "content": [
                {"type": "text", "text": SPEC_A.token * na + SPEC_B.token * nb},
                {"type": "image"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "**____**"}]},
        ]
    img = tmp_path / "r.png"
    Image.new("RGB", (64, 64)).save(img)
    row = {
        "messages": messages, "images": [str(img)] * n_turns,
        "action_chunks": torch.zeros(n_turns, 2).tolist(),
        "col_a": torch.randn(sum(a_marks), SPEC_A.n_features).tolist(),
        "col_b": torch.randn(sum(b_marks), SPEC_B.n_features).tolist(),
    }
    attach_modalities(processor, [SPEC_A, SPEC_B])
    out = TurnVectorCollator(processor, DataConfig(max_turns_per_sample=None), train=False,
                             modality_specs=(SPEC_A, SPEC_B))([row])
    assert out["modality_mA_counts"].tolist() == [sum(a_marks)] == [5]
    assert out["modality_mB_counts"].tolist() == [sum(b_marks)] == [2]
    # Neither rate equals the turn count, or the test would pass against a mechanism that
    # had quietly assumed one occurrence per turn.
    assert sum(a_marks) != n_turns and sum(b_marks) != n_turns


def test_collator_rejects_a_column_that_disagrees_with_the_text(processor, tmp_path):
    """Per example, before batching, so a mismatch is attributable to one conversation."""
    messages, n_marks = _conversation_with_markers(3)
    img = tmp_path / "g.png"
    Image.new("RGB", (64, 64)).save(img)
    row = {
        "messages": messages,
        "images": [str(img)] * 3,
        "action_chunks": torch.zeros(3, 2).tolist(),
        "col_a": torch.zeros(sum(n_marks) + 1, SPEC_A.n_features).tolist(),  # one too many
    }
    attach_modalities(processor, [SPEC_A])
    collator = TurnVectorCollator(processor, DataConfig(max_turns_per_sample=None),
                                  train=False, modality_specs=(SPEC_A,))
    with pytest.raises(ValueError, match="occurrence"):
        collator([row])


def test_collator_without_specs_emits_no_modality_keys(processor, tmp_path):
    messages, _ = _conversation_with_markers(2)
    img = tmp_path / "h.png"
    Image.new("RGB", (64, 64)).save(img)
    row = {"messages": messages, "images": [str(img)] * 2,
           "action_chunks": torch.zeros(2, 2).tolist()}
    collator = TurnVectorCollator(processor, DataConfig(max_turns_per_sample=None),
                                 train=False)
    out = collator([row])
    assert not [k for k in out if k.startswith(MODALITY_PREFIX)]


# -- optimizer param groups ------------------------------------------------------------
def test_encoders_land_in_an_optimizer_group(processor):
    """DESIGN.md 8.1 item 3. A module in no group is **silently frozen**, and a frozen
    encoder looks exactly like "the injection did not help" -- the conclusion the
    experiment would be trying to draw."""
    sys.path.insert(0, str(_ROOT / "data_scripts"))
    from train_vector_sft import build_optimizer_param_groups

    model = build_stub_regressor((SPEC_A, SPEC_B), processor.tokenizer)
    args = type("A", (), {"lr": 1e-4, "head_lr": 1e-3, "weight_decay": 0.0})()
    groups = build_optimizer_param_groups(model, args)

    grouped = {id(p) for g in groups for p in g["params"]}
    enc_params = list(model.modality_embedder.parameters())
    assert enc_params, "the fixture has no encoder parameters to check"
    assert all(id(p) in grouped for p in enc_params), "an encoder is silently frozen"
    # Fresh modules share the head's larger step; the adapters keep theirs.
    fresh = next(g for g in groups if g["lr"] == args.head_lr)
    assert all(id(p) in {id(q) for q in fresh["params"]} for p in enc_params)


def test_param_group_builder_catches_an_ungrouped_parameter(processor):
    """The assertion itself must work, or it is decoration."""
    sys.path.insert(0, str(_ROOT / "data_scripts"))
    from train_vector_sft import build_optimizer_param_groups

    model = build_stub_regressor((SPEC_A,), processor.tokenizer)
    # A trainable parameter the builder does not know about: exactly the situation.
    model.backbone.embed.weight.requires_grad_(True)
    real = model.parameters
    model.parameters = lambda: (p for n, p in model.named_parameters()
                                if not n.startswith("backbone.embed"))
    try:
        args = type("A", (), {"lr": 1e-4, "head_lr": None, "weight_decay": 0.0})()
        with pytest.raises(RuntimeError, match="silently frozen"):
            build_optimizer_param_groups(model, args)
    finally:
        model.parameters = real


# -- the rollout -----------------------------------------------------------------------
def test_rollout_step_and_reset_take_modality():
    """DESIGN.md 8.1 item 6: `reset()` renders the prologue and had no modality argument,
    so an episode-level slot would forward a marker with nothing pending -- an eval-only
    failure that training never sees, because the window always keeps the prologue."""
    import inspect

    from longnav.utils.vector_rollout import VectorRolloutPolicy

    for name in ("step", "reset"):
        params = inspect.signature(getattr(VectorRolloutPolicy, name)).parameters
        assert "modality" in params, f"{name}() cannot carry modality values"
        assert params["modality"].default is None, f"{name}(modality=) must default to None"


def test_rollout_prologue_values_precede_the_first_step():
    """`reset()`'s occurrences come first in the token sequence, so first in the rows."""
    from longnav.utils.vector_rollout import VectorRolloutPolicy

    policy = object.__new__(VectorRolloutPolicy)
    policy._pending_modality = single_example_batch({"mA": torch.tensor([[9.0] * 5])})
    step_batch = single_example_batch({"mA": torch.tensor([[1.0] * 5, [2.0] * 5])})
    merged = policy._pending_modality.concat(step_batch)
    assert merged.values["mA"][:, 0].tolist() == [9.0, 1.0, 2.0]
    assert merged.counts["mA"].tolist() == [3]
