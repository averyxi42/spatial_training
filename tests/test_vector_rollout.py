"""
Tests for `longnav.utils.vector_rollout`.

The cheap ones (no model) check the property the whole rollout rests on: the text the
policy builds turn by turn must tokenize *identically* to the single-shot
`apply_chat_template` of the same conversation that `format_action_chunk_dataset.py`
writes for training. If those two ever diverge, the head is pooling a `**` token with a
different left context than it trained on, and nothing errors -- accuracy just quietly
drops. Concatenating tokenizations of substrings is not generally equal to tokenizing the
whole string (BPE merges across boundaries), so this is asserted on ids, not eyeballed.

The GPU test rolls a trained checkpoint out step by step and compares its per-step chunks
against a single-shot forward over the same conversation -- the incremental-vs-batch
parity check, the analogue of "live vs cache" for the deploy demo.

    pytest tests/test_vector_rollout.py -k "not checkpoint"
    pytest tests/test_vector_rollout.py --ckpt dump/vector_sft_3090/final
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for p in (str(_ROOT), str(_ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import pytest
import torch
from transformers import AutoProcessor

from longnav.utils.vector_rollout import (
    DEFAULT_SYSTEM_PROMPT,
    RolloutConfig,
    assistant_tail,
    full_context_text,
    render_turn,
)

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"


@pytest.fixture(scope="module")
def processor():
    return AutoProcessor.from_pretrained(MODEL_ID)


def training_conversation(n_turns, goal="chest_of_drawers", placeholder="**forward**"):
    """Exactly what `format_action_chunk_dataset.build_messages` produces."""
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": DEFAULT_SYSTEM_PROMPT.format(goal=goal)}],
        }
    ]
    for i in range(n_turns):
        messages += [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Observation {i}:"},
                    {"type": "image"},
                    {"type": "text", "text": "Action:"},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": placeholder}]},
        ]
    return messages


@pytest.mark.parametrize("n_turns", [1, 3, 8])
def test_rollout_text_matches_training_template(processor, n_turns):
    cfg = RolloutConfig()
    goal = "chest_of_drawers"
    rollout_text = full_context_text(
        processor, cfg, n_turns, DEFAULT_SYSTEM_PROMPT.format(goal=goal)
    )
    training_text = processor.apply_chat_template(
        training_conversation(n_turns, goal), tokenize=False, add_generation_prompt=False
    )
    assert rollout_text == training_text

    # And token-exact, which is the claim that actually matters.
    tok = processor.tokenizer
    assert tok.encode(rollout_text, add_special_tokens=False) == tok.encode(
        training_text, add_special_tokens=False
    )


def test_incremental_chunks_concatenate_to_the_same_ids(processor):
    """Per-turn tokenization must not shift at the chunk boundaries.

    Each forward pass tokenizes only its own chunk, so if BPE merged differently across
    a boundary the cached context would not equal the single-shot sequence.
    """
    cfg = RolloutConfig()
    tok = processor.tokenizer
    n_turns = 4
    prologue = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": DEFAULT_SYSTEM_PROMPT.format(goal="g")}]}],
        tokenize=False,
        add_generation_prompt=False,
    )
    chunks, pending = [], prologue
    for i in range(n_turns):
        chunks.append(pending + render_turn(processor, cfg, i))
        pending = assistant_tail(cfg)

    per_chunk = [tok.encode(c, add_special_tokens=False) for c in chunks]
    incremental = [t for c in per_chunk for t in c]
    single_shot = tok.encode(
        full_context_text(processor, cfg, n_turns, DEFAULT_SYSTEM_PROMPT.format(goal="g")),
        add_special_tokens=False,
    )
    # The rollout stops at the last turn's `**`, so it is a strict prefix of the full
    # conversation (which also has that turn's closing text).
    assert incremental == single_shot[: len(incremental)]
    assert tok.decode(incremental[-1:]) == "**", "the pooled position must be the '**' token"


def test_pooled_position_is_the_last_token(processor):
    """The reason `step()` can read `last_hidden_state[:, -1]` with no span search."""
    cfg = RolloutConfig()
    ids = processor.tokenizer.encode(render_turn(processor, cfg, 0), add_special_tokens=False)
    assert processor.tokenizer.decode(ids[-1:]) == "**"


def test_placeholder_must_open_with_the_pooled_token():
    from longnav.utils.vector_rollout import VectorRolloutPolicy

    class _FakeModel:
        class model_cfg:
            affixes = "action"
            shift_left = True

    with pytest.raises(ValueError, match="must start with"):
        VectorRolloutPolicy.__init__(
            object.__new__(VectorRolloutPolicy),
            _FakeModel(),
            None,
            RolloutConfig(placeholder="forward"),
        )


def test_rejects_content_pooling_models():
    """A model trained to pool assistant content cannot be rolled out incrementally."""
    from longnav.utils.vector_rollout import VectorRolloutPolicy

    class _FakeModel:
        class model_cfg:
            affixes = "template"
            shift_left = False

    with pytest.raises(ValueError, match="affixes='action'"):
        VectorRolloutPolicy.__init__(
            object.__new__(VectorRolloutPolicy), _FakeModel(), None, RolloutConfig()
        )


# ======================================================================================
# Checkpoint test: incremental rollout vs one single-shot forward
# ======================================================================================
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_checkpoint_rollout_matches_single_shot(request):
    """Roll out N steps and compare against pooling the same conversation in one pass.

    They are not bit-identical -- the sparsifier dedups a frame against a growing
    database incrementally versus all frames at once, and merging LoRA changes the
    numerics slightly -- so this asserts closeness, and prints the deviation.
    """
    ckpt = request.config.getoption("--ckpt")
    if not ckpt:
        pytest.skip("pass --ckpt <checkpoint dir> to run the parity check")

    from datasets import load_from_disk

    from longnav.utils.turn_vectors import extract_turn_vectors
    from longnav.utils.vector_rollout import VectorRolloutPolicy
    from longnav.utils.vector_sft import load_image

    n_turns = 6
    policy = VectorRolloutPolicy.from_checkpoint(ckpt, RolloutConfig(merge_lora=True))
    proc = policy.processor
    ds = load_from_disk("dump/datasets/action_chunks_conversational")["validation"]
    row = ds[0]
    images = [load_image(p) for p in row["images"][:n_turns]]
    goal = row.get("goal_text") or "the goal object"

    policy.reset(goal_text=goal)
    incremental = torch.stack([policy.step(im) for im in images])

    # Single-shot: the same conversation, pooled with the training-time code path.
    messages = training_conversation(n_turns, goal)
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    inputs = proc(text=text, images=images, videos=None, padding=False, return_tensors="pt")
    with torch.inference_mode():
        outputs = policy.model.backbone(
            **{k: v.to(policy.cfg.device) for k, v in inputs.items()},
            use_cache=False,
            logits_to_keep=1,
        )
        vectors, spans = extract_turn_vectors(
            outputs,
            inputs["input_ids"],
            policy.model.head,
            prefix_ids=policy.model.prefix_ids,
            postfix_ids=policy.model.postfix_ids,
            shift_left=True,
            strict=True,
        )
    single = policy.model.normalizer.denormalize(
        vectors.view(-1, *policy.model.target_shape)
    ).float().cpu()

    assert len(spans) == n_turns
    err = (incremental - single).abs()
    scale = single.abs().mean().clamp(min=1e-6)
    print(f"\nincremental vs single-shot: max abs {err.max():.4f}, mean abs "
          f"{err.mean():.4f}, target scale {scale:.4f} "
          f"(relative {float(err.mean() / scale):.3%})")
    assert err.mean() < 0.25 * scale, (
        "incremental rollout drifted from the single-shot forward; the cached context "
        "or the rope offsets do not match the training-time sequence"
    )
