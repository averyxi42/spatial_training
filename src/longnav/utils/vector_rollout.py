"""
Minimal turn-by-turn rollout for a model trained by `vector_sft.py`.

`VLMWorker` (see `vlm_worker.py`) does incremental multi-turn inference, but it is built
for the RL stack: Ray actors, logprobs, value heads, adapter merge/unmerge bookkeeping,
rollout buffers. This is the same KV-cache mechanics with none of that -- feed one
observation, get one continuous action chunk, repeat. Nothing here trains, distributes or
logs. `vlm_worker.py` is untouched.

Turn structure is described the way `VLMWorker` describes it: give it the affix strings
and the constant placeholder, and it derives the rest. `split_assistant_turn` runs the
*training-time* `find_turn_spans` over the assistant turn to locate the readout positions,
splits the turn's tokens there into what to emit now and what is owed to the next forward,
and reports how many positions the head will be handed. `_assert_head_matches_readout`
then checks that count against the head, because a pooled head accepts the wrong number of
positions without complaint. Nothing hardcodes `**`, a single readout token, or a
particular placeholder -- change `RolloutConfig.prefix/postfix/placeholder` and the split
follows, or fails loudly if that combination cannot be located (e.g. `'**[…]**'`, where BPE
merges `]` into the closing `**`).

Because the emitted block ends exactly at the last readout position, `step()` reads
`last_hidden_state[:, -n_readout:]` with no span search, and no generation is needed -- the
action for step k is available as soon as step k's image is encoded. Those positions are
text tokens, which the sparsifier never drops.

Composition is done in token ids rather than text: the user block comes from the processor
(which expands the image placeholder) and the assistant pieces are constant id lists, so
concatenation cannot merge differently than the training-time single-shot tokenization did.

Per-step mechanics, mirroring `VLMWorker.infer_step`:

  * only the new tokens are forwarded; `past_key_values` carries the rest.
  * `position_ids` come from `get_rope_index` on this turn's tokens plus a running
    offset advanced by the turn length and the rope delta (`_pos_id_fast`'s arithmetic).
  * with the sparse backbone, `attention_mask=None` and the visual embedding database
    (`past_image_embeds` / `save_image_db`) is threaded turn to turn, so redundant visual
    tokens from earlier frames are dropped exactly as they are in a single-shot pass.

Text composition is *token-exact* against the training-time single-shot tokenization --
each turn's chunk is the previous assistant turn's tail plus this turn's user block plus
the assistant opening, and concatenating the chunks reproduces
`apply_chat_template(whole_conversation)` byte for byte. `tests/test_vector_rollout.py`
asserts this; if it ever drifts, the rollout context stops matching what the model saw in
training and the head silently degrades.

    policy = VectorRolloutPolicy.from_checkpoint("dump/vector_sft_3090/final")
    print(policy.describe())                # affixes, emitted tokens, readout width
    policy.reset(goal_text="chest_of_drawers")
    for image in observations:
        chunk = policy.step(image)          # (N_action, 3) in the target's own units
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import torch

from longnav.utils.modality_embed import ModalityBatch, single_example_batch
from longnav.utils.vector_sft import TurnVectorRegressor

# The assistant turn's closing text, appended to the *next* forward pass because with a
# KV cache it is context, not output.
TURN_CLOSE = "<|im_end|>\n"
# What the chat template emits to open an assistant turn.
CHAT_ASSISTANT_OPEN = "<|im_start|>assistant\n"

# The canonical prompt, imported by `data_scripts/format_action_chunk_dataset.py` so the
# training conversations and the rollout context cannot drift apart.
DEFAULT_SYSTEM_PROMPT = (
    "You are a robot navigating an indoor environment toward a goal object.\n"
    "Goal: {goal}\n"
    "At each step you receive the current RGB observation. Produce the next short "
    "trajectory of poses to follow, relative to your current pose."
)


@dataclass
class RolloutConfig:
    """Everything the policy needs that is not stored in the checkpoint.

    Turn structure is described the way `VLMWorker` describes it -- by the affix strings --
    and everything else is derived, so no convention is hardcoded. `prefix`/`postfix`
    default to the checkpoint's own (`ModelConfig.prefix`/`postfix`), and `placeholder` is
    the constant assistant message; together with `shift_left` they determine which token
    positions the head reads, via the same `find_turn_spans` the training used.

    The strings must match the training data's conversation format. A mismatch changes the
    context the head was trained under, so `VectorRolloutPolicy` re-derives the span and
    asserts it against what the head expects rather than trusting them.
    """

    placeholder: str = "**____**"
    prefix: Optional[str] = None   # None -> the checkpoint's own prefix
    postfix: Optional[str] = None
    shift_left: Optional[bool] = None
    user_text_before: str = "Observation {step}:"
    user_text_after: str = "Action:"
    use_sparse: bool = True
    merge_lora: bool = True  # fold adapters into the base weights for inference speed
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
    # Optional guard: raise once the cached context passes this many tokens, instead of
    # discovering the limit as an OOM mid-episode. 0 disables.
    max_context_tokens: int = 0


def render_prologue(processor, system_prompt: str) -> str:
    return processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": system_prompt}]}],
        tokenize=False,
        add_generation_prompt=False,
    )


def render_user_block(processor, cfg: RolloutConfig, step: int) -> str:
    """One turn's user block, ending with the chat template's assistant opening."""
    user = {
        "role": "user",
        "content": [
            {"type": "text", "text": cfg.user_text_before.format(step=step)},
            {"type": "image"},
            {"type": "text", "text": cfg.user_text_after},
        ],
    }
    return processor.apply_chat_template([user], tokenize=False, add_generation_prompt=True)


def assistant_turn_text(cfg: RolloutConfig) -> str:
    """The complete assistant turn as it appears in training data."""
    return CHAT_ASSISTANT_OPEN + cfg.placeholder + TURN_CLOSE


def split_assistant_turn(tokenizer, cfg: RolloutConfig, prefix_ids, postfix_ids,
                         shift_left: bool):
    """Split the assistant turn at the readout boundary. Returns (emit, tail, n_readout).

    `emit` is everything up to and including the last position the head reads, so a forward
    pass ending there leaves those states as the final `n_readout` positions of the
    sequence -- which is why `step()` can take `last_hidden_state[:, -n:]` with no span
    search. `tail` is the remainder, owed to the next forward because with a KV cache it is
    context rather than output.

    The split is found with `find_turn_spans` -- the same function training used -- so any
    affix/placeholder combination is handled without special cases, and a combination that
    cannot be split (e.g. `'**[…]**'`, where BPE merges `]` into the closing `**` so the
    postfix never matches) fails here instead of silently training on nothing.
    """
    import torch as _torch

    from longnav.utils.turn_vectors import find_turn_spans

    turn = tokenizer.encode(assistant_turn_text(cfg), add_special_tokens=False)
    # Two copies: find_turn_spans needs a closed turn, and repeating it proves the split is
    # stable rather than an artifact of the sequence end.
    ids = _torch.tensor([turn * 2])
    spans = find_turn_spans(ids, prefix_ids, postfix_ids, shift_left=shift_left)[0]
    if len(spans) != 2:
        raise ValueError(
            f"placeholder {cfg.placeholder!r} with prefix {tokenizer.decode(prefix_ids)!r} / "
            f"postfix {tokenizer.decode(postfix_ids)!r} yields {len(spans)} readout span(s) "
            f"in two turns, expected 2. Its tokens are "
            f"{[tokenizer.decode([t]) for t in turn]} -- a placeholder whose characters "
            "merge into the affixes cannot be located"
        )
    start, end = spans[0].start, spans[0].end
    if end > len(turn):
        raise ValueError("readout span crosses the turn boundary; check the affixes")
    return turn[:end], turn[end:], end - start


def full_context_text(processor, cfg: RolloutConfig, n_turns: int,
                      system_prompt: Optional[str] = None) -> str:
    """The exact text an `n_turns` rollout builds. Must equal the training-time
    single-shot `apply_chat_template` of the same conversation -- asserted in
    `tests/test_vector_rollout.py`."""
    parts = [render_prologue(processor, system_prompt)] if system_prompt else []
    for i in range(n_turns):
        parts.append(render_user_block(processor, cfg, i))
        parts.append(cfg.placeholder + TURN_CLOSE)
    return "".join(parts)


class VectorRolloutPolicy:
    """Stateful, single-episode policy: `reset()` then `step()` per observation."""

    def __init__(self, model: TurnVectorRegressor, processor, cfg: Optional[RolloutConfig] = None):
        self.cfg = cfg or RolloutConfig()
        self.model = model
        self.processor = processor

        # Affixes come from the checkpoint unless overridden, exactly as VLMWorker takes
        # them from config: no convention is assumed anywhere below.
        from longnav.utils.turn_vectors import resolve_affix_ids

        mcfg = self.model.model_cfg
        self.prefix = self.cfg.prefix if self.cfg.prefix is not None else mcfg.prefix
        self.postfix = self.cfg.postfix if self.cfg.postfix is not None else mcfg.postfix
        self.shift_left = (
            self.cfg.shift_left if self.cfg.shift_left is not None
            else self.model.model_cfg.shift_left
        )
        self.prefix_ids, self.postfix_ids = resolve_affix_ids(
            processor.tokenizer, self.prefix, self.postfix
        )
        self.emit_ids, self.tail_ids, self.n_readout = split_assistant_turn(
            processor.tokenizer, self.cfg, self.prefix_ids, self.postfix_ids, self.shift_left
        )
        self._assert_head_matches_readout()
        # The processor may not be the one the model was built with. If its tokenizer is
        # missing a marker the literal BPEs back into ordinary tokens, the scatter finds
        # nothing, and the failure surfaces much later as a count mismatch.
        if self.model.modality_embedder:
            self.model.modality_embedder.bind_tokenizer(processor.tokenizer)

        self.model.to(self.cfg.device).eval()
        backbone = self.model.backbone
        if self.cfg.merge_lora and hasattr(backbone, "merge_adapter"):
            backbone.merge_adapter()
            self._merged = True
        else:
            self._merged = False

        # `get_rope_index` lives on Qwen3VLModel; unwrap peft to reach it.
        base = getattr(backbone, "base_model", backbone)
        self.vl_for_cond_gen = getattr(base, "model", base)
        self.vl_model = self.vl_for_cond_gen.model
        self.language_model = self.vl_model.language_model
        self.reset()

    def _assert_head_matches_readout(self):
        """The head pools a fixed number of positions; the affixes must produce that many.

        Cheap to check, and the failure it prevents is silent: a head trained on 3 content
        tokens fed 1 position (or vice versa) still returns a vector of the right shape.
        """
        head = self.model.head
        if head.mode == "flat" and head.content_len != self.n_readout:
            raise ValueError(
                f"head was trained with mode='flat' over {head.content_len} position(s) but "
                f"placeholder {self.cfg.placeholder!r} with these affixes yields "
                f"{self.n_readout}; the flat head's input width would not match"
            )
        if self.n_readout < 1:
            raise ValueError("affixes yield an empty readout span")
        # Any pooling mode still has to see the same *number* of positions it trained on,
        # or the pooled statistic changes meaning even though the shapes work out.
        trained = getattr(self.model, "train_content_len", None)
        if trained is not None and trained != self.n_readout:
            raise ValueError(
                f"head was trained pooling {trained} position(s) per turn, rollout would "
                f"pool {self.n_readout}"
            )

    def describe(self) -> str:
        tok = self.processor.tokenizer
        out = (
            f"prefix={self.prefix!r} postfix={self.postfix!r} "
            f"placeholder={self.cfg.placeholder!r} shift_left={self.shift_left}\n"
            f"  emitted per turn: {[tok.decode([t]) for t in self.emit_ids]}\n"
            f"  readout: last {self.n_readout} token(s) = "
            f"{[tok.decode([t]) for t in self.emit_ids[-self.n_readout:]]}\n"
            f"  owed to next turn: {[tok.decode([t]) for t in self.tail_ids]}"
        )
        if self.model.modality_embedder:
            out += ("\n  modality (pass step(..., modality={key: (N, F)})):\n"
                    + self.model.modality_embedder.describe())
        return out

    # ---------------------------------------------------------------------------------
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: Union[str, Path],
        cfg: Optional[RolloutConfig] = None,
        processor=None,
    ) -> "VectorRolloutPolicy":
        from transformers import AutoProcessor

        cfg = cfg or RolloutConfig()
        checkpoint_dir = Path(checkpoint_dir)
        if processor is None:
            # Trainer saves the processor next to the head; fall back to the base model id.
            try:
                processor = AutoProcessor.from_pretrained(str(checkpoint_dir))
            except Exception:
                import json

                from longnav.utils.vector_sft import HEAD_CONFIG_FILE

                meta = json.loads((checkpoint_dir / HEAD_CONFIG_FILE).read_text())
                processor = AutoProcessor.from_pretrained(meta["model"]["model_id"])
        model = TurnVectorRegressor.from_pretrained(
            checkpoint_dir, processor, dtype=cfg.dtype, device=cfg.device
        )
        return cls(model, processor, cfg)

    # ---------------------------------------------------------------------------------
    def reset(self, goal_text: Optional[str] = None, system_prompt: Optional[str] = None,
              modality: Optional[Dict[str, Any]] = None):
        """Start a fresh episode. The prologue is prepended to the first turn's tokens.

        `modality` carries values for any markers in the **prologue** -- an episode-level
        slot such as a goal location, a scene descriptor, an origin declaration. Training
        gets this for free (the window always keeps the prologue), so without it the
        failure is eval-only and confusing to debug.

        The prologue's tokens are owed to the first `step()`, so its occurrences come
        first in the sequence and its value rows are prepended to that step's.
        """
        self.past_key_values = None
        self.past_image_embeds = None
        self.rope_offset = 0
        self.step_index = 0
        self.cached_tokens = 0
        self.dense_tokens = 0
        if system_prompt is None and goal_text is not None:
            system_prompt = DEFAULT_SYSTEM_PROMPT.format(goal=goal_text)
        # Owed to the next forward pass, as token ids: the prologue on the first step, the
        # previous turn's tail afterwards.
        self._pending: List[int] = (
            self.processor.tokenizer.encode(
                render_prologue(self.processor, system_prompt), add_special_tokens=False
            )
            if system_prompt
            else []
        )
        self._pending_modality: Optional[ModalityBatch] = (
            single_example_batch(modality) if modality else None
        )
        self.last_stats: Dict[str, Any] = {}
        return self

    def _render_prologue(self, system_prompt: str) -> str:
        return render_prologue(self.processor, system_prompt)

    # ---------------------------------------------------------------------------------
    @torch.inference_mode()
    def step(self, image, user_text: Optional[str] = None,
             modality: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """Encode one observation and return its action chunk, shaped `target_shape`.

        Composition is done in *token ids*, not text: the user block comes from the
        processor (which expands the image placeholder), and the assistant pieces are the
        constant `tail`/`emit` id lists derived once in `__init__`. Concatenating ids
        removes any chance of BPE merging differently at a chunk boundary than it did in
        the training-time single-shot tokenization.

        `modality` is `{key: (N, F)}` for the markers in *this* chunk, in occurrence
        order. B == 1, the same `ModalityBatch`, the same hooks and the same assertions as
        training -- one scatter implementation for both paths, which is what makes a
        parity check meaningful rather than decorative. Where the values come from is the
        caller's problem.
        """
        block = self._render_user_block(image, user_text)
        ids = torch.tensor(
            [self._pending + block["input_ids"][0].tolist() + self.emit_ids],
            dtype=torch.long,
        )
        inputs = {
            k: v for k, v in block.items()
            # The processor's mask/ids describe the user block alone; ids below are the
            # concatenation, so a stale mask would be shorter than the sequence and
            # get_rope_index would index past it.
            if k not in ("input_ids", "attention_mask") and v is not None
        }
        inputs["input_ids"] = ids
        inputs["attention_mask"] = torch.ones_like(ids)

        # The prologue's occurrences precede this turn's in `ids`, so its rows precede
        # this turn's too -- occurrence order is the only binding.
        step_modality = single_example_batch(modality) if modality else None
        pending = self._pending_modality
        self._pending_modality = None
        if pending is not None:
            step_modality = pending.concat(step_modality or ModalityBatch())

        t0 = time.perf_counter()
        outputs = self._forward(inputs, step_modality)

        # The emitted block ends exactly at the last readout position, so the states the
        # head wants are the final `n_readout` of the sequence -- no span search needed, and
        # they are text tokens, which the sparsifier never drops.
        n = self.n_readout
        hidden = outputs["last_hidden_state"][:, -n:, :]
        head_dtype = next(self.model.head.parameters()).dtype
        vector = self.model.head(hidden.to(head_dtype))
        chunk = self.model.normalizer.denormalize(
            vector.view(-1, *self.model.target_shape)
        )[0].float().cpu()

        self._pending = list(self.tail_ids)
        self.step_index += 1
        self.last_stats = {
            "step": self.step_index,
            "new_tokens": int(ids.shape[1]),
            "readout_tokens": n,
            "sparse_tokens": int(outputs["last_hidden_state"].shape[1]),
            "cached_tokens": self.cached_tokens,
            "dense_tokens": self.dense_tokens,
            "latency_s": time.perf_counter() - t0,
        }
        return chunk

    def _render_user_block(self, image, user_text: Optional[str]):
        """Processor output for this turn's user block (image expanded), without the
        assistant opening -- that comes from `emit_ids`."""
        text = (
            user_text
            if user_text is not None
            else render_user_block(self.processor, self.cfg, self.step_index)
        )
        # The chat template already appends the assistant opening; drop it, since emit_ids
        # carries the opening plus whatever of the placeholder precedes the readout.
        if text.endswith(CHAT_ASSISTANT_OPEN):
            text = text[: -len(CHAT_ASSISTANT_OPEN)]
        return self.processor(
            text=text, images=[image], videos=None, padding=False, return_tensors="pt"
        )

    def _forward(self, inputs, modality: Optional[ModalityBatch] = None):
        device = self.cfg.device
        turn = {k: v.to(device) for k, v in inputs.items() if v is not None}
        n_new = turn["input_ids"].shape[1]

        # Rope positions for the new tokens only, continued from where the last turn
        # ended (VLMWorker._pos_id_fast's arithmetic: advance by length + rope delta).
        position_ids, deltas = self.vl_model.get_rope_index(
            input_ids=turn["input_ids"],
            image_grid_thw=turn.get("image_grid_thw"),
            video_grid_thw=None,
            attention_mask=turn.get("attention_mask"),
        )
        turn["position_ids"] = position_ids + self.rope_offset
        self.rope_offset += n_new + int(deltas.reshape(-1)[0].item())

        if self.cfg.use_sparse:
            # The sparse path derives its own mask; a dense mask would not match the
            # sparsified sequence length. The visual DB is what lets it drop frames that
            # are redundant with *earlier turns*, not just within this one.
            turn["attention_mask"] = None
            turn["past_image_embeds"] = self.past_image_embeds
            turn["save_image_db"] = True

        embedder = self.model.modality_embedder
        if embedder:
            embedder.check_placement(
                turn["input_ids"], self.vl_model.config.vision_start_token_id
            )
        with embedder.pending(modality.to(device) if modality is not None else None):
            outputs = self.model.backbone(
                **turn, past_key_values=self.past_key_values, use_cache=True,
                logits_to_keep=1,
            )
        self.past_key_values = outputs["past_key_values"]
        self.dense_tokens += n_new
        self.cached_tokens = int(
            self.past_key_values.get_seq_length() if self.past_key_values is not None else 0
        )
        if self.cfg.max_context_tokens and self.cached_tokens > self.cfg.max_context_tokens:
            raise RuntimeError(
                f"context reached {self.cached_tokens} cached tokens, over the configured "
                f"limit of {self.cfg.max_context_tokens}; shorten the episode or raise "
                "max_context_tokens"
            )

        if self.cfg.use_sparse:
            kept = getattr(self.language_model, "kept_visual_embeds", None)
            if kept:
                if self.past_image_embeds is None:
                    self.past_image_embeds = [k.clone() for k in kept]
                else:
                    self.past_image_embeds = [
                        torch.cat([old, new]) for old, new in zip(self.past_image_embeds, kept)
                    ]
        return outputs

    # ---------------------------------------------------------------------------------
    def full_context_text(self, n_turns: int, system_prompt: Optional[str] = None) -> str:
        return full_context_text(self.processor, self.cfg, n_turns, system_prompt)

    def unmerge(self):
        """Undo `merge_lora` (e.g. before saving or further training)."""
        if self._merged and hasattr(self.model.backbone, "unmerge_adapter"):
            self.model.backbone.unmerge_adapter()
            self._merged = False
