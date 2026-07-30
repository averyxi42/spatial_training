"""
Minimal turn-by-turn rollout for a model trained by `vector_sft.py`.

`VLMWorker` (see `vlm_worker.py`) does incremental multi-turn inference, but it is built
for the RL stack: Ray actors, logprobs, value heads, adapter merge/unmerge bookkeeping,
rollout buffers. This is the same KV-cache mechanics with none of that -- feed one
observation, get one continuous action chunk, repeat. Nothing here trains, distributes or
logs. `vlm_worker.py` is untouched.

The whole thing rests on one property of the training convention (see `vector_sft.py`):
the head pools the single `**` token that opens each assistant turn. At rollout we simply
stop the prompt at that token and read `last_hidden_state[:, -1]`, so no span search and
no generation is needed -- the action for step k is available the moment step k's image is
encoded. A model trained to pool assistant *content* instead cannot be rolled out this
way (the content does not exist yet), and `VectorRolloutPolicy` refuses to load one
rather than silently pooling a different position than it trained on.

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

from longnav.utils.vector_sft import TurnVectorRegressor

# The assistant turn's closing text, appended to the *next* forward pass because with a
# KV cache it is context, not output.
TURN_CLOSE = "<|im_end|>\n"

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

    `placeholder` and the user-turn strings must match the training data's conversation
    format; a mismatch changes the context the head was trained under.
    """

    placeholder: str = "**forward**"
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


def render_turn(processor, cfg: RolloutConfig, step: int) -> str:
    """One turn's user block plus the assistant opening, ending at the `**` token."""
    user = {
        "role": "user",
        "content": [
            {"type": "text", "text": cfg.user_text_before.format(step=step)},
            {"type": "image"},
            {"type": "text", "text": cfg.user_text_after},
        ],
    }
    head = processor.apply_chat_template([user], tokenize=False, add_generation_prompt=True)
    return head + "**"


def assistant_tail(cfg: RolloutConfig) -> str:
    """What is still owed once a turn's action has been read out of the `**` position."""
    return cfg.placeholder[len("**"):] + TURN_CLOSE


def full_context_text(processor, cfg: RolloutConfig, n_turns: int,
                      system_prompt: Optional[str] = None) -> str:
    """The exact text an `n_turns` rollout builds. Must equal the training-time
    single-shot `apply_chat_template` of the same conversation -- asserted in
    `tests/test_vector_rollout.py`."""
    parts = [render_prologue(processor, system_prompt)] if system_prompt else []
    for i in range(n_turns):
        parts.append(render_turn(processor, cfg, i))
        parts.append(assistant_tail(cfg))
    return "".join(parts)


class VectorRolloutPolicy:
    """Stateful, single-episode policy: `reset()` then `step()` per observation."""

    def __init__(self, model: TurnVectorRegressor, processor, cfg: Optional[RolloutConfig] = None):
        self.cfg = cfg or RolloutConfig()
        self.model = model
        self.processor = processor

        if not (self.model.model_cfg.affixes == "action" and self.model.model_cfg.shift_left):
            raise ValueError(
                "step-by-step rollout requires a model trained with affixes='action' and "
                f"shift_left=True (got affixes={self.model.model_cfg.affixes!r}, "
                f"shift_left={self.model.model_cfg.shift_left}). Any other convention "
                "pools assistant content, which does not exist until it is generated."
            )
        if not self.cfg.placeholder.startswith("**"):
            raise ValueError(
                f"placeholder {self.cfg.placeholder!r} must start with '**' to match "
                "ACTION_PREFIX, which is the token the head pools"
            )

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
    def reset(self, goal_text: Optional[str] = None, system_prompt: Optional[str] = None):
        """Start a fresh episode. The prologue is prepended to the first turn's tokens."""
        self.past_key_values = None
        self.past_image_embeds = None
        self.rope_offset = 0
        self.step_index = 0
        self.cached_tokens = 0
        self.dense_tokens = 0
        self._pending = ""  # previous assistant turn's tail, owed to the next forward
        if system_prompt is None and goal_text is not None:
            system_prompt = DEFAULT_SYSTEM_PROMPT.format(goal=goal_text)
        self._pending = self._render_prologue(system_prompt) if system_prompt else ""
        self.last_stats: Dict[str, Any] = {}
        return self

    def _render_prologue(self, system_prompt: str) -> str:
        return render_prologue(self.processor, system_prompt)

    def _render_turn(self, step: int) -> str:
        return render_turn(self.processor, self.cfg, step)

    def assistant_tail(self) -> str:
        return assistant_tail(self.cfg)

    # ---------------------------------------------------------------------------------
    @torch.inference_mode()
    def step(self, image, user_text: Optional[str] = None) -> torch.Tensor:
        """Encode one observation and return its action chunk, shaped `target_shape`."""
        text = self._pending + (
            user_text + "**" if user_text is not None else self._render_turn(self.step_index)
        )
        inputs = self.processor(
            text=text, images=[image], videos=None, padding=False, return_tensors="pt"
        )
        t0 = time.perf_counter()
        outputs = self._forward(inputs)

        # The prompt ends at the `**` the head was trained to pool, so the position is
        # simply the last one -- and it is a text token, hence never sparsified away.
        hidden = outputs["last_hidden_state"][:, -1:, :]
        vector = self.model.head(hidden.to(next(self.model.head.parameters()).dtype))
        chunk = self.model.normalizer.denormalize(
            vector.view(-1, *self.model.target_shape)
        )[0].float().cpu()

        self._pending = self.assistant_tail()
        self.step_index += 1
        self.last_stats = {
            "step": self.step_index,
            "new_tokens": int(inputs["input_ids"].shape[1]),
            "sparse_tokens": int(outputs["last_hidden_state"].shape[1]),
            "cached_tokens": self.cached_tokens,
            "dense_tokens": self.dense_tokens,
            "latency_s": time.perf_counter() - t0,
        }
        return chunk

    def _forward(self, inputs):
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

        outputs = self.model.backbone(
            **turn, past_key_values=self.past_key_values, use_cache=True, logits_to_keep=1
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
