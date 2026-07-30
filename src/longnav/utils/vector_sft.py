"""
SFT with a continuous per-turn regression objective, for sparse Qwen3-VL.

Standard conversational SFT trains `lm_head` to predict the next token. This trains a
`TurnVectorHead` (see `longnav/utils/turn_vectors.py`) to predict one continuous target
*per assistant turn*, jointly with LoRA adapters on the backbone. The token-level LM loss
is not used at all: the assistant text is a fixed placeholder, and the thing being
learned lives in the head's output vector.

Grounding example (the shape this was built against, without being specific to it):
vision-language navigation where each turn's target is an action chunk of
`[N_action, 3]` relative poses `(dx, dy, dtheta)`. `data_scripts/build_*_dataset.py` in
the `continuous_demos` project emits exactly this -- per-observation image plus an
`action_chunks` column of shape `(n_obs, N_action, 3)` -- and its conversation builder
writes `**forward**` as the assistant message with the comment "actual action will be
predicted by a different head than LMHead". Nothing below knows any of that: the target
is an arbitrary fixed-shape tensor from a configurable column.

What the dataset must provide
-----------------------------
The standard HF conversational multimodal layout, one row per conversation:

  messages  list of {"role": ..., "content": [...]} with image placeholders, e.g.
            {"type": "image"} parts. Assistant turns are the ones that get a target.
  images    list of images (PIL, HF `Image` feature, or path strings), in the order the
            placeholders appear.
  <target>  list/array of per-assistant-turn targets, shape (n_turns, *target_shape).
            Column name is configurable (`DataConfig.target_column`).

Alignment is checked, not assumed: the number of assistant turns the tokenizer actually
produces must equal the number of targets, on every batch.

Why `shift_left` + action affixes is the default
------------------------------------------------
With `ACTION_PREFIX`/`ACTION_POSTFIX` and `shift_left=True`, the pooled span is the
single `**` token that opens every assistant turn. Two consequences, both wanted:

  * The assistant content is a constant placeholder, so pooling it adds nothing -- the
    vector is a pure function of the multimodal context up to that turn.
  * At inference the prompt ends `<|im_start|>assistant\\n**`, so that position exists
    *before* anything is generated. The head can be read turn-by-turn during rollout
    with no generation at all, which is what makes a continuous-action policy possible.

Set `shift_left=False` (and `affixes="template"`) for datasets whose assistant text is
real content that should inform the vector.

Hard constraint: batch size 1 per device
----------------------------------------
`modeling.py` unions the visual keep mask across the batch (`batch_keep_mask.any(dim=0)`)
and has open TODOs for B > 1; `turn_vectors.py` says the same. So
`per_device_train_batch_size` must be 1 and `TurnVectorSFTTrainer` refuses anything else.
Scale with `gradient_accumulation_steps` x DDP ranks instead. The upside: with B == 1
there is no padding, so no pad-aware masking is needed anywhere below.

Loss scaling
------------
Episodes have wildly different turn counts (the reference dataset has rows from a handful
of observations up to 223). Per-sample averaging would weight a 10-turn episode like a
200-turn one, so the loss is a *sum over turns* divided by the total number of turns in
the whole accumulated, all-ranks batch -- `_get_num_items_in_batch` counts turns the way
the stock Trainer counts unmasked label tokens, and HF's existing
`average_tokens_across_devices` plumbing then handles the DDP correction.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer

from longnav.utils.turn_vectors import (
    ACTION_POSTFIX,
    ACTION_PREFIX,
    DEFAULT_POSTFIX,
    DEFAULT_PREFIX,
    TurnVectorHead,
    extract_turn_vectors,
    resolve_affix_ids,
)

HEAD_WEIGHTS_FILE = "turn_vector_head.pt"
HEAD_CONFIG_FILE = "turn_vector_head_config.json"
ADAPTER_SUBDIR = "adapter"


# ======================================================================================
# Config
# ======================================================================================
@dataclass
class ModelConfig:
    """Backbone + head + span-convention settings.

    `affixes`/`shift_left` decide which token positions become the turn vector; see the
    module docstring for why the defaults are what they are.
    """

    model_id: str = "Qwen/Qwen3-VL-2B-Instruct"
    attn_impl: str = "sdpa"  # or flash_attention_2
    affixes: str = "action"  # "action" -> the '**' convention, "template" -> whole message
    shift_left: bool = True
    pool_mode: str = "mean"  # mean/last/attn/flat; irrelevant for 1-token spans
    head_hidden_dims: Tuple[int, ...] = (1024, 1024)
    head_dropout: float = 0.0
    head_layer_norm: bool = True
    standardize_head_inputs: bool = False
    freeze_vision_tower: bool = True


@dataclass
class DataConfig:
    """Column names and the turn-window policy."""

    target_column: str = "action_chunks"
    messages_column: str = "messages"
    images_column: str = "images"
    # Cap turns per sample. Episodes can be hundreds of observations long and every one
    # carries an image; without a cap the sequence blows past any token budget. Train
    # takes a random contiguous window, eval takes the first N, and messages/images/
    # targets are always sliced with one shared range.
    max_turns_per_sample: Optional[int] = 16
    # Names for the last target dimension, used only for metric keys.
    target_dim_names: Optional[Tuple[str, ...]] = None


@dataclass
class LoraSpec:
    r: int = 64
    alpha: int = 128
    dropout: float = 0.05
    target_modules: Tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )


@dataclass
class LossConfig:
    kind: str = "huber"  # huber | mse | l1
    huber_beta: float = 1.0
    normalize_targets: bool = True
    # Rows sampled to fit the target normalizer. None -> the whole train split.
    normalizer_fit_rows: Optional[int] = 512


# ======================================================================================
# Target normalization
# ======================================================================================
class TargetNormalizer(nn.Module):
    """Per-column standardization of the target's last dimension.

    Regression targets here are physically heterogeneous -- in the reference dataset a
    chunk's `dtheta` is O(1e-1) rad while early `dx`/`dy` are O(1e-11) m -- so an
    unnormalized L2/Huber objective is dominated by whichever column happens to be
    largest. Statistics live in buffers so they are saved and restored with the
    checkpoint and applied identically at eval and inference; metrics are always reported
    back in original units via `denormalize`.
    """

    def __init__(self, dim: int, enabled: bool = True):
        super().__init__()
        self.enabled = enabled
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))
        self.register_buffer("fitted", torch.zeros((), dtype=torch.bool))

    @torch.no_grad()
    def fit(self, targets: torch.Tensor):
        """`targets`: (..., dim) over as many turns as you can afford to look at."""
        flat = targets.reshape(-1, targets.shape[-1]).float()
        self.mean.copy_(flat.mean(0))
        self.std.copy_(flat.std(0).clamp(min=1e-6))
        self.fitted.fill_(True)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x
        if not bool(self.fitted):
            raise RuntimeError(
                "TargetNormalizer used before fit(); call fit_target_normalizer() on the "
                "train split (its stats are saved with the checkpoint)"
            )
        return (x - self.mean) / self.std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x
        return x * self.std + self.mean


def fit_target_normalizer(
    normalizer: TargetNormalizer,
    dataset,
    target_column: str,
    max_rows: Optional[int] = 512,
) -> int:
    """Fit `normalizer` from a dataset's target column. Returns the turn count used.

    Streams row by row so a large `IterableDataset` stays workable, and keeps only the
    running moments -- not the targets.
    """
    count = 0
    chunks: List[torch.Tensor] = []
    for i, row in enumerate(dataset):
        if max_rows is not None and i >= max_rows:
            break
        t = torch.as_tensor(row[target_column], dtype=torch.float32)
        chunks.append(t.reshape(-1, t.shape[-1]))
        count += t.reshape(-1, t.shape[-1]).shape[0]
    if not chunks:
        raise ValueError(f"no rows found to fit the target normalizer on {target_column!r}")
    total = torch.cat(chunks, dim=0)
    normalizer.fit(total)
    return count


# ======================================================================================
# Conversation windowing + collation
# ======================================================================================
def _content_parts(message: Dict[str, Any]) -> List[Dict[str, Any]]:
    content = message.get("content")
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return [p for p in content if isinstance(p, dict)]


def n_images_in_message(message: Dict[str, Any]) -> int:
    """Count image/video placeholders, however the row spells them."""
    return sum(1 for p in _content_parts(message) if p.get("type") in ("image", "video"))


def assistant_turn_indices(messages: Sequence[Dict[str, Any]]) -> List[int]:
    return [i for i, m in enumerate(messages) if m.get("role") == "assistant"]


def slice_conversation(
    messages: Sequence[Dict[str, Any]],
    images: Sequence[Any],
    targets: torch.Tensor,
    start_turn: int,
    n_turns: int,
) -> Tuple[List[Dict[str, Any]], List[Any], torch.Tensor]:
    """Take a contiguous window of `n_turns` assistant turns, consistently.

    Everything before the first image-bearing message is treated as a prologue (the
    system/instruction block) and always kept, so windowing away turn 0 does not throw
    away the task description. Turn `t`'s block is every message after turn `t-1` up to
    and including turn `t`, which keeps whatever user messages introduce it. Images are
    selected by counting placeholders in the kept messages, so the image list stays in
    lockstep with the text no matter how the placeholders are arranged.
    """
    turns = assistant_turn_indices(messages)
    if not turns:
        raise ValueError("conversation has no assistant turns")
    if n_turns >= len(turns) and start_turn == 0:
        return list(messages), list(images), targets

    end_turn = start_turn + n_turns
    prologue_end = next(
        (i for i, m in enumerate(messages) if n_images_in_message(m) > 0), 0
    )
    block_start = prologue_end if start_turn == 0 else turns[start_turn - 1] + 1
    keep_idx = list(range(prologue_end)) + list(range(block_start, turns[end_turn - 1] + 1))

    # Prefix sums over image placeholders -> which images belong to the kept messages.
    per_msg = [n_images_in_message(m) for m in messages]
    offsets = [0]
    for n in per_msg:
        offsets.append(offsets[-1] + n)
    if offsets[-1] != len(images):
        raise ValueError(
            f"conversation has {offsets[-1]} image placeholder(s) but {len(images)} "
            "image(s) were provided"
        )
    kept_images = [
        images[j] for i in keep_idx for j in range(offsets[i], offsets[i + 1])
    ]
    return [messages[i] for i in keep_idx], kept_images, targets[start_turn:end_turn]


def load_image(entry: Any):
    """Accept a PIL image, an HF `Image` dict, or a path string."""
    from PIL import Image as PILImage

    if isinstance(entry, str):
        return PILImage.open(entry).convert("RGB")
    if isinstance(entry, dict):
        if entry.get("bytes"):
            import io

            return PILImage.open(io.BytesIO(entry["bytes"])).convert("RGB")
        if entry.get("path"):
            return PILImage.open(entry["path"]).convert("RGB")
        raise ValueError(f"cannot decode image dict with keys {list(entry)}")
    return entry.convert("RGB") if hasattr(entry, "convert") else entry


@dataclass
class TurnVectorCollator:
    """One conversation -> model inputs + `(n_turns, *target_shape)` targets.

    B == 1 only (see the module docstring), which is asserted rather than papered over:
    the alternative is silently pooling the wrong positions.

    Note on `seed`: the rng is created once and then pickled into each dataloader worker,
    so every worker draws the same sequence of window offsets. Harmless (they see
    different rows) but it makes `seed` less of a global control than it looks with
    `dataloader_num_workers > 0`.
    """

    processor: Any
    data: DataConfig
    train: bool = True
    seed: int = 0
    _rng: Any = field(default=None, repr=False)

    def __post_init__(self):
        import numpy as np

        self._rng = np.random.default_rng(self.seed)

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        if len(examples) != 1:
            raise ValueError(
                f"per-device batch size must be 1 (got {len(examples)}): the sparse "
                "backbone unions its visual keep mask across the batch "
                "(modeling.py: batch_keep_mask.any(dim=0)). Use "
                "gradient_accumulation_steps / more ranks instead."
            )
        ex = examples[0]
        messages = ex[self.data.messages_column]
        images = ex[self.data.images_column]
        targets = torch.as_tensor(ex[self.data.target_column], dtype=torch.float32)
        if targets.dim() == 1:  # a scalar per turn
            targets = targets[:, None]

        turns = assistant_turn_indices(messages)
        if len(turns) != targets.shape[0]:
            raise ValueError(
                f"{len(turns)} assistant turn(s) in messages but "
                f"{targets.shape[0]} target(s) in column "
                f"{self.data.target_column!r} -- the dataset row is inconsistent"
            )

        cap = self.data.max_turns_per_sample
        if cap is not None and len(turns) > cap:
            start = int(self._rng.integers(0, len(turns) - cap + 1)) if self.train else 0
            messages, images, targets = slice_conversation(
                messages, images, targets, start, cap
            )

        pil_images = [load_image(im) for im in images]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        batch = self.processor(
            text=text,
            images=pil_images or None,
            videos=None,
            padding=False,
            return_tensors="pt",
        )
        out = dict(batch)
        out["targets"] = targets
        out["num_turns"] = torch.tensor(targets.shape[0], dtype=torch.long)
        return out


# ======================================================================================
# Model: LoRA backbone + turn-vector head, as one DDP-friendly module
# ======================================================================================
class TurnVectorRegressor(nn.Module):
    """Sparse VLM backbone (LoRA) + `TurnVectorHead`, with the regression loss inside.

    One `nn.Module` on purpose: DDP wraps a single module and needs the loss computed in
    `forward`, and HF `Trainer` needs `forward(**batch) -> {"loss": ...}`. Everything
    returned is a tensor, since DDP/accelerate move outputs across devices.
    """

    def __init__(
        self,
        backbone: nn.Module,
        head: TurnVectorHead,
        normalizer: TargetNormalizer,
        target_shape: Tuple[int, ...],
        prefix_ids: Sequence[int],
        postfix_ids: Sequence[int],
        model_cfg: ModelConfig,
        loss_cfg: LossConfig,
    ):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.normalizer = normalizer
        self.target_shape = tuple(target_shape)
        self.prefix_ids = list(prefix_ids)
        self.postfix_ids = list(postfix_ids)
        self.model_cfg = model_cfg
        self.loss_cfg = loss_cfg

    # -- construction ------------------------------------------------------------------
    @classmethod
    def build(
        cls,
        model_cfg: ModelConfig,
        loss_cfg: LossConfig,
        lora: Optional[LoraSpec],
        target_shape: Sequence[int],
        processor,
        dtype: torch.dtype = torch.bfloat16,
    ) -> "TurnVectorRegressor":
        from peft import LoraConfig, get_peft_model
        from transformers import AutoConfig

        from longnav.utils.modeling import Qwen3VLSparseForConditionalGeneration

        if model_cfg.attn_impl == "flash_attention_2":
            from longnav.utils.turn_vectors import patch_flash_attention_packing

            patch_flash_attention_packing()

        config = AutoConfig.from_pretrained(model_cfg.model_id, trust_remote_code=True)
        backbone = Qwen3VLSparseForConditionalGeneration.from_pretrained(
            model_cfg.model_id,
            config=config,
            dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            attn_implementation=model_cfg.attn_impl,
        )
        backbone.config.use_cache = False
        # Required for LoRA + gradient checkpointing: the checkpointed blocks need an
        # input that requires grad, and with a frozen embedding table there is none.
        backbone.enable_input_require_grads()
        if model_cfg.freeze_vision_tower:
            for p in backbone.model.visual.parameters():
                p.requires_grad_(False)

        if lora is None:
            # No adapters means head-only training, so the backbone must be frozen --
            # otherwise every one of its 2B parameters lands in the optimizer and AdamW's
            # states alone OOM a 24GB card before the first step.
            backbone.requires_grad_(False)
        else:
            backbone = get_peft_model(
                backbone,
                LoraConfig(
                    r=lora.r,
                    lora_alpha=lora.alpha,
                    lora_dropout=lora.dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                    target_modules=list(lora.target_modules),
                ),
            )

        target_shape = tuple(int(d) for d in target_shape)
        out_dim = int(math.prod(target_shape))
        hidden_size = config.text_config.hidden_size
        head = TurnVectorHead(
            hidden_size=hidden_size,
            out_dim=out_dim,
            mode=model_cfg.pool_mode,
            content_len=1 if model_cfg.pool_mode == "flat" else None,
            hidden_dims=tuple(model_cfg.head_hidden_dims),
            dropout=model_cfg.head_dropout,
            layer_norm=model_cfg.head_layer_norm,
            standardize=model_cfg.standardize_head_inputs,
        )
        prefix, postfix = affix_strings(model_cfg.affixes)
        prefix_ids, postfix_ids = resolve_affix_ids(processor.tokenizer, prefix, postfix)
        normalizer = TargetNormalizer(target_shape[-1], enabled=loss_cfg.normalize_targets)
        return cls(
            backbone, head, normalizer, target_shape, prefix_ids, postfix_ids,
            model_cfg, loss_cfg,
        )

    # -- plumbing HF Trainer expects ---------------------------------------------------
    def gradient_checkpointing_enable(self, **kwargs):
        self.backbone.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        self.backbone.gradient_checkpointing_disable()

    def enable_input_require_grads(self):
        base = getattr(self.backbone, "base_model", self.backbone)
        model = getattr(base, "model", base)
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    @property
    def device(self) -> torch.device:
        return next(self.backbone.parameters()).device

    def trainable_parameter_report(self) -> str:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        head = sum(p.numel() for p in self.head.parameters() if p.requires_grad)
        return (
            f"{trainable:,} trainable / {total:,} total params "
            f"({100 * trainable / total:.2f}%); head {head:,}, "
            f"adapters {trainable - head:,}"
        )

    # -- the objective -----------------------------------------------------------------
    def _elementwise_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        kind = self.loss_cfg.kind
        if kind == "huber":
            return F.smooth_l1_loss(pred, target, reduction="none", beta=self.loss_cfg.huber_beta)
        if kind == "mse":
            return F.mse_loss(pred, target, reduction="none")
        if kind == "l1":
            return F.l1_loss(pred, target, reduction="none")
        raise ValueError(f"unknown loss kind {kind!r}")

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        num_turns: Optional[torch.Tensor] = None,
        num_items_in_batch: Optional[Union[int, torch.Tensor]] = None,
        **multimodal,
    ) -> Dict[str, torch.Tensor]:
        multimodal.pop("labels", None)
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            logits_to_keep=1,  # never materialize (B, S, vocab); the LM head is unused
            **multimodal,
        )

        vectors, spans = extract_turn_vectors(
            outputs,
            input_ids,
            self.head,
            prefix_ids=self.prefix_ids,
            postfix_ids=self.postfix_ids,
            shift_left=self.model_cfg.shift_left,
            strict=True,
        )
        # The alignment gate. Turn k's vector must line up with target k; a tokenizer or
        # windowing change that shifts the count would otherwise train silently against
        # mismatched pairs.
        if vectors.shape[0] != targets.shape[0]:
            tail = input_ids[0, -64:].tolist()
            raise RuntimeError(
                f"found {vectors.shape[0]} assistant turn span(s) but got "
                f"{targets.shape[0]} target(s). affixes="
                f"{self.model_cfg.affixes!r} shift_left={self.model_cfg.shift_left}. "
                f"Last 64 input_ids: {tail}"
            )

        targets = targets.to(next(self.head.parameters()).dtype).to(vectors.device)
        pred = vectors.view(-1, *self.target_shape)
        tgt_norm = self.normalizer.normalize(targets)

        per_turn = self._elementwise_loss(pred, tgt_norm).flatten(1).mean(dim=1)
        total = per_turn.sum()
        denom = num_items_in_batch if num_items_in_batch is not None else per_turn.numel()
        denom = torch.as_tensor(denom, dtype=total.dtype, device=total.device).clamp(min=1)
        loss = total / denom

        with torch.no_grad():
            pred_orig = self.normalizer.denormalize(pred.detach())
            err = (pred_orig - targets).reshape(-1, self.target_shape[-1])
            metrics = {
                "loss_sum": total.detach(),
                "sum_sq_err": err.pow(2).sum(0),
                "sum_abs_err": err.abs().sum(0),
                "n_rows": torch.tensor(err.shape[0], device=err.device),
                "n_turns": torch.tensor(pred.shape[0], device=err.device),
                "n_tokens": torch.tensor(
                    outputs["last_hidden_state"].shape[1], device=err.device
                ),
                "n_dense_tokens": torch.tensor(input_ids.shape[1], device=err.device),
                "n_steps": torch.tensor(1, device=err.device),
            }
        return {"loss": loss, **metrics}

    # -- checkpointing -----------------------------------------------------------------
    def save_pretrained(self, output_dir: Union[str, Path]):
        """Save only what trains: the adapter, the head, and the normalizer buffers.

        Trainer's default would pickle the whole frozen backbone into every checkpoint.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(self.backbone, "save_pretrained") and hasattr(self.backbone, "peft_config"):
            self.backbone.save_pretrained(str(output_dir / ADAPTER_SUBDIR))
        torch.save(
            {
                "head": self.head.state_dict(),
                "normalizer": self.normalizer.state_dict(),
            },
            output_dir / HEAD_WEIGHTS_FILE,
        )
        (output_dir / HEAD_CONFIG_FILE).write_text(
            json.dumps(
                {
                    "model": asdict(self.model_cfg),
                    "loss": asdict(self.loss_cfg),
                    "target_shape": list(self.target_shape),
                    "prefix_ids": self.prefix_ids,
                    "postfix_ids": self.postfix_ids,
                },
                indent=2,
            )
        )

    def load_head_state(self, checkpoint_dir: Union[str, Path], strict: bool = True):
        """Load the head weights and the normalizer buffers."""
        blob = torch.load(
            Path(checkpoint_dir) / HEAD_WEIGHTS_FILE, map_location="cpu", weights_only=False
        )
        self.head.load_state_dict(blob["head"], strict=strict)
        self.normalizer.load_state_dict(blob["normalizer"], strict=strict)
        return self

    def load_adapter_state(self, checkpoint_dir: Union[str, Path]) -> bool:
        """Load LoRA weights into the adapter this model already has. False if none saved.

        Deliberately `set_peft_model_state_dict` rather than `backbone.load_adapter(...,
        adapter_name="default")`: `build()` has already created an adapter under that name,
        and loading into an existing name is version-dependent (overwrite vs. warn vs.
        no-op). Writing the state dict into the live adapter is unambiguous, and a missing
        or unexpected key raises instead of leaving half the weights at init.
        """
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        adapter_dir = Path(checkpoint_dir) / ADAPTER_SUBDIR
        if not adapter_dir.exists():
            return False  # head-only checkpoint (trained with lora=None)
        if not hasattr(self.backbone, "peft_config"):
            raise RuntimeError(
                f"{adapter_dir} holds LoRA weights but this model was built without "
                "adapters; rebuild with a LoraSpec (or use from_pretrained) or the "
                "backbone would silently stay at its pretrained values"
            )
        state = load_file(adapter_dir / "adapter_model.safetensors")
        result = set_peft_model_state_dict(self.backbone, state)
        unexpected = getattr(result, "unexpected_keys", [])
        if unexpected:
            raise RuntimeError(f"adapter checkpoint has unexpected keys: {unexpected[:8]}")
        return True

    def load_trainable(self, checkpoint_dir: Union[str, Path], strict: bool = True,
                       adapter: bool = True):
        """Inverse of `save_pretrained` for an already-built model (resume / eval)."""
        self.load_head_state(checkpoint_dir, strict=strict)
        if adapter:
            self.load_adapter_state(checkpoint_dir)
        return self

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_dir: Union[str, Path],
        processor,
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[str] = None,
        **overrides,
    ) -> "TurnVectorRegressor":
        """Rebuild a trained model for inference from a checkpoint directory."""
        checkpoint_dir = Path(checkpoint_dir)
        meta = json.loads((checkpoint_dir / HEAD_CONFIG_FILE).read_text())
        model_cfg = ModelConfig(**{**meta["model"], **overrides})
        loss_cfg = LossConfig(**meta["loss"])
        model = cls.build(
            model_cfg, loss_cfg, lora=None,
            target_shape=meta["target_shape"], processor=processor, dtype=dtype,
        )
        adapter_dir = checkpoint_dir / ADAPTER_SUBDIR
        if adapter_dir.exists():
            from peft import PeftModel

            # This already loads the adapter weights, so `load_trainable` must not do it
            # again -- hence adapter=False below.
            model.backbone = PeftModel.from_pretrained(model.backbone, str(adapter_dir))
        model.load_trainable(checkpoint_dir, adapter=False)
        if device:
            model.to(device)
        return model.eval()


def affix_strings(kind: str) -> Tuple[str, str]:
    if kind == "action":
        return ACTION_PREFIX, ACTION_POSTFIX
    if kind == "template":
        return DEFAULT_PREFIX, DEFAULT_POSTFIX
    raise ValueError(f"affixes must be 'action' or 'template', got {kind!r}")


# ======================================================================================
# Trainer
# ======================================================================================
class TurnVectorSFTTrainer(Trainer):
    """`Trainer` wired for a per-turn regression objective.

    Overrides, and why each is needed:

      `_get_num_items_in_batch`  count turns instead of unmasked label tokens, so the
          loss is a true per-turn mean across gradient accumulation and ranks.
      `compute_loss`             delegate to the parent (keeping HF's accumulation/DDP
          loss scaling exactly) and siphon off the metric tensors the model returns.
      `evaluate`                 a manual loop. Turn counts vary per row, so the stock
          evaluation loop's padded `nested_concat` gathering does not apply.
      `_save` / `_load_from_checkpoint`  save/restore adapter + head only.
    """

    def __init__(self, *args, data_config: Optional[DataConfig] = None,
                 eval_data_collator: Optional[Any] = None, **kwargs):
        super().__init__(*args, **kwargs)
        # Eval must not use the train collator: that one samples a *random* turn window
        # per row, so eval numbers would move for reasons unrelated to the model.
        self.eval_data_collator = eval_data_collator
        if self.args.per_device_train_batch_size != 1:
            raise ValueError(
                "per_device_train_batch_size must be 1: the sparse backbone unions its "
                "visual keep mask across the batch (modeling.py) and turn extraction "
                "assumes B == 1. Use gradient_accumulation_steps or more ranks."
            )
        self.data_config = data_config or DataConfig()
        # Lets the parent pass `num_items_in_batch` into forward and skip its own
        # divide-by-accumulation-steps, which our normalization already accounts for.
        self.model_accepts_loss_kwargs = True
        self._sums: Dict[str, torch.Tensor] = {}

    # -- turn counting -----------------------------------------------------------------
    def _get_num_items_in_batch(self, batch_samples: List[Dict[str, Any]], device):
        if not batch_samples or "num_turns" not in batch_samples[0]:
            return None
        n = sum(int(b["num_turns"]) for b in batch_samples)
        n = torch.tensor(n, device=device)
        if self.args.average_tokens_across_devices and self.args.world_size > 1:
            n = self.accelerator.gather(n).sum()
        return n

    # -- metric bookkeeping ------------------------------------------------------------
    def _accumulate(self, outputs: Dict[str, torch.Tensor]):
        for key in ("loss_sum", "sum_sq_err", "sum_abs_err", "n_rows", "n_turns",
                    "n_tokens", "n_dense_tokens", "n_steps"):
            if key not in outputs:
                continue
            v = outputs[key].detach().float()
            self._sums[key] = v.clone() if key not in self._sums else self._sums[key] + v

    def _dim_names(self, n: int) -> List[str]:
        names = self.data_config.target_dim_names
        return list(names) if names and len(names) == n else [str(i) for i in range(n)]

    def _drain_metrics(self, prefix: str = "") -> Dict[str, float]:
        if not self._sums or "n_rows" not in self._sums:
            self._sums = {}
            return {}
        sums = self._sums
        self._sums = {}
        rows = sums["n_rows"].clamp(min=1)
        turns = sums["n_turns"].clamp(min=1)
        steps = sums.get("n_steps", torch.tensor(1.0)).clamp(min=1)
        out = {
            f"{prefix}turns": float(sums["n_turns"]),
            f"{prefix}turn_loss": float(sums["loss_sum"] / turns),
            f"{prefix}turns_per_sample": float(sums["n_turns"] / steps),
        }
        if "n_tokens" in sums:
            out[f"{prefix}sparse_tokens"] = float(sums["n_tokens"] / steps)
            out[f"{prefix}dense_tokens"] = float(sums["n_dense_tokens"] / steps)
        rmse = (sums["sum_sq_err"] / rows).sqrt()
        mae = sums["sum_abs_err"] / rows
        for name, r, m in zip(self._dim_names(rmse.numel()), rmse.tolist(), mae.tolist()):
            out[f"{prefix}rmse_{name}"] = r
            out[f"{prefix}mae_{name}"] = m
        out[f"{prefix}rmse_mean"] = float(rmse.mean())
        return out

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss, outputs = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )
        self._accumulate(outputs)
        return (loss, outputs) if return_outputs else loss

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None):
        if self.model.training:
            logs = {**logs, **self._drain_metrics()}
        super().log(logs, start_time)

    # -- evaluation --------------------------------------------------------------------
    def get_eval_dataloader(self, eval_dataset=None):
        if self.eval_data_collator is None:
            return super().get_eval_dataloader(eval_dataset)
        train_collator, self.data_collator = self.data_collator, self.eval_data_collator
        try:
            return super().get_eval_dataloader(eval_dataset)
        finally:
            self.data_collator = train_collator

    @torch.no_grad()
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Manual eval loop with all-reduced sums, reported in original target units."""
        dataloader = self.get_eval_dataloader(eval_dataset)
        model = self._wrap_model(self.model, training=False)
        was_training = model.training
        model.eval()
        self._sums = {}
        for inputs in dataloader:
            inputs = self._prepare_inputs(inputs)
            inputs.pop("num_items_in_batch", None)
            outputs = model(**inputs)
            self._accumulate(outputs)

        # Sum every accumulator across ranks before turning them into rates.
        if self.args.world_size > 1:
            self._sums = {
                k: self.accelerator.reduce(v.to(self.args.device), reduction="sum")
                for k, v in self._sums.items()
            }
        metrics = self._drain_metrics(prefix=f"{metric_key_prefix}_")
        # `eval_loss` is the key `load_best_model_at_end` / early stopping look for.
        if f"{metric_key_prefix}_turn_loss" in metrics:
            metrics[f"{metric_key_prefix}_loss"] = metrics[f"{metric_key_prefix}_turn_loss"]
        # Explicit print: the callback that echoes `log()` to stdout does not surface
        # these when a tqdm bar owns the terminal, and a silent eval is worse than noise.
        if self.args.local_rank in (-1, 0):
            print("  " + "  ".join(
                f"{k}={v:.4f}" for k, v in sorted(metrics.items()) if isinstance(v, float)
            ), flush=True)
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics
        )
        model.train(was_training)
        return metrics

    # -- checkpoints -------------------------------------------------------------------
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        unwrapped = self.accelerator.unwrap_model(self.model)
        unwrapped.save_pretrained(output_dir)
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        model = model or self.model
        self.accelerator.unwrap_model(model).load_trainable(resume_from_checkpoint)
