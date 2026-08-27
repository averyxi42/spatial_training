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
`action_chunks` column of shape `(n_obs, N_action, 3)`, and the conversation builder writes
a constant placeholder as the assistant message because the action comes from this head
rather than from `lm_head`. Nothing below knows any of that: the target is an arbitrary
fixed-shape tensor from a configurable column.

The placeholder is `**____**` (see `docs/placeholder_tokens.md`). It is not `**forward**`
because that asserts a false action history -- the placeholder stays in the KV cache for
every later turn, so a 200-observation episode would claim the robot went forward 200
times. `____` is a single token (id 2130) that reads as an elision instead.

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

Why the default affixes wrap the content in `**`, with `shift_left`
------------------------------------------------------------------
A turn is `prefix + content + postfix`; the defaults are `ACTION_PREFIX` /
`ACTION_POSTFIX` (`"<|im_start|>assistant\n**"` ... `"**<|im_end|>"`), which with
`shift_left=True` make the pooled span the single `**` token that opens every assistant
turn. Two consequences, both wanted:

  * The assistant content is a constant placeholder, so pooling it adds nothing -- the
    vector is a pure function of the multimodal context up to that turn. The `**` opener
    also carries a "a word follows" prior from pretraining, which measurably keeps the
    readout more sensitive to the image than a bare placeholder token does.
  * At inference the prompt ends `<|im_start|>assistant\\n**`, so that position exists
    *before* anything is generated. The head can be read turn-by-turn during rollout
    with no generation at all, which is what makes a continuous-action policy possible.

Set `shift_left=False` (with `DEFAULT_PREFIX`/`DEFAULT_POSTFIX`, the bare chat template)
for datasets whose assistant text is real content that should inform the vector -- or any
other affix pair; nothing downstream is specific to these two.

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

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer

from longnav.utils.modality_embed import (
    ModalityBatch,
    ModalityEmbedder,
    ModalityEmbedSpec,
    coerce_specs,
    resolve_token_ids,
)
from longnav.utils.stop_head import (
    StopHead,
    StopHeadConfig,
    episode_stop_labels,
    stop_metrics,
)
from longnav.utils.turn_vectors import (
    ACTION_POSTFIX,
    ACTION_PREFIX,
    DEFAULT_POSTFIX,
    DEFAULT_PREFIX,
    TurnVectorHead,
    extract_turn_vectors,
    resolve_affix_ids,
)

def _shim_peft_tp_import() -> None:
    """Let ``set_peft_model_state_dict`` import under transformers 4.57.

    peft 0.19 calls ``_maybe_shard_state_dict_for_tp`` whenever
    ``torch.distributed.is_initialized()`` -- true for ordinary DDP, not just
    tensor parallelism -- and that function begins by importing
    ``EmbeddingParallel`` from ``transformers.integrations.tensor_parallel``.
    transformers 4.57.6 does not export that name, so **resuming from a
    checkpoint dies on an ImportError while fresh training never touches the
    path**. That asymmetry is why it survived until the first resume, on
    2026-08-07, and cost two runs a restart.

    The function is provably inert here: it skips every layer whose base layer
    lacks ``_hf_tp_plan``/``_hf_device_mesh``, and nothing in this codebase sets
    those. The missing symbol therefore only has to exist, never work.
    Injecting a placeholder is narrower than pinning either package, and it
    stops mattering the day transformers reintroduces the name.
    """
    try:
        import transformers.integrations.tensor_parallel as _tp
    except Exception:  # no TP module at all; peft's own import would fail too
        return
    for _name in ("EmbeddingParallel",):
        if not hasattr(_tp, _name):
            setattr(_tp, _name, type(_name, (), {}))


HEAD_WEIGHTS_FILE = "turn_vector_head.pt"
HEAD_CONFIG_FILE = "turn_vector_head_config.json"
ADAPTER_SUBDIR = "adapter"


# ======================================================================================
# Config
# ======================================================================================
@dataclass
class ModelConfig:
    """Backbone + head + span-convention settings.

    A turn is `prefix + content + postfix`, and the affixes are exactly those two strings
    -- there is no enum of blessed conventions. `prefix`/`postfix` plus `shift_left` decide
    which token positions become the turn vector: with `shift_left` the readout is the last
    token of `prefix` (so it is content-independent), otherwise it is the content itself.
    `longnav.utils.turn_vectors` exports the two pairs used so far
    (`ACTION_PREFIX`/`ACTION_POSTFIX`, which wrap the content in `**`, and
    `DEFAULT_PREFIX`/`DEFAULT_POSTFIX`, the bare chat template) but any pair the tokenizer
    can locate works. See the module docstring for why the defaults are what they are.
    """

    model_id: str = "Qwen/Qwen3-VL-2B-Instruct"
    attn_impl: str = "sdpa"  # or flash_attention_2
    prefix: str = ACTION_PREFIX
    postfix: str = ACTION_POSTFIX
    shift_left: bool = True
    pool_mode: str = "mean"  # mean/last/attn/flat; irrelevant for 1-token spans
    head_hidden_dims: Tuple[int, ...] = (1024, 1024)
    head_dropout: float = 0.0
    head_layer_norm: bool = True
    standardize_head_inputs: bool = False
    freeze_vision_tower: bool = True
    # Learned per-occurrence embeddings injected at marker tokens; see
    # `longnav.utils.modality_embed`. Empty (the default, and what every checkpoint
    # written before this existed says) leaves the whole mechanism inert.
    modality_specs: Tuple[ModalityEmbedSpec, ...] = ()
    # Auxiliary binary "is this the episode end" readout on the pooled turn context; see
    # `longnav.utils.stop_head`. None (the default, and what every checkpoint written
    # before this existed says) means no head, no loss term and no metrics.
    stop_head: Optional[StopHeadConfig] = None

    def __post_init__(self):
        # Round-tripping through JSON turns the specs into plain dicts; coerce them back
        # here so `ModelConfig(**meta["model"])` yields the same object either way.
        self.modality_specs = coerce_specs(self.modality_specs)
        if self.stop_head is not None:
            self.stop_head = StopHeadConfig.from_dict(self.stop_head)


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


def n_markers_in_message(message: Dict[str, Any], token: str) -> int:
    """Count occurrences of a modality marker literal in a message's text parts.

    The modality analogue of `n_images_in_message`, and it has to be: windowing selects
    images by counting placeholders in the kept messages, and the modality column is bound
    by the same occurrence order, so it must be selected by counting the same way.
    """
    return sum(
        p.get("text", "").count(token)
        for p in _content_parts(message)
        if p.get("type") == "text"
    )


def conversation_window_indices(
    messages: Sequence[Dict[str, Any]], start_turn: int, n_turns: int
) -> List[int]:
    """Message indices kept by a window of `n_turns` assistant turns from `start_turn`.

    Everything before the first image-bearing message is the prologue (the
    system/instruction block) and is always kept, so windowing away turn 0 does not throw
    away the task description. Turn `t`'s block is every message after turn `t-1` up to
    and including turn `t`, which keeps whatever user messages introduce it.
    """
    turns = assistant_turn_indices(messages)
    if not turns:
        raise ValueError("conversation has no assistant turns")
    end_turn = start_turn + n_turns
    prologue_end = next(
        (i for i, m in enumerate(messages) if n_images_in_message(m) > 0), 0
    )
    block_start = prologue_end if start_turn == 0 else turns[start_turn - 1] + 1
    return list(range(prologue_end)) + list(range(block_start, turns[end_turn - 1] + 1))


def select_by_occurrence(
    items: Sequence[Any],
    per_message: Sequence[int],
    keep_idx: Sequence[int],
    noun: str = "image",
) -> List[Any]:
    """Take the entries of `items` belonging to the kept messages.

    `items` is a flat list in occurrence order and `per_message[i]` is how many of them
    message `i` accounts for; prefix sums map one to the other. This is the single idiom
    that keeps images and modality values in lockstep with the text, no matter how the
    placeholders are arranged within a message.
    """
    offsets = [0]
    for n in per_message:
        offsets.append(offsets[-1] + n)
    if offsets[-1] != len(items):
        raise ValueError(
            f"conversation has {offsets[-1]} {noun} placeholder(s) but {len(items)} "
            f"{noun}(s) were provided"
        )
    return [items[j] for i in keep_idx for j in range(offsets[i], offsets[i + 1])]


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

    keep_idx = conversation_window_indices(messages, start_turn, n_turns)
    kept_images = select_by_occurrence(
        images, [n_images_in_message(m) for m in messages], keep_idx, noun="image"
    )
    return [messages[i] for i in keep_idx], kept_images, targets[start_turn:start_turn + n_turns]


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

    Modality columns
    ----------------
    `modality_specs` is the model's own spec list, so which column feeds which marker has
    exactly one definition. This is the right place for the count check: it is the single
    seam that templates, processes and validates, and it counts **marker ids in
    `input_ids`** -- which catches a chat template mangling the literal, something a
    message-level check on the raw text would miss. Per example, before batching, so a
    mismatch is attributable to one conversation.
    """

    processor: Any
    data: DataConfig
    train: bool = True
    seed: int = 0
    modality_specs: Tuple[ModalityEmbedSpec, ...] = ()
    # Emit `stop_targets` -- the episode-end indicator, sliced with the window. Off unless
    # the model declares a stop head, so nothing changes for a run that has none.
    stop_labels: bool = False
    _rng: Any = field(default=None, repr=False)
    _token_ids: Any = field(default=None, repr=False)

    def __post_init__(self):
        import numpy as np

        self._rng = np.random.default_rng(self.seed)
        self.modality_specs = coerce_specs(self.modality_specs)

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

        # Raw column width, which is not necessarily `n_features`: a spec may declare a
        # transform that changes it. Without a transform `raw_width == n_features` and this
        # is the reshape it always was.
        # THE CONTRACT: no marker occurrences means no values are required. A source may
        # express "this row has no pose" as an absent column, a `None`, or an empty list,
        # and all three mean the same thing -- the marker is a modality like `<image>`, and
        # a conversation that never writes it simply supplies nothing for it. That is what
        # lets a mixture pair a pose-carrying corpus with a pose-free one.
        #
        # Enforcement is not weakened by any of this, because it lives one step later:
        # `_modality_kwargs` compares the row count against the marker count actually found
        # in `input_ids`. Zero rows against zero markers passes; zero rows against N markers
        # still fails, and so does N rows against zero. `ModalityEmbedder` agrees -- "a key
        # with zero rows and zero occurrences is fine".
        #
        # Note `check_compatible` cannot catch a mismatch here: it requires only the three
        # columns the collator reads, and the pose column is named by the model's spec.
        modality = {}
        for s in self.modality_specs:
            raw = ex.get(s.source_column)
            if raw is None:
                raw = []
            modality[s.key] = torch.as_tensor(raw, dtype=torch.float32).reshape(
                -1, s.raw_width
            )

        n_total_turns = len(turns)
        start = 0
        cap = self.data.max_turns_per_sample
        if cap is not None and len(turns) > cap:
            start = int(self._rng.integers(0, len(turns) - cap + 1)) if self.train else 0
            # The modality column must slice with the window, through the same
            # count-the-placeholders idiom images use. It is bound by occurrence order,
            # so counting marker literals in the kept messages is the only correct rule --
            # anything keyed on turn index would break the moment a message carries two
            # markers or none.
            keep_idx = conversation_window_indices(messages, start, cap)
            for s in self.modality_specs:
                rows = select_by_occurrence(
                    list(range(modality[s.key].shape[0])),
                    [n_markers_in_message(m, s.token) for m in messages],
                    keep_idx,
                    noun=f"{s.token} marker",
                )
                modality[s.key] = modality[s.key][torch.tensor(rows, dtype=torch.long)]
            messages, images, targets = slice_conversation(
                messages, images, targets, start, cap
            )

        # After the window, never before. A transform sees one example's values as the
        # model will see them, so for pose "relative to the first row" means the first row
        # *in context* -- which is the window's first observation when windowing is on and
        # the episode's first when it is off. Transforming before the slice would anchor to
        # an origin the agent never observed, and the same number would then mean different
        # things in training and in a rollout that always starts at its own first frame.
        for s in self.modality_specs:
            modality[s.key] = s.apply_transform(modality[s.key])

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
        if self.stop_labels:
            # MOTION-stop labelling, for `stop_head.py`'s head only. Derived from the
            # row's own turn count rather than read from a column: for that head the
            # episode end IS the definition of a stop, so a second representation in the
            # dataset would only create something to disagree with.
            #
            # The EPISODE-stop head does not use this. `ProbeCollator` overwrites
            # `stop_targets` from the dataset's own per-frame column, which is the metric
            # `distance_to_goal <= success_radius` -- see the comment there and
            # `data_scripts/add_stop_targets.py`. If you are reading this while wondering
            # how stops are labelled today, the answer is the column, not this line.
            out["stop_targets"] = episode_stop_labels(
                targets.shape[0], start, n_total_turns
            )
        out.update(self._modality_kwargs(modality, out["input_ids"]))
        return out

    def _modality_token_ids(self) -> Dict[str, int]:
        """Marker ids, resolved once and cached. Raises if the tokenizer lacks one."""
        if getattr(self, "_token_ids", None) is None:
            self._token_ids = resolve_token_ids(self.processor.tokenizer, self.modality_specs)
        return self._token_ids

    def _modality_kwargs(self, modality: Dict[str, torch.Tensor],
                         input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Flat wire-format keys, after checking each spec's count against `input_ids`."""
        if not self.modality_specs:
            return {}
        token_ids = self._modality_token_ids()
        # `get_rope_index` derives the image/video count from the token right after
        # `<|vision_start|>`; a marker there gives silently wrong positions, not an error.
        # Checked here, per example, so a bad dataset format fails on the first batch.
        vision_start = self.processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        if isinstance(vision_start, int) and vision_start >= 0:
            starts = (input_ids == vision_start).nonzero(as_tuple=False)
            marker_ids = set(token_ids.values())
            for b, s in starts.tolist():
                if s + 1 < input_ids.shape[1] and int(input_ids[b, s + 1]) in marker_ids:
                    raise ValueError(
                        f"a modality marker sits immediately after <|vision_start|> at "
                        f"position {s + 1}. get_rope_index derives the image/video count "
                        "from that position; put the marker anywhere else."
                    )
        batch = ModalityBatch()
        for s in self.modality_specs:
            n_found = int((input_ids == token_ids[s.key]).sum())
            values = modality[s.key]
            if values.shape[0] != n_found:
                raise ValueError(
                    f"modality {s.token} : column {s.source_column!r} has "
                    f"{values.shape[0]} row(s) but the tokenized conversation contains "
                    f"{n_found} occurrence(s) of the marker. The dataset row and the "
                    "rendered text disagree."
                )
            batch.values[s.key] = values
            batch.counts[s.key] = torch.tensor([n_found], dtype=torch.long)
        return batch.to_kwargs()


@dataclass
class TurnEncoding:
    """What the trunk produced for one collated conversation, before any head runs.

    `pooled` is the seam of this codebase: everything upstream of it is the frozen
    trunk (backbone + LoRA + modality encoders + the pooling rule), everything
    downstream is the head. `harvest_context.py` writes exactly this vector to disk and
    `head_only.py` trains on it, which is only sound because `forward` reads the same
    one -- hence a named object rather than a tuple, so a caller cannot quietly take the
    unpooled states and call them the context.
    """

    pooled: torch.Tensor            # (N_turns, pooled_dim) -- head input
    states: torch.Tensor            # (N_turns, L_max, H) -- padded content states
    span_mask: torch.Tensor         # (N_turns, L_max) bool
    spans: List[Any]                # the TurnSpans, in row order
    n_sparse_tokens: int
    n_dense_tokens: int


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
        modality_embedder: Optional[ModalityEmbedder] = None,
        stop_head: Optional[StopHead] = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.head = head
        # None unless `model_cfg.stop_head` declares one, so the module list, the parameter
        # count, the loss and the saved blob are all unchanged without it.
        self.stop_head = stop_head
        self.normalizer = normalizer
        self.target_shape = tuple(target_shape)
        self.prefix_ids = list(prefix_ids)
        self.postfix_ids = list(postfix_ids)
        self.model_cfg = model_cfg
        self.loss_cfg = loss_cfg
        # A submodule, so its parameters are this model's parameters -- which is what puts
        # them in the optimizer's groups and makes DDP see them. Empty and inert unless
        # `model_cfg.modality_specs` declares something.
        self.modality_embedder = modality_embedder or ModalityEmbedder()
        # How many positions per turn the head pools. Observed on the first forward and
        # saved with the checkpoint, so inference can assert its affixes reproduce it --
        # a pooled head silently accepts the wrong count.
        self.train_content_len: Optional[int] = None

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
        prefix_ids, postfix_ids = resolve_affix_ids(
            processor.tokenizer, model_cfg.prefix, model_cfg.postfix
        )
        normalizer = TargetNormalizer(target_shape[-1], enabled=loss_cfg.normalize_targets)
        # Registered on the processor's tokenizer here, not by the caller: the markers must
        # be present before anything tokenizes, and this is the one place that sees both
        # the spec list and the tokenizer.
        embedder = ModalityEmbedder(model_cfg.modality_specs, d_model=hidden_size)
        if embedder:
            embedder.register(processor.tokenizer)
        # Sized off the head's own pooled width, so 'flat' pooling (which concatenates
        # `content_len` positions) cannot silently hand the stop head the wrong input dim.
        stop_head = (
            None if model_cfg.stop_head is None
            else StopHead(head.pooled_dim, model_cfg.stop_head)
        )
        model = cls(
            backbone, head, normalizer, target_shape, prefix_ids, postfix_ids,
            model_cfg, loss_cfg, modality_embedder=embedder, stop_head=stop_head,
        )
        model.attach_modality_hooks()
        return model

    def attach_modality_hooks(self) -> "TurnVectorRegressor":
        """(Re)install the embedding hooks. Idempotent, and a no-op with no specs.

        Called again whenever the backbone is re-wrapped (peft), because the hooks live on
        a module reached through the wrapper.
        """
        if self.modality_embedder:
            self.modality_embedder.attach(self.backbone.get_input_embeddings())
        return self

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
        mod = sum(p.numel() for p in self.modality_embedder.parameters() if p.requires_grad)
        stop = (0 if self.stop_head is None
                else sum(p.numel() for p in self.stop_head.parameters() if p.requires_grad))
        return (
            f"{trainable:,} trainable / {total:,} total params "
            f"({100 * trainable / total:.2f}%); head {head:,}, "
            f"modality encoders {mod:,}, stop head {stop:,}, "
            f"adapters {trainable - head - mod - stop:,}"
        )

    # -- the encoder half --------------------------------------------------------------
    def encode_turns(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **backbone_inputs,
    ) -> "TurnEncoding":
        """Batch -> one pooled trunk state per assistant turn. Everything before the head.

        Factored out of `forward` because three callers need *exactly* this and none of
        them may acquire its own version of it:

          * `forward` (this class and `TurnFlowPolicy`) -- training;
          * `data_scripts/harvest_context.py` -- writing the frozen-context dataset that
            head-only training consumes;
          * `data_scripts/reattach_head.py` -- proving, after reattachment, that the
            trunk still computes the context that was harvested from it.

        The modality wiring in particular has to be identical across all three. It is the
        step that fails silently: a harvest that never injected `<pose>` produces a
        perfectly plausible dataset conditioned on an input the model was not trained
        with, and nothing downstream would raise. There is now one place it can be got
        wrong, and every caller shares it.

        `backbone_inputs` is forwarded wholesale, so the `modality_*` keys must come out
        of it by name first -- the backbone does not accept them.
        """
        backbone_inputs.pop("labels", None)
        modality = ModalityBatch.pop_from(
            backbone_inputs, known_keys=self.modality_embedder.keys
        )
        # The context wraps only the backbone call: entering it twice under one context
        # trips the consume-once assert instead of silently reusing the same values.
        with self.modality_embedder.pending(modality):
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                logits_to_keep=1,  # never materialize (B, S, vocab); the LM head is unused
                **backbone_inputs,
            )
        # `head=None` and pool explicitly, so the motion head and the stop head read the
        # same pooled context from one extraction rather than two.
        states, spans, span_mask = extract_turn_vectors(
            outputs,
            input_ids,
            None,
            prefix_ids=self.prefix_ids,
            postfix_ids=self.postfix_ids,
            shift_left=self.model_cfg.shift_left,
            strict=True,
            return_mask=True,
        )
        if self.train_content_len is None and spans:
            self.train_content_len = int(len(spans[0]))
        return TurnEncoding(
            pooled=self.head.pooled_context(states, span_mask),
            states=states,
            span_mask=span_mask,
            spans=spans,
            n_sparse_tokens=int(outputs["last_hidden_state"].shape[1]),
            n_dense_tokens=int(input_ids.shape[1]),
        )

    def _assert_turn_alignment(self, n_vectors: int, n_targets: int,
                               input_ids: torch.Tensor) -> None:
        """Turn k's vector must line up with target k.

        A tokenizer or windowing change that shifts the count would otherwise train
        silently against mismatched pairs.
        """
        if n_vectors == n_targets:
            return
        tail = input_ids[0, -64:].tolist()
        raise RuntimeError(
            f"found {n_vectors} assistant turn span(s) but got {n_targets} target(s). "
            f"prefix={self.model_cfg.prefix!r} postfix={self.model_cfg.postfix!r} "
            f"shift_left={self.model_cfg.shift_left}. Last 64 input_ids: {tail}"
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
        stop_targets: Optional[torch.Tensor] = None,
        **backbone_inputs,
    ) -> Dict[str, torch.Tensor]:
        """`backbone_inputs` is whatever else the collator produced -- `pixel_values`,
        `image_grid_thw` and friends -- and it is forwarded to the backbone **wholesale**.

        That is why the modality keys have to come out first and by name. The backbone
        does not accept them, so anything left in here is a `TypeError` from deep inside
        the model; `pop_from` owns the entire `modality_*` prefix and raises on a key it
        does not recognise rather than letting one through.
        """
        enc = self.encode_turns(input_ids, attention_mask, **backbone_inputs)
        # `head.project(enc.pooled)` is exactly `self.head(states, span_mask)`; going
        # through the pooled seam keeps this identical to what the harvest records.
        vectors = self.head.project(enc.pooled)
        self._assert_turn_alignment(vectors.shape[0], targets.shape[0], input_ids)
        targets = targets.to(next(self.head.parameters()).dtype).to(vectors.device)
        pred = vectors.view(-1, *self.target_shape)
        tgt_norm = self.normalizer.normalize(targets)

        per_turn = self._elementwise_loss(pred, tgt_norm).flatten(1).mean(dim=1)
        total = per_turn.sum()
        denom = num_items_in_batch if num_items_in_batch is not None else per_turn.numel()
        denom = torch.as_tensor(denom, dtype=total.dtype, device=total.device).clamp(min=1)
        loss = total / denom

        # An example with no occurrences of a modality leaves its encoder out of the
        # backward graph, which under `ddp_find_unused_parameters=False` hangs or errors.
        # This term is identically zero -- it changes no gradient and no metric -- and
        # exists only so every encoder parameter is always reached.
        touch = self.modality_embedder.zero_touch(loss.device, loss.dtype)
        if touch is not None:
            loss = loss + touch

        motion_loss = loss
        stop_extra: Dict[str, torch.Tensor] = {}
        if self.stop_head is not None:
            pooled = enc.pooled
            if self.stop_head.cfg.stop_grad:
                # The constraint that makes this head free. Detached, the stop gradient
                # reaches the stop head's own parameters and stops: not the backbone, not
                # the adapters, not the motion head. So no weight of this loss can damage
                # the motion objective, and `loss_weight` needs no tuning.
                pooled = pooled.detach()
            stop_logits = self.stop_head(pooled)
            if stop_targets is not None:
                stop_targets = stop_targets.reshape(-1).to(stop_logits.device)
                if stop_targets.shape[0] != stop_logits.shape[0]:
                    raise RuntimeError(
                        f"{stop_logits.shape[0]} turn(s) but {stop_targets.shape[0]} stop "
                        "label(s); the stop labels were not sliced with the turn window"
                    )
                stop_loss = self.stop_head.loss(stop_logits, stop_targets)
                loss = loss + float(self.stop_head.cfg.loss_weight) * stop_loss.to(loss.dtype)
                stop_extra["stop_loss_sum"] = stop_loss.detach() * stop_logits.shape[0]
                stop_extra["stop_labels"] = stop_targets.detach().float()
            else:
                # No labels (inference, or a probe harvesting logits). Keep the head in the
                # backward graph anyway, for the same DDP reason as `zero_touch` above.
                loss = loss + stop_logits.sum() * 0.0
            stop_extra["stop_logits"] = stop_logits.detach().float()
            stop_extra["stop_n"] = torch.tensor(
                float(stop_logits.shape[0]), device=loss.device
            )
            # Only with a stop head, and deliberately: it exists to show how the total
            # splits between two objectives, and with one objective it would just be
            # `turn_loss` under a second name -- a new logged series on every existing run
            # for no information. Emitting it unconditionally is also what stopped this
            # tree being bit-identical to the base one; see `inertness_check.py`.
            stop_extra["motion_loss_sum"] = motion_loss.detach() * vectors.shape[0]

        with torch.no_grad():
            pred_orig = self.normalizer.denormalize(pred.detach())
            err = (pred_orig - targets).reshape(-1, self.target_shape[-1])
            metrics = {
                "loss_sum": total.detach(),
                "sum_sq_err": err.pow(2).sum(0),
                "sum_abs_err": err.abs().sum(0),
                "n_rows": torch.tensor(err.shape[0], device=err.device),
                "n_turns": torch.tensor(pred.shape[0], device=err.device),
                "n_tokens": torch.tensor(enc.n_sparse_tokens, device=err.device),
                "n_dense_tokens": torch.tensor(enc.n_dense_tokens, device=err.device),
                "n_steps": torch.tensor(1, device=err.device),
            }
        return {"loss": loss, **metrics, **stop_extra}

    # -- checkpointing -----------------------------------------------------------------
    def save_pretrained(self, output_dir: Union[str, Path]):
        """Save only what trains: the adapter, the head, and the normalizer buffers.

        Trainer's default would pickle the whole frozen backbone into every checkpoint.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(self.backbone, "save_pretrained") and hasattr(self.backbone, "peft_config"):
            self.backbone.save_pretrained(str(output_dir / ADAPTER_SUBDIR))
        blob = {
            "head": self.head.state_dict(),
            "normalizer": self.normalizer.state_dict(),
        }
        # A third top-level entry alongside "head" and "normalizer", written only when
        # something is declared -- so a no-spec checkpoint is byte-identical to one from
        # before this existed.
        modality = self.modality_embedder.state_blob()
        if modality is not None:
            blob["modality"] = modality
        if self.stop_head is not None:
            blob["stop_head"] = self.stop_head.state_dict()
        # SAFETY NET. Anything trainable that none of the entries above covers is saved
        # here, LOUDLY. A module attached to the model after `build()` -- which is how
        # heads get added -- lands in no state dict, so training succeeds, checkpoints
        # look complete, and the omission only surfaces when something tries to LOAD.
        # That cost a full run once. The warning is the point: silently saving it would
        # hide a module that nobody declared, which is its own failure.
        extra = self._uncovered_trainable(blob)
        if extra:
            mods = sorted({k.split(".")[0] for k in extra})
            warnings.warn(
                f"save_pretrained: {len(extra)} trainable tensor(s) in {mods} are not "
                f"covered by any declared save path; writing them to 'extra_trainable' "
                f"so the checkpoint is loadable. Give them an explicit home -- attaching "
                f"a module to a SAVED submodule (e.g. the normalizer) is usually right.",
                RuntimeWarning, stacklevel=2)
            blob["extra_trainable"] = extra
        torch.save(blob, output_dir / HEAD_WEIGHTS_FILE)
        model_meta = asdict(self.model_cfg)
        if model_meta.get("stop_head") is None:
            # Same reasoning as `modality_specs` below: omitted rather than written as
            # `null`, so a checkpoint without a stop head stays byte-identical to one
            # written before the field existed and still loads in an older reader.
            model_meta.pop("stop_head", None)
        if not model_meta.get("modality_specs"):
            # Omitted rather than written as `[]`, so a checkpoint with no modalities is
            # byte-identical to one written before this field existed -- and, more to the
            # point, still loadable by an older reader that would choke on an unexpected
            # `ModelConfig` keyword.
            model_meta.pop("modality_specs", None)
        (output_dir / HEAD_CONFIG_FILE).write_text(
            json.dumps(
                {
                    "model": model_meta,
                    "loss": asdict(self.loss_cfg),
                    "target_shape": list(self.target_shape),
                    "train_content_len": self.train_content_len,
                    "prefix_ids": self.prefix_ids,
                    "postfix_ids": self.postfix_ids,
                },
                indent=2,
            )
        )

    def _uncovered_trainable(self, blob) -> dict:
        """`{name: tensor}` for every `requires_grad` parameter no blob entry holds.

        Coverage is by IDENTITY, not by name: the same tensor can appear under different
        prefixes, and a name-based check would call a shared parameter uncovered.
        """
        covered = set()
        for mod_name in ("head", "normalizer", "modality_embedder", "stop_head"):
            mod = getattr(self, mod_name, None)
            if mod is not None and hasattr(mod, "parameters"):
                covered |= {id(q) for q in mod.parameters()}
        bb = getattr(self, "backbone", None)          # the adapter saves separately
        if bb is not None:
            covered |= {id(q) for q in bb.parameters()}
        return {n: q.detach().cpu() for n, q in self.named_parameters()
                if q.requires_grad and id(q) not in covered}

    def load_head_state(self, checkpoint_dir: Union[str, Path], strict: bool = True):
        """Load the head weights, the normalizer buffers and the modality encoders.

        Modality loading is strict *regardless of* `strict`. That flag exists to tolerate
        head-shape evolution; letting it also silently drop encoders would give a model
        that behaves differently from what its own config claims.
        """
        blob = torch.load(
            Path(checkpoint_dir) / HEAD_WEIGHTS_FILE, map_location="cpu", weights_only=False
        )
        self.head.load_state_dict(blob["head"], strict=strict)
        self.normalizer.load_state_dict(blob["normalizer"], strict=strict)
        self.modality_embedder.load_state_blob(blob.get("modality"))
        # The symmetric half of the safety net. Restored by name, and a name the model
        # does not have is an ERROR: it means the checkpoint carries a module this build
        # dropped, which is exactly the mismatch the net exists to make visible.
        _extra = blob.get("extra_trainable")
        if _extra:
            own = dict(self.named_parameters())
            missing = [k for k in _extra if k not in own]
            if missing:
                raise KeyError(
                    f"checkpoint has extra_trainable tensors this model has no home for: "
                    f"{missing[:5]}")
            warnings.warn(
                f"load_head_state: restoring {len(_extra)} tensor(s) from "
                f"'extra_trainable'; these had no declared save path when written",
                RuntimeWarning, stacklevel=2)
            with torch.no_grad():
                for k, v in _extra.items():
                    own[k].copy_(v.to(own[k].dtype))
        # Strict in both directions, for the same reason the modality encoders are: the
        # config decides whether a stop head exists, and a model that quietly has one more
        # or one fewer readout than its own config claims is worse than a load error.
        stop_blob = blob.get("stop_head")
        if self.stop_head is None:
            if stop_blob:
                raise RuntimeError(
                    "checkpoint carries stop-head weights but the config declares no stop "
                    "head; loading it would give a model that behaves differently from "
                    "what its config claims"
                )
        else:
            if not stop_blob:
                raise RuntimeError(
                    "config declares a stop head but the checkpoint has no stop-head "
                    "weights -- mismatched checkpoint"
                )
            self.stop_head.load_state_dict(stop_blob, strict=True)
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
        _shim_peft_tp_import()
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

    def warm_start(self, checkpoint_dir: Union[str, Path],
                   adapter: bool = True) -> List[str]:
        """Initialize from a checkpoint that predates some of this model's modules.

        Not a resume, and deliberately a separate entry point rather than a flag on
        `load_trainable`. A **resume** must be exactly as strict as it is: the checkpoint
        and the config describe the same run, so a declared module with no weights is a
        mismatched checkpoint and the DESIGN.md 7.1 rule that makes that a hard error is
        right. A **warm start** is the one case where it is legitimately not -- the new
        modules did not exist when those weights were written, and starting them fresh is
        the point.

        So exactly one thing is forgiven: a module this model declares that the checkpoint
        has *no entry for at all*. Everything else keeps full strictness --

          * `head` and `normalizer` load with `strict=True`; shape drift raises, since a
            shared module that only mostly matches is a config mismatch, not a warm start
          * a modality blob that IS present goes through `load_state_blob`, which is strict
            in both directions -- so weights for an undeclared spec, a declared spec missing
            from a blob that has other specs, and shape drift all still raise
          * stop-head weights for a model that declares no stop head raise, for the same
            reason: that is an unexpected module, not a missing one

        Returns the names of the modules left at their fresh initialization, so the caller
        can print them. Silence about which modules were NOT loaded is how a warm start
        turns into an accidental from-scratch run.
        """
        checkpoint_dir = Path(checkpoint_dir)
        blob = torch.load(
            checkpoint_dir / HEAD_WEIGHTS_FILE, map_location="cpu", weights_only=False
        )
        self.head.load_state_dict(blob["head"], strict=True)

        fresh: List[str] = []
        # The normalizer stays strict in both directions with ONE exception: a `LatentSplit`
        # installed on a `FlowActionCodec` is fresh by construction when warm-starting from a
        # deterministic checkpoint, which is the whole point of the CVAE conversion (see
        # docs/LATENT_RL.md). Missing `latent.*` keys are reported as fresh rather than
        # raising -- anything else missing still raises, because a normalizer that only
        # mostly matches is a config mismatch and not a warm start.
        norm_blob = blob["normalizer"]
        if getattr(self, "_reinit_decoder_on_warm_start", False):
            # Keep the backbone, the readout and the codec's buffers; drop the velocity
            # field, so it trains alongside the posterior instead of arriving already expert
            # at expressing the residual through its own base noise.
            norm_blob = {k: v for k, v in norm_blob.items() if not k.startswith("decoder.")}
            fresh.append("normalizer.decoder (reinit)")
        report = self.normalizer.load_state_dict(norm_blob, strict=False)
        unexpected = list(report.unexpected_keys)
        skip = ("latent.",) + (("decoder.",)
                               if getattr(self, "_reinit_decoder_on_warm_start", False)
                               else ())
        missing = [k for k in report.missing_keys if not k.startswith(skip)]
        if missing or unexpected:
            raise RuntimeError(
                "normalizer state_dict mismatch beyond the latent split: "
                f"missing={missing} unexpected={unexpected}"
            )
        if len(report.missing_keys) > len(missing):
            fresh.append("normalizer.latent")

        modality = blob.get("modality")
        if modality:
            self.modality_embedder.load_state_blob(modality)
        elif self.modality_embedder:
            fresh.extend(f"modality_embedder.{k}" for k in self.modality_embedder.keys)

        stop_blob = blob.get("stop_head")
        if stop_blob:
            if self.stop_head is None:
                raise RuntimeError(
                    "checkpoint carries stop-head weights but the config declares no stop "
                    "head; loading it would give a model that behaves differently from "
                    "what its config claims"
                )
            self.stop_head.load_state_dict(stop_blob, strict=True)
        elif self.stop_head is not None:
            fresh.append("stop_head")

        if adapter:
            self.load_adapter_state(checkpoint_dir)
        return fresh

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
        model_cfg = ModelConfig(**{**migrate_model_config(meta["model"]), **overrides})
        loss_cfg = LossConfig(**meta["loss"])
        model = cls.build(
            model_cfg, loss_cfg, lora=None,
            target_shape=meta["target_shape"], processor=processor, dtype=dtype,
        )
        model.train_content_len = meta.get("train_content_len")
        adapter_dir = checkpoint_dir / ADAPTER_SUBDIR
        if adapter_dir.exists():
            from peft import PeftModel

            # This already loads the adapter weights, so `load_trainable` must not do it
            # again -- hence adapter=False below.
            model.backbone = PeftModel.from_pretrained(model.backbone, str(adapter_dir))
            # The backbone was re-wrapped; re-point the hooks through the new wrapper.
            model.attach_modality_hooks()
        model.load_trainable(checkpoint_dir, adapter=False)
        if device:
            model.to(device)
        return model.eval()


def migrate_model_config(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Read a checkpoint's model config, translating the retired `affixes` name.

    Checkpoints written before the affixes became plain strings carry
    `affixes: "action" | "template"`. Translate rather than break them; the pairs are the
    ones those names stood for.
    """
    meta = dict(meta)
    kind = meta.pop("affixes", None)
    if kind is not None and "prefix" not in meta:
        pairs = {"action": (ACTION_PREFIX, ACTION_POSTFIX),
                 "template": (DEFAULT_PREFIX, DEFAULT_POSTFIX)}
        if kind not in pairs:
            raise ValueError(f"checkpoint has unknown affixes={kind!r}")
        meta["prefix"], meta["postfix"] = pairs[kind]
    return meta


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

    # Both reasons for the B == 1 rule are properties of the backbone: the sparse model
    # unions its visual keep mask across the batch, and turn extraction indexes row 0.
    # A subclass that has no backbone (head-only training on cached trunk states) has
    # neither, and batching there is the whole point -- so the rule is a class attribute
    # to be turned off deliberately rather than a constant to be worked around.
    requires_unit_batch = True

    def __init__(self, *args, data_config: Optional[DataConfig] = None,
                 eval_data_collator: Optional[Any] = None, **kwargs):
        super().__init__(*args, **kwargs)
        # Eval must not use the train collator: that one samples a *random* turn window
        # per row, so eval numbers would move for reasons unrelated to the model.
        self.eval_data_collator = eval_data_collator
        if self.requires_unit_batch and self.args.per_device_train_batch_size != 1:
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
        self._stop_logits: List[torch.Tensor] = []
        self._stop_labels: List[torch.Tensor] = []
        self._last_stop_scores: Optional[Tuple[Any, Any]] = None

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
                    "n_tokens", "n_dense_tokens", "n_steps", "motion_loss_sum",
                    "stop_loss_sum", "stop_n"):
            if key not in outputs:
                continue
            v = outputs[key].detach().float()
            self._sums[key] = v.clone() if key not in self._sums else self._sums[key] + v
        # Ranking metrics are not sums: average precision and AUC depend on the whole
        # ordering, so the scores have to be kept rather than reduced as they arrive.
        if "stop_logits" in outputs:
            self._stop_logits.append(outputs["stop_logits"].detach().float().cpu())
            labels = outputs.get("stop_labels")
            self._stop_labels.append(
                torch.full_like(self._stop_logits[-1], float("nan")) if labels is None
                else labels.detach().float().cpu()
            )

    def _drain_stop_scores(self) -> Tuple[Any, Any]:
        """The concatenated logits/labels seen since the last drain, gathered over ranks."""
        import numpy as np

        logits = self._stop_logits
        labels = self._stop_labels
        self._stop_logits, self._stop_labels = [], []
        if not logits:
            return np.zeros(0), np.zeros(0)
        s = torch.cat(logits).numpy()
        y = torch.cat(labels).numpy()
        if self.args.world_size > 1:
            # `all_gather_object` rather than a tensor gather: each rank has seen a
            # different number of turns, and padding to a common length then trimming would
            # need a length exchange anyway. These arrays are one float per turn.
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                bucket: List[Any] = [None] * self.args.world_size
                dist.all_gather_object(bucket, (s, y))
                s = np.concatenate([b[0] for b in bucket])
                y = np.concatenate([b[1] for b in bucket])
        keep = ~np.isnan(y)
        return s[keep], y[keep]

    def _dim_names(self, n: int) -> List[str]:
        names = self.data_config.target_dim_names
        return list(names) if names and len(names) == n else [str(i) for i in range(n)]

    def _drain_metrics(self, prefix: str = "") -> Dict[str, float]:
        stop_scores, stop_labels = self._drain_stop_scores()
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

        # Two objectives, two scalars. `turn_loss` above is the total; these say how it
        # splits, which is the only way an auxiliary that has gone flat is visible.
        if "motion_loss_sum" in sums:
            out[f"{prefix}motion_loss"] = float(sums["motion_loss_sum"] / turns)
        if "stop_loss_sum" in sums and "stop_n" in sums:
            out[f"{prefix}stop_loss"] = float(
                sums["stop_loss_sum"] / sums["stop_n"].clamp(min=1)
            )
        for k, v in stop_metrics(stop_scores, stop_labels).items():
            out[f"{prefix}{k}"] = v
        self._last_stop_scores = (stop_scores, stop_labels)
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
        """Manual eval loop with all-reduced sums, reported in original target units.

        A DICT of eval datasets is evaluated one component at a time, each under its own
        `eval_<name>_` prefix -- the same contract `transformers.Trainer` offers, which this
        override otherwise bypasses (it hands whatever it is given straight to
        `get_eval_dataloader`, and a dict then fails with `KeyError: 0` on the first fetch).
        That matters for a mixture: one blended number cannot say WHICH component regressed,
        and a run that forgets ObjectNav while improving PointNav reads as flat.
        """
        target = eval_dataset if eval_dataset is not None else self.eval_dataset
        if isinstance(target, dict):
            merged = {}
            for name, ds in target.items():
                merged.update(self.evaluate(
                    eval_dataset=ds, ignore_keys=ignore_keys,
                    metric_key_prefix=f"{metric_key_prefix}_{name}",
                ))
            # `_maybe_log_save_evaluate` and any `metric_for_best_model` want a bare
            # `<prefix>_loss`; without it a dict eval silently has no scalar to track.
            losses = [v for k, v in merged.items() if k.endswith("_loss")]
            if losses and f"{metric_key_prefix}_loss" not in merged:
                merged[f"{metric_key_prefix}_loss"] = sum(losses) / len(losses)
            self.log(merged)
            return merged
        dataloader = self.get_eval_dataloader(eval_dataset)
        model = self._wrap_model(self.model, training=False)
        was_training = model.training
        model.eval()
        self._sums = {}
        self._stop_logits, self._stop_labels = [], []
        for inputs in dataloader:
            inputs = self._prepare_inputs(inputs)
            inputs.pop("num_items_in_batch", None)
            outputs = model(**inputs)
            self._accumulate(outputs)

        # Sum every accumulator across ranks before turning them into rates.
        #
        # The key set must be AGREED across ranks before any reduce, and iterated in a
        # rank-independent order. `self._sums` is an ordinary dict, so iterating it
        # directly means iterating INSERTION order, and accumulators that are only
        # emitted for some rows (`state_probe.metrics` skips a component's stop keys
        # entirely when a row carries no finite label) are inserted at different points
        # -- or not at all -- on different ranks. The per-tensor all-reduces then pair
        # DIFFERENT KEYS across ranks and every affected metric comes out stable,
        # plausible and wrong: on this run pointnav's eval stop rate reduced to
        # 1787/4752 = 0.376, a sum across four ranks of `stop_pos` against `stop_n`
        # against `stop_fn`, when no row in that corpus exceeds 0.167. Training is
        # unaffected -- this reduce serves eval metrics only.
        if self.args.world_size > 1:
            import torch.distributed as _dist
            local = sorted(self._sums)
            keys = local
            if _dist.is_available() and _dist.is_initialized():
                gathered = [None] * _dist.get_world_size()
                _dist.all_gather_object(gathered, local)
                keys = sorted({k for part in gathered for k in part})
            reduced = {}
            for k in keys:
                v = self._sums.get(k)
                # A rank missing the key contributes zero rather than desynchronising
                # the collective -- every rank must call reduce the same number of
                # times, in the same order, or this deadlocks instead of merely lying.
                v = (torch.zeros((), device=self.args.device) if v is None
                     else v.to(self.args.device))
                reduced[k] = self.accelerator.reduce(v, reduction="sum")
            self._sums = reduced
        metrics = self._drain_metrics(prefix=f"{metric_key_prefix}_")
        self._save_stop_scores(metric_key_prefix)
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

    def _save_stop_scores(self, prefix: str):
        """Write this eval's raw stop logits and labels next to the checkpoints.

        The head is judged on ranking and its operating point is meant to be fitted after
        the fact, which is only possible if the scores survive the run. Saving the logits
        (not probabilities) keeps temperature a free parameter of that later fit.
        """
        scores = getattr(self, "_last_stop_scores", None)
        self._last_stop_scores = None
        if not scores or len(scores[0]) == 0 or self.args.local_rank not in (-1, 0):
            return
        import numpy as np

        out_dir = Path(self.args.output_dir) / "stop_scores"
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            out_dir / f"{prefix}_step{int(self.state.global_step)}.npz",
            logits=scores[0], labels=scores[1],
        )

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
