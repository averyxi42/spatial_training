"""
Per-turn continuous vectors from the sparse Qwen3-VL model.

The idea: every assistant turn is `prefix + content + postfix`, where prefix/postfix
are fixed chat-template structure (e.g. `<|im_start|>assistant\\n` ... `<|im_end|>`).
We locate the prefix/postfix token patterns in the *dense* `input_ids`, take the
content indices between them, remap those indices through the sparse model's
`seq_keep_mask`, gather `last_hidden_state` at the remapped positions and push them
through a small head to get ONE vector per assistant turn.

Why the remap is needed
-----------------------
`Qwen3VLSparseTextModel` (see `longnav/utils/modeling.py`) drops redundant *visual*
tokens before the LM forward, so `last_hidden_state` is shorter than `input_ids`.
Only visual tokens are ever dropped (`modeling.py` masks `visual_pos_masks`
positions and then re-adds the keepers), so assistant text tokens always survive --
but their positions shift left. The shift is a prefix-sum over the keep mask:

    ranks = seq_keep_mask.long().cumsum(0)
    sparse_idx = ranks[dense_idx] - 1

which is the same math as `VLMWorker._get_sparse_logit_indices`. This works for both
the single-shot path (one mask for the whole sequence) and the incremental/KV-cache
rollout path, where the per-turn masks concatenate into a full-length mask exactly as
`VLMWorker.infer_step` accumulates them -- pass the concatenated mask and the maths is
unchanged.

Typical use
-----------
    from longnav.utils.turn_vectors import (
        TurnVectorHead, resolve_affix_ids, extract_turn_vectors,
    )

    prefix_ids, postfix_ids = resolve_affix_ids(processor.tokenizer)
    head = TurnVectorHead(hidden_size=model.config.text_config.hidden_size,
                          out_dim=256, mode="mean").to(model.device)
    outputs = model(**inputs, use_cache=False, logits_to_keep=1)   # sparse forward
    vectors, turns = extract_turn_vectors(outputs, inputs["input_ids"], head,
                                         prefix_ids=prefix_ids,
                                         postfix_ids=postfix_ids)
    # vectors: (n_turns, out_dim)

Batching note: helper signatures are batch-shaped, but `modeling.py` only supports
B == 1 for the sparsification path today (it unions keep masks across the batch and
has explicit TODOs for B > 1), so run with B == 1 until that is fixed.
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Standard Qwen chat-template affixes. Content is then the raw assistant message,
# e.g. "**____**".
DEFAULT_PREFIX = "<|im_start|>assistant\n"
DEFAULT_POSTFIX = "<|im_end|>"

# Action-style affixes matching `VLMWorker`'s defaults; content is then the bare
# action word or placeholder, e.g. "____" (the `**` markers become part of the affixes).
ACTION_PREFIX = "<|im_start|>assistant\n**"
ACTION_POSTFIX = "**<|im_end|>"

POOLING_MODES = ("mean", "last", "attn", "flat")


def resolve_affix_ids(
    tokenizer,
    prefix: str = DEFAULT_PREFIX,
    postfix: str = DEFAULT_POSTFIX,
) -> Tuple[List[int], List[int]]:
    """Tokenize the prefix/postfix strings into id lists usable by `find_turn_spans`.

    Mirrors how `VLMWorker.__init__` builds `self.prefix_ids` / `self.postfix_ids`.
    Use `ACTION_PREFIX` / `ACTION_POSTFIX` for the `**action**` convention.
    """
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    postfix_ids = tokenizer.encode(postfix, add_special_tokens=False)
    if not prefix_ids or not postfix_ids:
        raise ValueError("prefix/postfix tokenized to an empty id list")
    return prefix_ids, postfix_ids


@dataclass
class TurnSpan:
    """One assistant turn's content, located in a token sequence.

    Attributes:
        batch_idx: row of `input_ids` this turn came from.
        start: first content token index (inclusive), in the DENSE sequence.
        end: one past the last content token index, in the DENSE sequence.
        indices: token positions to gather hidden states at. Equal to
            `arange(start, end)` before `to_sparse_indices` is applied, and the
            remapped positions into the sparse sequence afterwards.
        shifted: whether the span was shifted left by one to align with the
            hidden states that *predict* the content tokens.
    """

    batch_idx: int
    start: int
    end: int
    indices: torch.Tensor
    shifted: bool = False

    def __len__(self) -> int:
        return int(self.indices.numel())


def _find_subsequence(seq: torch.Tensor, sub: Sequence[int]) -> torch.Tensor:
    """Start indices of every occurrence of `sub` inside 1-D `seq`."""
    win = len(sub)
    if win == 0 or seq.numel() < win:
        return torch.empty(0, dtype=torch.long, device=seq.device)
    pattern = torch.as_tensor(sub, dtype=seq.dtype, device=seq.device)
    windows = seq.unfold(0, win, 1)  # (S - win + 1, win)
    return (windows == pattern).all(dim=1).nonzero(as_tuple=True)[0]


def find_turn_spans(
    input_ids: torch.Tensor,
    prefix_ids: Sequence[int],
    postfix_ids: Sequence[int],
    shift_left: bool = False,
    valid_mask: Optional[torch.Tensor] = None,
) -> List[List[TurnSpan]]:
    """Locate every assistant turn's content span in `input_ids`.

    Generalizes `VLMWorker._get_sandwich_indices`, which asserts exactly one
    prefix/postfix pair; here every turn in a multi-turn conversation is paired up,
    using the same rule as `collators.mask_user_turns`: content runs from the end of
    a prefix match to the start of the FIRST postfix match at or after it.

    Args:
        input_ids: (B, S) token ids. Must be the dense, pre-sparsification ids.
        prefix_ids: token ids marking the start of a turn (see `resolve_affix_ids`).
        postfix_ids: token ids marking the end of a turn.
        shift_left: if True, shift each span left by one so the gathered hidden
            states are the ones that *predict* the content tokens (next-token /
            logit-prior alignment, matching `logit_start = prefix_end - 1` in
            `VLMWorker._get_sandwich_indices`). Index `start - 1` is the last prefix
            token, i.e. still a text token, so it is never dropped by sparsification.

            With single-token content this makes the span land entirely on the last
            prefix token -- the same token in every turn (e.g. the `**` of
            `ACTION_PREFIX`), so the vector is a pure function of context rather than
            of the content value. For turn-by-turn inference that is the point: the
            content is a fixed template, is not yet generated, or is not meant to
            influence the vector, and `shift_left=True` gives you positions whose
            identity is fixed and whose value comes only from context.

            The real constraint to know about is that this function needs the postfix
            to be present in order to close a turn, so a trailing unterminated turn
            (a generation prompt) is skipped. Extracting a vector for a turn that has
            not been generated yet needs a mode that takes prefix + known content
            length and does not look for a postfix.
        valid_mask: optional (B, S) bool/int mask of real (non-pad) tokens. Spans
            touching padding are discarded.

    Returns:
        Per-batch-row lists of `TurnSpan`, ordered by position in the sequence.
        Empty spans (a prefix immediately followed by a postfix) are skipped.
    """
    if input_ids.dim() != 2:
        raise ValueError(f"input_ids must be (B, S), got {tuple(input_ids.shape)}")

    prefix_len = len(prefix_ids)
    results: List[List[TurnSpan]] = []

    for b in range(input_ids.shape[0]):
        row = input_ids[b]
        prefix_starts = _find_subsequence(row, prefix_ids).tolist()
        postfix_starts = _find_subsequence(row, postfix_ids).tolist()
        row_valid = None if valid_mask is None else valid_mask[b].bool()

        spans: List[TurnSpan] = []
        cursor = 0  # earliest token index still available for a new turn
        p_idx = 0  # pointer into postfix_starts
        for prefix_start in prefix_starts:
            if prefix_start < cursor:
                # Overlaps a turn we already consumed (can happen if the prefix
                # pattern also occurs inside a previous turn's content).
                continue
            content_start = prefix_start + prefix_len
            while p_idx < len(postfix_starts) and postfix_starts[p_idx] < content_start:
                p_idx += 1
            if p_idx >= len(postfix_starts):
                break  # unterminated final turn (e.g. a generation prompt)
            content_end = postfix_starts[p_idx]

            span_start, span_end = content_start, content_end
            if shift_left:
                span_start, span_end = span_start - 1, span_end - 1
            if span_end <= span_start or span_start < 0:
                cursor = content_end
                continue
            if row_valid is not None and not bool(row_valid[span_start:span_end].all()):
                cursor = content_end
                continue

            spans.append(
                TurnSpan(
                    batch_idx=b,
                    start=span_start,
                    end=span_end,
                    indices=torch.arange(span_start, span_end, dtype=torch.long),
                    shifted=shift_left,
                )
            )
            cursor = content_end  # next turn must start after this postfix
        results.append(spans)

    return results


def sparse_rank_map(seq_keep_mask: torch.Tensor) -> torch.Tensor:
    """Dense position -> sparse position lookup table.

    `ranks[i] - 1` is the index of dense token `i` inside the sparsified sequence
    (only meaningful where `seq_keep_mask[i]` is True). Same construction as
    `VLMWorker._get_sparse_logit_indices`.
    """
    if seq_keep_mask.dim() != 1:
        raise ValueError(
            f"seq_keep_mask must be 1-D (S,), got {tuple(seq_keep_mask.shape)}"
        )
    return seq_keep_mask.long().cumsum(dim=0) - 1


def map_dense_indices(
    indices: torch.Tensor,
    seq_keep_mask: torch.Tensor,
    strict: bool = True,
) -> torch.Tensor:
    """Remap dense token indices onto the sparsified sequence.

    Args:
        indices: 1-D long tensor of dense positions.
        seq_keep_mask: (S,) bool mask as returned by the sparse model
            (`outputs.seq_keep_mask`), or the concatenation of per-turn masks from
            an incremental rollout.
        strict: if True, raise when any requested position was dropped. Dropped
            positions silently collapse onto a neighbour, so leaving this on is
            the guard against pooling over the wrong tokens.
    """
    mask = seq_keep_mask.bool().to(indices.device)
    if strict and not bool(mask[indices].all()):
        dropped = indices[~mask[indices]].tolist()
        raise ValueError(
            f"{len(dropped)} requested token(s) were dropped by sparsification "
            f"(first few: {dropped[:8]}). Content spans must contain only tokens "
            "the sparse model keeps -- text tokens always are, visual tokens may "
            "not be."
        )
    return sparse_rank_map(mask)[indices]


def to_sparse_indices(
    turns: List[List[TurnSpan]],
    seq_keep_mask: Optional[torch.Tensor],
    strict: bool = True,
) -> List[List[TurnSpan]]:
    """Return copies of `turns` whose `.indices` point into the sparse sequence.

    A `seq_keep_mask` of None (or a non-tensor, which is how `modeling.py` signals
    "keep everything") is a no-op passthrough.
    """
    if seq_keep_mask is None or not isinstance(seq_keep_mask, torch.Tensor):
        return turns
    return [
        [
            TurnSpan(
                batch_idx=s.batch_idx,
                start=s.start,
                end=s.end,
                indices=map_dense_indices(s.indices, seq_keep_mask, strict=strict),
                shifted=s.shifted,
            )
            for s in row
        ]
        for row in turns
    ]


def gather_turn_states(
    last_hidden_state: torch.Tensor,
    turns: List[List[TurnSpan]],
) -> Tuple[torch.Tensor, torch.Tensor, List[TurnSpan]]:
    """Gather per-turn content hidden states into one padded tensor.

    Args:
        last_hidden_state: (B, S_sparse, H) from the model output.
        turns: spans whose `.indices` already point into `S_sparse`
            (i.e. after `to_sparse_indices`).

    Returns:
        states: (N_turns, L_max, H) padded with zeros, flattened over the batch.
        mask: (N_turns, L_max) bool, True on real content positions.
        flat_turns: the spans in the same order as the rows of `states`.
    """
    flat = [s for row in turns for s in row]
    if not flat:
        h = last_hidden_state.shape[-1]
        empty = last_hidden_state.new_zeros((0, 0, h))
        return empty, empty.new_zeros((0, 0), dtype=torch.bool), []

    seq_len = last_hidden_state.shape[1]
    lengths = [len(s) for s in flat]
    max_len = max(lengths)
    device = last_hidden_state.device

    states = last_hidden_state.new_zeros((len(flat), max_len, last_hidden_state.shape[-1]))
    mask = torch.zeros((len(flat), max_len), dtype=torch.bool, device=device)
    for i, span in enumerate(flat):
        idx = span.indices.to(device)
        if int(idx.max()) >= seq_len:
            raise ValueError(
                f"turn index {int(idx.max())} out of range for hidden states of "
                f"length {seq_len}; did you forget `to_sparse_indices`?"
            )
        states[i, : idx.numel()] = last_hidden_state[span.batch_idx, idx]
        mask[i, : idx.numel()] = True
    return states, mask, flat


class TurnVectorHead(nn.Module):
    """Pools a turn's content hidden states into a single vector.

    Modes:
        mean: masked mean over content positions (any content length).
        last: hidden state of the final content position (any content length).
        attn: learned-query single-head attention pooling (any content length).
        flat: concatenate all content positions and run an MLP. Requires a fixed
            `content_len` -- the cheapest option when every turn has the same
            content length.

    In every mode the pooled representation goes through an optional MLP trunk and a
    final projection to `out_dim`. Inputs are cast to `dtype` (fp32 by default, since
    the backbone runs in bf16), following the `ValueHead` pattern in `vlm_worker.py`.

    `standardize=True` additionally centers/scales each input feature dimension by
    dataset statistics that you supply once via `fit_input_stats`. These live in
    registered buffers, so they are saved and loaded with the state dict and applied
    automatically at inference -- there is no separate normalization step to remember.
    Leave it off (the default) when the head is trained jointly with the backbone or on
    plenty of data; see `fit_input_stats` for when it actually helps.
    """

    def __init__(
        self,
        hidden_size: int,
        out_dim: int,
        mode: str = "mean",
        content_len: Optional[int] = None,
        hidden_dims: Sequence[int] = (),
        dropout: float = 0.0,
        layer_norm: bool = True,
        normalize_output: bool = False,
        standardize: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if mode not in POOLING_MODES:
            raise ValueError(f"mode must be one of {POOLING_MODES}, got {mode!r}")
        if mode == "flat" and content_len is None:
            raise ValueError("mode='flat' requires content_len (fixed tokens per turn)")

        self.mode = mode
        self.hidden_size = hidden_size
        self.out_dim = out_dim
        self.content_len = content_len
        self.normalize_output = normalize_output
        self.standardize = standardize
        self._dtype = dtype

        # Buffers, not parameters: fixed statistics that travel with the checkpoint.
        self.register_buffer("input_mean", torch.zeros(hidden_size, dtype=dtype))
        self.register_buffer("input_std", torch.ones(hidden_size, dtype=dtype))
        self.register_buffer("stats_fitted", torch.zeros((), dtype=torch.bool))

        if mode == "attn":
            self.query = nn.Parameter(torch.randn(hidden_size, dtype=dtype) * hidden_size ** -0.5)
            self.key_proj = nn.Linear(hidden_size, hidden_size, dtype=dtype)
            self.value_proj = nn.Linear(hidden_size, hidden_size, dtype=dtype)

        pooled_dim = hidden_size * content_len if mode == "flat" else hidden_size
        self.pre_norm = nn.LayerNorm(pooled_dim, dtype=dtype) if layer_norm else nn.Identity()

        layers: List[nn.Module] = []
        curr = pooled_dim
        for h in hidden_dims:
            layers += [nn.Linear(curr, h, dtype=dtype), nn.Mish(), nn.Dropout(dropout)]
            curr = h
        layers.append(nn.Linear(curr, out_dim, dtype=dtype))
        self.mlp = nn.Sequential(*layers)

    @torch.no_grad()
    def fit_input_stats(self, states: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """Record per-dimension mean/std of the content hidden states.

        Call once on the training split, before training, when `standardize=True`.

        Why this is ever needed: `last_hidden_state` has already been through the text
        model's final RMSNorm, which is why the pretrained `lm_head` consumes it
        directly with no extra work. But RMSNorm only fixes the overall scale -- it does
        not center the stream or equalize per-dimension variance, and transformer
        residual streams carry a few outlier dimensions orders of magnitude larger than
        the rest. `lm_head` was trained jointly with the backbone and its weights are
        adapted to that geometry. A freshly initialized head is not: the informative
        variation sits in a tiny fraction of the total variance, which makes the problem
        badly conditioned when you have a frozen backbone, few samples and few steps.
        Standardizing removes the dominant constant directions so the small task-relevant
        signal is actually reachable by the optimizer.

        This is a conditioning aid for the small-data / frozen-backbone regime, not a
        correctness requirement. Trained jointly (or with enough data and steps), a head
        learns the same rescaling itself and `standardize=False` is fine.
        """
        flat = states.reshape(-1, states.shape[-1]) if mask is None else states[mask.bool()]
        flat = flat.to(self.input_mean.dtype).to(self.input_mean.device)
        self.input_mean.copy_(flat.mean(0))
        self.input_std.copy_(flat.std(0).clamp(min=1e-5))
        self.stats_fitted.fill_(True)

    def pool(self, states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """(N, L, H) + (N, L) bool -> (N, H) (or (N, L*H) for mode='flat')."""
        keep = mask.unsqueeze(-1).to(states.dtype)
        if self.mode == "mean":
            return (states * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1.0)
        if self.mode == "last":
            last_idx = (mask.long().sum(dim=1) - 1).clamp(min=0)
            return states[torch.arange(states.shape[0], device=states.device), last_idx]
        if self.mode == "attn":
            scores = self.key_proj(states) @ self.query / self.hidden_size ** 0.5
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
            weights = scores.softmax(dim=-1).unsqueeze(-1)
            return (self.value_proj(states) * weights).sum(dim=1)
        # flat
        if states.shape[1] != self.content_len:
            raise ValueError(
                f"mode='flat' expects exactly {self.content_len} content tokens per "
                f"turn, got {states.shape[1]}"
            )
        if not bool(mask.all()):
            raise ValueError("mode='flat' requires equal-length turns (no padding)")
        return states.flatten(1)

    def pooled_context(self, states: torch.Tensor,
                       mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """(N_turns, L, H) -> the pooled vector the MLP consumes, `(N_turns, pooled_dim)`.

        Split out of `forward` so an auxiliary readout (the stop head) can be trained on
        *the same* context this head regresses from, rather than on a second pooling that
        would only mostly agree with it. `forward` is the sole caller besides those, so
        there is still one definition of the cast, the standardization and the pooling.
        """
        states = states.to(self._dtype)
        if mask is None:
            mask = torch.ones(states.shape[:2], dtype=torch.bool, device=states.device)
        if self.standardize:
            if not bool(self.stats_fitted):
                raise RuntimeError(
                    "standardize=True but input stats were never fitted; call "
                    "fit_input_stats(train_states, train_mask) once before training "
                    "(the stats are then saved with the state dict)."
                )
            states = (states - self.input_mean) / self.input_std
        return self.pool(states, mask)

    @property
    def pooled_dim(self) -> int:
        """Width of `pooled_context`'s output -- what an auxiliary head must accept."""
        return self.hidden_size * self.content_len if self.mode == "flat" else self.hidden_size

    def project(self, pooled: torch.Tensor) -> torch.Tensor:
        """`(N_turns, pooled_dim)` -> `(N_turns, out_dim)`. The trainable half of the head.

        Split out of `forward` for the same reason `pooled_context` was: something other
        than `forward` needs exactly this computation and must not acquire a second
        definition of it. Here that something is head-only training
        (`longnav.utils.head_only`), which replays a *cached* pooled vector -- the frozen
        trunk's output -- through the head. `forward` is pooling followed by projection,
        and the seam between them is precisely where a harvested context is cut.
        """
        out = self.mlp(self.pre_norm(pooled.to(self._dtype)))
        return F.normalize(out, dim=-1) if self.normalize_output else out

    def forward(self, states: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """(N_turns, L, H) -> (N_turns, out_dim)."""
        if states.shape[0] == 0:
            return states.to(self._dtype).new_zeros((0, self.out_dim))
        return self.project(self.pooled_context(states, mask))


def extract_turn_vectors(
    outputs: Any,
    input_ids: torch.Tensor,
    head: Optional[TurnVectorHead] = None,
    *,
    prefix_ids: Sequence[int],
    postfix_ids: Sequence[int],
    shift_left: bool = False,
    seq_keep_mask: Optional[torch.Tensor] = None,
    valid_mask: Optional[torch.Tensor] = None,
    strict: bool = True,
    return_mask: bool = False,
) -> Tuple[torch.Tensor, List[TurnSpan]]:
    """One vector per assistant turn, end to end.

    Args:
        outputs: model output with `last_hidden_state` (and, for the sparse model,
            `seq_keep_mask`), or a raw (B, S_sparse, H) hidden-state tensor.
        input_ids: (B, S) DENSE token ids that produced `outputs`.
        head: pooling head. If None, returns the padded content states instead of
            pooled vectors (useful for inspection / custom heads).
        prefix_ids / postfix_ids: turn affix token ids (see `resolve_affix_ids`).
        shift_left: align to the states that predict the content (see
            `find_turn_spans`).
        seq_keep_mask: overrides `outputs.seq_keep_mask`. Pass the concatenated
            per-turn masks when replaying an incremental rollout.
        valid_mask: optional (B, S) non-pad mask for `input_ids`.
        strict: error out if a content token was dropped by sparsification.
        return_mask: also return the (N_turns, L_max) validity mask. Off by default
            so the arity of the common call is unchanged; a caller that wants the
            padded states *and* intends to pool them itself needs the mask, and
            reconstructing it from the spans downstream would be a second definition
            of the same thing.

    Returns:
        vectors: (N_turns, out_dim) -- or (N_turns, L_max, H) if `head` is None.
        flat_turns: the `TurnSpan`s in row order. `.start`/`.end` stay in dense
            coordinates (decode them against `input_ids` to verify alignment),
            `.indices` are the sparse positions actually gathered.
        mask: only when `return_mask`; (N_turns, L_max) bool, True on real content.
    """
    if isinstance(outputs, torch.Tensor):
        hidden = outputs
    else:
        hidden = outputs["last_hidden_state"]
        if seq_keep_mask is None:
            seq_keep_mask = (
                outputs["seq_keep_mask"] if "seq_keep_mask" in outputs else None
            )

    turns = find_turn_spans(
        input_ids,
        prefix_ids,
        postfix_ids,
        shift_left=shift_left,
        valid_mask=valid_mask,
    )
    turns = to_sparse_indices(turns, seq_keep_mask, strict=strict)
    states, mask, flat_turns = gather_turn_states(hidden, turns)
    out = states if head is None else head(states, mask)
    return (out, flat_turns, mask) if return_mask else (out, flat_turns)


def patch_flash_attention_packing():
    """Stop FA2 from mistaking sparsified position_ids for a packed sequence.

    `VLMWorker.__init__` applies this same patch before loading the model; call it
    if you build the sparse model directly and want `flash_attention_2`. Not needed
    for `sdpa`.
    """
    import transformers.modeling_flash_attention_utils as fa_utils

    fa_utils._is_packed_sequence = lambda position_ids, batch_size: False
