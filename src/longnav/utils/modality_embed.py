"""
Learned per-occurrence embeddings injected at marker tokens.

A general mechanism for putting a continuous value into the conversation at a designated
token. Nothing here knows what the value *means* -- it is `F` floats per occurrence, and
an encoder turns those into one `d_model` vector. See `dump/modality_embed/DESIGN.md`.

The binding, and the only one
----------------------------
    the k-th occurrence of a marker token in the sequence receives the k-th value row.

That is the whole contract. Turns do not appear in it. A marker every other turn, two in
one message, one in the system prompt, an example with none at all, two modalities at
different rates -- all of these are free rather than cases to handle. The per-example
dimension is `N`, the number of marker occurrences in that example, which is
data-dependent and in general **not** the turn count. Any `N == num_turns` check belongs
in whatever writes the markers, which knows it chose one per turn; putting it here would
silently reimpose the constraint one layer down.

Wire format, per spec, flat over the batch (mirroring `pixel_values` + `image_grid_thw`):

    modality_<key>_values : (sum N, F) float32, occurrence order, concatenated over batch
    modality_<key>_counts : (B,)       int64,   occurrences per example

Flat and not padded-plus-valid-mask: because alignment *is* occurrence order, flat
concatenation in batch order is already row-major over `(b, position)` -- exactly the
order `masked_scatter` enumerates -- so no reshape and no validity mask are needed. The
counts serve the per-example assertion and reconstruction. The check must be per example:
a total-only check passes when example A has one occurrence too many and B one too few,
which is precisely the misalignment that trains plausibly and means nothing.

How the injection happens
-------------------------
Two hooks on the backbone's input embedding module, and the marker id never reaches the
embedding table:

  * a **forward pre-hook** rewrites marker ids in `embed_tokens`' *argument* to a safe id,
    so the table is never indexed at a row that may not exist. `input_ids` itself is left
    alone and continues to the rest of the model unchanged -- which is what preserves
    `get_rope_index` and `get_placeholder_mask` behaviour (both treat a marker as an
    ordinary text token, verified: an id-for-id substitution leaves every position id and
    the rope delta bit-identical).
  * a **forward hook** scatters `encoder(values)` at the positions the pre-hook recorded,
    and asserts it consumed exactly that many rows at exactly those positions.

The two hooks verify each other, so a skipped or misaligned scatter cannot pass silently.
Nothing is added to the base model's embedding table, so there is no dependency on free
vocab rows, no `resize_token_embeddings`, no `modules_to_save`, and no coupling to
`vocab_size`.

Constraint on marker placement
------------------------------
`get_rope_index` decides image-vs-video by reading `input_ids[vision_start_index + 1]`.
A marker written *immediately after* `<|vision_start|>` would make that position neither
an image nor a video token, and the whole span would be scored as text -- silently wrong
positions, no error. `ModalityEmbedder.check_placement` asserts against it. Anywhere else
in the text is fine.

Inertness
---------
With no specs registered, nothing is constructed, no hooks are attached, and every entry
point returns immediately. Existing checkpoints and configs are unaffected.
"""

from __future__ import annotations

import contextlib
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

# Every dataloader key this mechanism owns starts with this. `ModalityBatch.pop_from` is
# the one function that knows the convention, and it claims the entire prefix: an
# unrecognised `modality_*` key is an error, never something forwarded to the backbone.
MODALITY_PREFIX = "modality_"
VALUES_SUFFIX = "_values"
COUNTS_SUFFIX = "_counts"

# The id the marker is rewritten to before `embed_tokens` sees it. Its embedding is
# overwritten by the scatter, so the choice is arbitrary -- it only has to be a valid row.
SAFE_EMBED_ID = 0


# ======================================================================================
# Spec
# ======================================================================================
@dataclass
class ModalityEmbedSpec:
    """Declarative description of one modality. Serialised into the checkpoint config.

    The token string is the identity; `key` is derived from it. There is deliberately no
    separate `name` field -- a second identifier buys renaming stability and costs the
    ability to silently disagree with the token that is actually in the text.

    `d_model` is *not* here. It is injected at construction from the backbone config, so
    it cannot drift from the model it has to match.
    """

    token: str
    n_features: int
    encoder: str
    encoder_kwargs: Dict[str, Any] = field(default_factory=dict)
    # Dataset column feeding this spec. None -> `key`. This is the data source, not a
    # second identity: it never participates in matching a spec to weights or to a token.
    column: Optional[str] = None

    def __post_init__(self):
        if not (self.token.startswith("<") and self.token.endswith(">") and len(self.token) > 2):
            raise ValueError(
                f"modality token {self.token!r} must look like '<name>': the angle brackets "
                "are what keep it out of ordinary text and make the corpus check meaningful"
            )
        if int(self.n_features) < 1:
            raise ValueError(f"{self.token}: n_features must be >= 1, got {self.n_features}")
        self.n_features = int(self.n_features)
        self.encoder_kwargs = dict(self.encoder_kwargs or {})

    @property
    def key(self) -> str:
        """Dict key / wire-format name: the token without its brackets."""
        return self.token.strip("<>")

    @property
    def source_column(self) -> str:
        return self.column if self.column else self.key

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ModalityEmbedSpec":
        known = {"token", "n_features", "encoder", "encoder_kwargs", "column"}
        unknown = set(d) - known
        if unknown:
            raise ValueError(
                f"unknown ModalityEmbedSpec field(s) {sorted(unknown)}; known: {sorted(known)}"
            )
        return cls(**d)


def coerce_specs(specs: Optional[Iterable[Any]]) -> Tuple[ModalityEmbedSpec, ...]:
    """Accept specs as dataclasses or plain dicts (the checkpoint-config form)."""
    if not specs:
        return ()
    out = [s if isinstance(s, ModalityEmbedSpec) else ModalityEmbedSpec.from_dict(s)
           for s in specs]
    tokens = [s.token for s in out]
    if len(set(tokens)) != len(tokens):
        dupes = sorted({t for t in tokens if tokens.count(t) > 1})
        raise ValueError(f"duplicate modality token(s) {dupes}: the token is the identity")
    return tuple(out)


# ======================================================================================
# Encoder registry
# ======================================================================================
ENCODER_REGISTRY: Dict[str, type] = {}


def register_encoder(name: str):
    """Class decorator adding an encoder to the registry under `name`.

    The registry key is what the checkpoint stores, so code is never serialised. An
    unknown key at load time raises and lists what is known.
    """

    def deco(cls):
        if name in ENCODER_REGISTRY and ENCODER_REGISTRY[name] is not cls:
            raise ValueError(f"encoder key {name!r} is already registered to "
                             f"{ENCODER_REGISTRY[name].__name__}")
        ENCODER_REGISTRY[name] = cls
        return cls

    return deco


def build_encoder(spec: ModalityEmbedSpec, d_model: int) -> "ModalityEncoder":
    if spec.encoder not in ENCODER_REGISTRY:
        raise KeyError(
            f"unknown modality encoder {spec.encoder!r} for {spec.token}; "
            f"known: {sorted(ENCODER_REGISTRY)}"
        )
    return ENCODER_REGISTRY[spec.encoder](
        n_features=spec.n_features, d_model=d_model, **spec.encoder_kwargs
    )


class ModalityEncoder(nn.Module):
    """`(N, F) -> (N, d_model)`. Owns all of its parameters; nothing lives in the base
    model's tables.

    Subclasses implement `encode`, which is always called in fp32 with autocast disabled.
    The backbone runs under bf16 autocast, and continuous features can lose real precision
    there -- bf16 has 8 mantissa bits, so nearby values collapse onto the same
    representation before the encoder ever sees them. The cast to the embedding dtype
    happens once, at the very end, in `forward`.
    """

    def __init__(self, n_features: int, d_model: int):
        super().__init__()
        self.n_features = int(n_features)
        self.d_model = int(d_model)

    def encode(self, values: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def forward(self, values: torch.Tensor, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        if values.dim() != 2 or values.shape[1] != self.n_features:
            raise ValueError(
                f"{type(self).__name__} expects (N, {self.n_features}), got "
                f"{tuple(values.shape)}"
            )
        with torch.autocast(device_type=values.device.type, enabled=False):
            out = self.encode(values.float())
        if out.shape != (values.shape[0], self.d_model):
            raise RuntimeError(
                f"{type(self).__name__}.encode returned {tuple(out.shape)}, expected "
                f"{(values.shape[0], self.d_model)}"
            )
        return out if dtype is None else out.to(dtype)


def _zero_init(linear: nn.Linear) -> nn.Linear:
    """Zero a module's *output* layer, so step 0 is bit-identical to no injection.

    Safe specifically because it is the last layer: the gradient with respect to its own
    weights is proportional to its (nonzero) input, so it moves off zero immediately.
    Zeroing a *first* layer is the opposite case and severs the pathway -- see
    `dump/cnf_head/flow/bugfix/FINDINGS.md`.
    """
    nn.init.zeros_(linear.weight)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)
    return linear


@register_encoder("constant")
class ConstantEncoder(ModalityEncoder):
    """Ignores its input and returns one learned `d_model` vector per occurrence.

    The degenerate case, and the reason it is worth having: it covers "add a new learned
    token embedding" with no vocab surgery at all. It is also the control for any
    experiment asking whether the *value* matters -- if a `ConstantEncoder` does as well,
    the model was reading the marker's presence, not what it carried.
    """

    def __init__(self, n_features: int, d_model: int, zero_init: bool = True):
        super().__init__(n_features, d_model)
        self.vector = nn.Parameter(torch.zeros(d_model) if zero_init
                                   else torch.randn(d_model) * 0.02)

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        return self.vector.unsqueeze(0).expand(values.shape[0], -1)


@register_encoder("mlp")
class MLPEncoder(ModalityEncoder):
    """A plain MLP over an arbitrary `F`. The general-purpose default.

    The final `LayerNorm` sits *before* the zero-initialised output projection rather than
    after it, which gets both properties at once: the pre-activations stay bounded as
    training grows the output (nothing constrains an injected vector's norm relative to
    ordinary token embeddings otherwise, and it can drown them out or vanish against
    them), and the output is still exactly zero at step 0.
    """

    def __init__(self, n_features: int, d_model: int,
                 hidden_dims: Sequence[int] = (256, 256),
                 dropout: float = 0.0, layer_norm: bool = True,
                 zero_init: bool = True):
        super().__init__(n_features, d_model)
        dims = [self.n_features, *[int(h) for h in hidden_dims]]
        layers: List[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.GELU()]
            if dropout:
                layers.append(nn.Dropout(dropout))
        if layer_norm:
            layers.append(nn.LayerNorm(dims[-1]))
        out = nn.Linear(dims[-1], d_model)
        layers.append(_zero_init(out) if zero_init else out)
        self.net = nn.Sequential(*layers)

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values)


@register_encoder("bucket")
class BucketEncoder(ModalityEncoder):
    """`nn.Embedding` over per-feature discretised values, summed across features.

    This makes "coarse symbolic tokens" and "a continuous embedding" two encoder choices
    under one mechanism rather than two competing architectures. Each feature gets its own
    block of `n_buckets` rows, so feature 0's bucket 3 and feature 1's bucket 3 are
    distinct embeddings.

    `lo`/`hi` are stored as **buffers**, not config: they travel with the checkpoint, so
    training and inference cannot disagree about where the bucket edges are. Values
    outside the range clamp into the end buckets.
    """

    def __init__(self, n_features: int, d_model: int, n_buckets: int = 32,
                 lo: Any = -1.0, hi: Any = 1.0, zero_init: bool = True):
        super().__init__(n_features, d_model)
        if int(n_buckets) < 2:
            raise ValueError(f"n_buckets must be >= 2, got {n_buckets}")
        self.n_buckets = int(n_buckets)
        lo_t = torch.as_tensor(lo, dtype=torch.float32).expand(self.n_features).contiguous()
        hi_t = torch.as_tensor(hi, dtype=torch.float32).expand(self.n_features).contiguous()
        if not bool((hi_t > lo_t).all()):
            raise ValueError(f"BucketEncoder needs hi > lo elementwise, got {lo}..{hi}")
        self.register_buffer("lo", lo_t)
        self.register_buffer("hi", hi_t)
        # Offset feature f into its own block of rows.
        self.register_buffer(
            "feature_offset",
            torch.arange(self.n_features, dtype=torch.long) * self.n_buckets,
            persistent=False,
        )
        self.table = nn.Embedding(self.n_features * self.n_buckets, d_model)
        if zero_init:
            nn.init.zeros_(self.table.weight)

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        frac = (values - self.lo) / (self.hi - self.lo)
        idx = (frac * self.n_buckets).floor().long().clamp(0, self.n_buckets - 1)
        return self.table(idx + self.feature_offset).sum(dim=1)


# ======================================================================================
# Batch transport
# ======================================================================================
@dataclass
class ModalityBatch:
    """The modality tensors for one batch, regrouped from the flat dataloader keys.

    Flat `modality_*` keys on the wire, a typed object inside. Transport stays a flat dict
    of tensors, which HF `Trainer._prepare_inputs` and the DDP path move to the device
    predictably; nested dicts are less reliable for that. `pop_from` regroups at exactly
    one boundary, so one function knows the naming convention.
    """

    values: Dict[str, torch.Tensor] = field(default_factory=dict)
    counts: Dict[str, torch.Tensor] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.values)

    @property
    def keys(self) -> List[str]:
        return sorted(self.values)

    @property
    def total_rows(self) -> int:
        """Occurrences across every key and every example. Zero is legal -- an example may
        simply contain no markers -- and is distinct from "no modality keys at all"."""
        return int(sum(v.shape[0] for v in self.values.values()))

    @classmethod
    def pop_from(cls, kwargs: Dict[str, Any],
                 known_keys: Optional[Iterable[str]] = None) -> "ModalityBatch":
        """Strip every `modality_*` key out of `kwargs` in place and return them typed.

        This owns the whole prefix. `TurnVectorRegressor.forward` forwards its leftover
        kwargs wholesale to a backbone that does not accept modality tensors, so anything
        left behind is a `TypeError` deep inside the model -- and anything silently
        dropped is a scatter that never happens. Both are avoided by making an
        unrecognised `modality_*` key a loud error here.
        """
        found: Dict[str, Dict[str, torch.Tensor]] = {}
        for name in [k for k in kwargs if k.startswith(MODALITY_PREFIX)]:
            tensor = kwargs.pop(name)
            body = name[len(MODALITY_PREFIX):]
            for suffix, slot in ((VALUES_SUFFIX, "values"), (COUNTS_SUFFIX, "counts")):
                if body.endswith(suffix) and len(body) > len(suffix):
                    found.setdefault(body[: -len(suffix)], {})[slot] = tensor
                    break
            else:
                raise ValueError(
                    f"unrecognised modality key {name!r}: the '{MODALITY_PREFIX}' prefix is "
                    f"owned by ModalityBatch and every key must end in '{VALUES_SUFFIX}' or "
                    f"'{COUNTS_SUFFIX}'. Rename it or it would reach the backbone."
                )

        known = None if known_keys is None else set(known_keys)
        batch = cls()
        for key, parts in found.items():
            if known is not None and key not in known:
                raise ValueError(
                    f"modality key {key!r} arrived in the batch but no spec declares it "
                    f"(declared: {sorted(known) or 'none'})"
                )
            missing = {"values", "counts"} - set(parts)
            if missing:
                raise ValueError(
                    f"modality {key!r} is missing {sorted(missing)}; values and counts "
                    "always travel together"
                )
            values, counts = parts["values"], parts["counts"]
            if values.dim() != 2:
                raise ValueError(f"modality {key!r} values must be (sum N, F), got "
                                 f"{tuple(values.shape)}")
            if counts.dim() != 1:
                raise ValueError(f"modality {key!r} counts must be (B,), got "
                                 f"{tuple(counts.shape)}")
            total = int(counts.sum())
            if total != values.shape[0]:
                raise ValueError(
                    f"modality {key!r}: counts sum to {total} but there are "
                    f"{values.shape[0]} value row(s)"
                )
            batch.values[key] = values
            batch.counts[key] = counts
        return batch

    def to_kwargs(self) -> Dict[str, torch.Tensor]:
        """Inverse of `pop_from`: the flat dataloader keys."""
        out: Dict[str, torch.Tensor] = {}
        for key in self.values:
            out[f"{MODALITY_PREFIX}{key}{VALUES_SUFFIX}"] = self.values[key]
            out[f"{MODALITY_PREFIX}{key}{COUNTS_SUFFIX}"] = self.counts[key]
        return out

    def to(self, device) -> "ModalityBatch":
        return ModalityBatch(
            values={k: v.to(device) for k, v in self.values.items()},
            counts={k: v.to(device) for k, v in self.counts.items()},
        )

    def concat(self, other: Optional["ModalityBatch"]) -> "ModalityBatch":
        """Append `other`'s occurrences after this one's, per key, for B == 1.

        Used by the rollout, where the prologue's occurrences precede the first turn's in
        the token sequence and so must precede them in the value rows too.
        """
        if other is None or not other:
            return self
        if not self:
            return other
        out = ModalityBatch()
        for key in set(self.values) | set(other.values):
            a_v = self.values.get(key)
            b_v = other.values.get(key)
            if a_v is None:
                out.values[key], out.counts[key] = b_v, other.counts[key]
                continue
            if b_v is None:
                out.values[key], out.counts[key] = a_v, self.counts[key]
                continue
            if a_v.shape[0] and b_v.shape[0] and a_v.shape[1] != b_v.shape[1]:
                raise ValueError(f"modality {key!r}: cannot concat F={a_v.shape[1]} with "
                                 f"F={b_v.shape[1]}")
            out.values[key] = torch.cat([a_v, b_v.to(a_v.dtype)], dim=0)
            out.counts[key] = self.counts[key] + other.counts[key].to(self.counts[key].device)
        return out


def resolve_token_ids(tokenizer, specs: Iterable[ModalityEmbedSpec]) -> Dict[str, int]:
    """`{key: id}`, asserting each marker resolves to exactly one id.

    Checked here rather than at first use. If a checkpoint's `added_tokens.json` lacks the
    marker then `<x>` BPEs back into several ordinary tokens, the scatter finds zero
    positions, and the failure surfaces much later as a confusing count mismatch.
    """
    ids: Dict[str, int] = {}
    for s in specs:
        encoded = tokenizer.encode(s.token, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(
                f"modality token {s.token!r} tokenizes to {len(encoded)} ids {encoded} "
                f"({[tokenizer.decode([i]) for i in encoded]}), not 1. The tokenizer does "
                "not have it registered -- register it before use, and make sure the "
                "checkpoint's added_tokens.json travels with it."
            )
        ids[s.key] = int(encoded[0])
    if len(set(ids.values())) != len(ids):
        raise ValueError(f"modality tokens collide on ids: {ids}")
    return ids


def attach_modalities(processor, specs: Optional[Iterable[Any]]) -> Dict[str, int]:
    """Register a checkpoint's marker tokens on an independently built processor.

    The processor is *derived*, never separately persisted: the spec list in the model
    config is the single source of truth. If the processor serialised its own token list
    the string would live in two files and could drift. Use this from anywhere that builds
    a processor by hand (an eval harness, a notebook) so it matches the checkpoint.
    """
    specs = coerce_specs(specs)
    if not specs:
        return {}
    processor.tokenizer.add_special_tokens(
        {"additional_special_tokens": [s.token for s in specs]}
    )
    return resolve_token_ids(processor.tokenizer, specs)


def single_example_batch(values_by_key: Mapping[str, Any]) -> ModalityBatch:
    """Build a B == 1 `ModalityBatch` from `{key: (N, F) array-like}`. For the rollout."""
    batch = ModalityBatch()
    for key, raw in values_by_key.items():
        v = torch.as_tensor(raw, dtype=torch.float32)
        if v.dim() == 1:
            v = v.unsqueeze(0)
        if v.dim() != 2:
            raise ValueError(f"modality {key!r} values must be (N, F), got {tuple(v.shape)}")
        batch.values[key] = v
        batch.counts[key] = torch.tensor([v.shape[0]], dtype=torch.long)
    return batch


# ======================================================================================
# The embedder
# ======================================================================================
class ModalityEmbedder(nn.Module):
    """Registration, position lookup, scatter, and the assertions that tie them together.

    Holds the encoders (so its parameters are the model's parameters, hence DDP-visible
    and optimizer-visible) and installs the two hooks on the backbone's input embedding.

    With `specs == ()` every method returns immediately, `attach` installs nothing, and
    `pending` is a no-op context manager.
    """

    def __init__(self, specs: Optional[Iterable[Any]] = None, d_model: Optional[int] = None):
        super().__init__()
        self.specs = coerce_specs(specs)
        self.d_model = int(d_model) if d_model is not None else None
        if self.specs and self.d_model is None:
            raise ValueError("d_model is required when specs are declared")
        self.encoders = nn.ModuleDict(
            {s.key: build_encoder(s, self.d_model) for s in self.specs}
        )
        # token id per key; filled by `bind_tokenizer`.
        self.token_ids: Dict[str, int] = {}
        self._handles: List[Any] = []
        self._pending: Optional[ModalityBatch] = None
        self._recorded: Optional[Dict[str, torch.Tensor]] = None
        self._consumed: bool = False

    # -- introspection ------------------------------------------------------------------
    def __bool__(self) -> bool:
        return bool(self.specs)

    @property
    def keys(self) -> List[str]:
        return [s.key for s in self.specs]

    def spec(self, key: str) -> ModalityEmbedSpec:
        for s in self.specs:
            if s.key == key:
                return s
        raise KeyError(f"no modality spec for key {key!r}; have {self.keys}")

    def specs_as_dicts(self) -> List[Dict[str, Any]]:
        return [s.to_dict() for s in self.specs]

    def describe(self) -> str:
        if not self.specs:
            return "no modality specs (mechanism inert)"
        return "\n".join(
            f"  {s.token} key={s.key} F={s.n_features} encoder={s.encoder} "
            f"column={s.source_column} id={self.token_ids.get(s.key, '?')}"
            for s in self.specs
        )

    # -- tokenizer ----------------------------------------------------------------------
    def register(self, tokenizer, corpus: Optional[Iterable[str]] = None) -> int:
        """Add the marker tokens to `tokenizer` and resolve their ids. Returns how many
        were newly added.

        Added tokens are applied as a pre-tokenization split over raw text, before BPE, so
        on text that does not contain the literal the split never fires and BPE sees
        byte-identical input. That is only true while the literal is genuinely absent,
        which is why `corpus` exists -- pass whatever text will be tokenized and it is
        checked rather than assumed.
        """
        if not self.specs:
            return 0
        if corpus is not None:
            self.check_corpus(corpus)
        added = tokenizer.add_special_tokens(
            {"additional_special_tokens": [s.token for s in self.specs]}
        )
        self.bind_tokenizer(tokenizer)
        return int(added)

    def bind_tokenizer(self, tokenizer) -> "ModalityEmbedder":
        """Resolve each spec's token to exactly one id, or raise. Checked at load."""
        if self.specs:
            self.token_ids = resolve_token_ids(tokenizer, self.specs)
        return self

    def check_corpus(self, texts: Iterable[str]) -> None:
        """Assert no marker literal already occurs in the text that will be tokenized."""
        literals = {s.token for s in self.specs}
        for i, text in enumerate(texts):
            if not isinstance(text, str):
                continue
            for lit in literals:
                if lit in text:
                    raise ValueError(
                        f"marker literal {lit!r} already occurs in the corpus (item {i}). "
                        "Adding it as a token would change how that text tokenizes; pick "
                        "a different marker."
                    )

    def check_placement(self, input_ids: torch.Tensor, vision_start_token_id: int) -> None:
        """No marker directly after `<|vision_start|>`.

        `get_rope_index` reads exactly that position to decide image versus video. A
        marker there makes the span count as neither, and every position after it is
        silently wrong -- no exception, no NaN, just a different sequence than training.
        """
        if not self.token_ids:
            return
        marker_ids = set(self.token_ids.values())
        starts = (input_ids == vision_start_token_id).nonzero(as_tuple=False)
        for b, s in starts.tolist():
            if s + 1 < input_ids.shape[1] and int(input_ids[b, s + 1]) in marker_ids:
                raise ValueError(
                    f"a modality marker sits immediately after <|vision_start|> at "
                    f"position {s + 1} (example {b}). get_rope_index derives the image/"
                    "video count from that position; put the marker anywhere else."
                )

    # -- masks --------------------------------------------------------------------------
    def masks(self, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """`{key: (B, S) bool}` for every declared modality present in `input_ids`."""
        return {key: input_ids == tid for key, tid in self.token_ids.items()}

    # -- hooks --------------------------------------------------------------------------
    def attach(self, embed_module: nn.Module) -> "ModalityEmbedder":
        """Install the pre/post hooks on the backbone's input embedding. Idempotent."""
        self.detach()
        if not self.specs:
            return self
        if not self.token_ids:
            raise RuntimeError(
                "attach() before bind_tokenizer(): the hooks need the marker ids"
            )
        self._handles = [
            embed_module.register_forward_pre_hook(self._pre_hook),
            embed_module.register_forward_hook(self._post_hook),
        ]
        return self

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    @contextlib.contextmanager
    def pending(self, batch: Optional[ModalityBatch]):
        """Make `batch` available to the hooks for exactly one embedding call.

        Wrap only the backbone call. If the backbone were entered twice under one context
        the consume-once assert fires, rather than the second pass silently reusing the
        first's values.
        """
        if not self.specs:
            yield
            return
        if self._pending is not None:
            raise RuntimeError("ModalityEmbedder.pending() is already active (re-entered)")
        self._pending = batch if batch is not None else ModalityBatch()
        self._recorded = None
        self._consumed = False
        body_ok = False
        try:
            yield
            body_ok = True
        finally:
            had_rows = self._pending is not None and self._pending.total_rows > 0
            consumed = self._consumed
            self._pending = None
            self._recorded = None
            self._consumed = False
        # Outside the finally, and only when the body itself succeeded: raising in a
        # finally would mask whatever real error the caller was already propagating.
        if body_ok and had_rows and not consumed:
            raise RuntimeError(
                "modality values were provided but nothing was injected: either the "
                "backbone's input embedding was never called under pending() (check what "
                "attach() targeted) or the sequence contains no marker occurrences."
            )

    def _pre_hook(self, module, args):
        """Rewrite marker ids in `embed_tokens`' argument to a safe row, and record where.

        `input_ids` itself is untouched: it continues to `get_rope_index` and
        `get_placeholder_mask` unchanged, which is exactly what keeps a marker behaving
        like an ordinary text token everywhere else in the model.
        """
        if not args or not torch.is_tensor(args[0]) or args[0].dtype.is_floating_point:
            return None
        input_ids = args[0]
        if input_ids.dim() != 2:
            return None
        masks = {k: m for k, m in self.masks(input_ids).items() if bool(m.any())}
        if not masks:
            self._recorded = {} if self._pending is not None else None
            return None

        if self._pending is None:
            raise RuntimeError(
                f"marker token(s) {sorted(masks)} are in the sequence but no modality "
                "values are pending. Nothing would be injected and the model would read a "
                "meaningless embedding. Pass modality=... to this call "
                "(a marker in the system prompt needs reset(modality=...), not step())."
            )
        if self._consumed:
            raise RuntimeError(
                "the input embedding fired twice under one pending() context. The values "
                "were already consumed, so this pass would reuse them silently."
            )

        safe = input_ids.clone()
        for m in masks.values():
            safe[m] = SAFE_EMBED_ID
        self._recorded = masks
        return (safe,) + tuple(args[1:])

    def _post_hook(self, module, args, output):
        """Scatter `encoder(values)` at the recorded positions, and check it consumed
        exactly those rows."""
        recorded, self._recorded = self._recorded, None
        if not recorded:
            return None
        pending = self._pending
        if pending is None:  # pragma: no cover - the pre-hook already raised
            raise RuntimeError("post-hook fired with nothing pending")

        missing = sorted(set(recorded) - set(pending.values))
        if missing:
            raise RuntimeError(
                f"marker token(s) {missing} occur in the sequence but no values were "
                f"provided for them (provided: {pending.keys or 'none'})"
            )
        # A key with zero rows and zero occurrences is fine (an example may carry no
        # markers); a key with rows but no occurrences would silently drop them.
        extra = sorted(k for k in pending.values
                       if k not in recorded and pending.values[k].shape[0] > 0)
        if extra:
            raise RuntimeError(
                f"values were provided for {extra} but those marker token(s) do not occur "
                "in this sequence, so they would be silently dropped"
            )

        embeds = output
        for key, mask in recorded.items():
            counts = pending.counts[key].to(mask.device)
            found = mask.sum(dim=1)
            # Per example, never the total: a total-only check passes when one example has
            # an extra occurrence and another is short by one, which is exactly the
            # misalignment that trains plausibly and means nothing.
            if counts.shape != found.shape or not bool((counts == found).all()):
                raise RuntimeError(
                    f"modality {key!r} count mismatch per example: values say "
                    f"{counts.tolist()}, the sequence has {found.tolist()}"
                )
            values = pending.values[key].to(mask.device)
            src = self.encoders[key](values, dtype=embeds.dtype)
            if src.shape[0] != int(found.sum()):
                raise RuntimeError(
                    f"modality {key!r}: encoder produced {src.shape[0]} row(s) for "
                    f"{int(found.sum())} occurrence(s)"
                )
            # masked_scatter enumerates True positions row-major over (b, position), which
            # IS occurrence order concatenated over the batch -- no reshape, no valid mask.
            embeds = embeds.masked_scatter(mask.unsqueeze(-1), src)
        self._consumed = True
        return embeds

    # -- DDP ----------------------------------------------------------------------------
    def zero_touch(self, device, dtype) -> Optional[torch.Tensor]:
        """A zero-valued scalar that connects every encoder parameter to the graph.

        An example with no occurrences of a modality gives its encoder no gradient, which
        under `ddp_find_unused_parameters=False` hangs or errors -- reachable as soon as
        markers are not on every turn, or as soon as windowing can cut them out. Adding
        this to the loss keeps the parameter in the backward graph with an exactly-zero
        contribution, so the numbers do not move.
        """
        if not self.specs:
            return None
        total = None
        for key in self.encoders:
            probe = torch.zeros(1, self.spec(key).n_features, device=device)
            out = self.encoders[key](probe).sum()
            total = out if total is None else total + out
        return None if total is None else (total * 0.0).to(dtype)

    # -- checkpointing ------------------------------------------------------------------
    def state_blob(self) -> Optional[Dict[str, Any]]:
        """What goes into `turn_vector_head.pt`. None when nothing is declared, so a
        no-spec checkpoint is byte-identical to one written before this existed."""
        if not self.specs:
            return None
        return {"encoders": self.encoders.state_dict()}

    def load_state_blob(self, blob: Optional[Mapping[str, Any]]) -> None:
        """Load encoder weights. Strict in both directions, regardless of any caller's
        `strict` flag: that flag exists to tolerate *head* shape evolution, and letting it
        also drop encoders would mean a model behaving differently from what its config
        says.

        The four cases from DESIGN.md 7.1:
          1. config declares a spec, blob has no weights  -> raise
          2. blob has weights for an undeclared spec      -> raise
          3. shape drift                                  -> raise (strict=True)
          4. config declares nothing                      -> silent no-op (legacy path)
        """
        if not self.specs:
            if blob:
                raise RuntimeError(
                    "checkpoint carries modality encoder weights but the config declares "
                    "no specs; loading it would give a model that behaves differently "
                    "from what its config claims"
                )
            return
        if not blob or "encoders" not in blob:
            raise RuntimeError(
                f"config declares modality spec(s) {self.keys} but the checkpoint has no "
                "encoder weights -- mismatched checkpoint"
            )
        state = blob["encoders"]
        have = {k.split(".", 1)[0] for k in state}
        want = set(self.encoders)
        if have - want:
            raise RuntimeError(
                f"checkpoint has modality encoder weights for {sorted(have - want)}, which "
                "the config does not declare"
            )
        if want - have:
            raise RuntimeError(
                f"config declares modality spec(s) {sorted(want - have)} with no weights "
                "in the checkpoint"
            )
        self.encoders.load_state_dict(state, strict=True)
