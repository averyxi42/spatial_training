"""Head-only training on a frozen context: the store, the model, the trainer, the probes.

The question this exists to answer
----------------------------------
`dump/cnf_sigma_ablation/FINDINGS.md` diagnosed the CNF head's failures as **context
collapse** -- the conditioning vector going near-constant (93% of every row the dataset
mean, ~2 effective dimensions) -- and not as a fault of the flow, the dequantisation
sigma or the warm start. On cached (context, chunk) pairs from a trunk known to be
informative, the same flow reached 91-98% of a ridge regression's R^2 on the same split.

That result was produced by an ad-hoc harness. This module is the same experiment made
first-class, so the two questions

    can the head learn?          (head on a frozen, known-good context)
    is the trunk giving it any?  (the same head, end to end)

can be asked with the *same code* and answered by comparing the *same metric names*. A
frozen context turns a ~4 s/step VLM run into a ~100 steps/s one, so a head change costs
minutes instead of a day.

What is here
------------
`HarvestProvenance` / `ContextStore`
    The on-disk artifact and its provenance. A frozen-context dataset whose source trunk
    is unknown is worthless -- the trunk is the independent variable -- so the manifest
    records the checkpoint path, its step, a content hash of its trainable weights, the
    corpus, the modality spec and the chunk shape, and `ContextStore.open` is the only
    reader.

`FrozenContextFlowPolicy`
    The head, and only the head: `TurnVectorHead.project` + `FlowActionDecoder`. Its
    objective is `flow_head.FlowObjectiveMixin.flow_objective`, imported, not
    reimplemented.

`HeadOnlyFlowTrainer`
    `FlowSFTTrainer` with the batch-size rule relaxed and the collapse probes attached.
    Every metric name comes from the inherited `_drain_metrics`, which is why they match.

`context_collapse_stats` / `coupling_context_ratio`
    The diagnostics FINDINGS identified, as pure functions of arrays -- testable, and
    logged per eval.

What this pipeline cannot tell you is in `docs/HEAD_ONLY_TRAINING.md`. The short form:
a head that trains well here says **nothing** about whether the trunk would supply such a
context under joint training. The context used here was made by a *different* objective.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from longnav.cnf_head.flow import ConditionalFlow
from longnav.utils.flow_head import (
    FLOW_CONFIG_FILE,
    FlowActionDecoder,
    FlowObjectiveMixin,
    FlowSFTTrainer,
)
from longnav.utils.turn_vectors import TurnVectorHead
from longnav.utils.vector_sft import HEAD_CONFIG_FILE, HEAD_WEIGHTS_FILE, ModelConfig

MANIFEST_FILE = "manifest.json"
STORE_VERSION = 1

# The three arrays every row carries, and the one optional one. Fixed-width float
# matrices, which is why they are memmapped `.npy` and not anything cleverer -- see
# `ContextStore` for the storage rationale.
ARRAY_SPECS = ("pooled", "context", "targets", "pose")


# =======================================================================================
# Provenance
# =======================================================================================
def hash_checkpoint(checkpoint_dir: os.PathLike | str) -> str:
    """A content hash of everything that makes the trunk what it is.

    Covers the head weights blob (head + normalizer + modality encoders) and the LoRA
    adapter, because between them they are the entire difference from the stock base
    model. Deliberately NOT the whole directory: `optimizer.pt`, `rng_state_*.pth` and
    `trainer_state.json` change with resume bookkeeping that does not change a single
    forward pass, and hashing them would make a legitimate reattachment fail.

    Read in 1 MiB blocks so a multi-hundred-MB adapter does not have to be resident.
    """
    checkpoint_dir = Path(checkpoint_dir)
    h = hashlib.sha256()
    names = [HEAD_WEIGHTS_FILE, HEAD_CONFIG_FILE, FLOW_CONFIG_FILE,
             "adapter/adapter_model.safetensors", "adapter/adapter_config.json"]
    for name in names:
        p = checkpoint_dir / name
        if not p.exists():
            continue
        h.update(name.encode())
        with p.open("rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
    return h.hexdigest()


@dataclass
class HarvestProvenance:
    """Everything needed to say which trunk produced this context, and on what.

    Written into the manifest and checked by `reattach_head.py`. The fields are not
    decoration: `source_hash` is what refuses a head trained on trunk A being bolted onto
    trunk B, and `context_dim` / `pooled_dim` / `chunk_shape` are what stop a head being
    built at the wrong width against a store that would broadcast rather than raise.
    """

    source_checkpoint: str
    source_step: Optional[int]
    source_hash: str
    source_target_shape: List[int]
    corpus: str
    split: str
    pooled_dim: int
    context_dim: int
    chunk_shape: List[int]
    modality_specs: List[Dict[str, Any]] = field(default_factory=list)
    pool_mode: str = "mean"
    head_hidden_dims: List[int] = field(default_factory=list)
    max_turns_per_sample: Optional[int] = None
    target_column: str = "action_chunks"
    obs_hz: Optional[float] = None
    dtype: str = "float32"
    store_version: int = STORE_VERSION
    notes: str = ""

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, blob: Dict[str, Any]) -> "HarvestProvenance":
        known = {f for f in cls.__dataclass_fields__}
        # Unknown keys are dropped rather than fatal, so a store written by a later
        # version still opens here; the version number is what a reader should gate on.
        return cls(**{k: v for k, v in blob.items() if k in known})

    def assert_matches_checkpoint(self, checkpoint_dir: os.PathLike | str) -> None:
        """Hard-error unless `checkpoint_dir` is the trunk this context came from.

        A head trained on one trunk's context and reattached to another loads cleanly,
        runs, and is meaningless -- there is no observable that would go wrong. So the
        check is here, it names both sides, and it is not a warning.
        """
        got = hash_checkpoint(checkpoint_dir)
        if got == self.source_hash:
            return
        raise RuntimeError(
            "refusing to reattach: this head was trained on context harvested from\n"
            f"    {self.source_checkpoint}\n"
            f"    sha256 {self.source_hash}\n"
            "but the trunk offered is\n"
            f"    {Path(checkpoint_dir)}\n"
            f"    sha256 {got}\n"
            "A head is only meaningful on the trunk whose representation it was fitted "
            "to. Nothing downstream would notice this, which is why it stops here."
        )


# =======================================================================================
# Storage
# =======================================================================================
@dataclass
class ShardMeta:
    """One shard's manifest entry. Written last, so a shard with no entry is incomplete."""

    index: int
    rows: int
    episodes: List[str]
    episode_rows: List[int]

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)


class ContextStore:
    """A sharded, memory-mapped frozen-context dataset.

    Storage rationale, since the brief asks for one
    ----------------------------------------------
    At the full corpus (~39k episodes x ~50 observations) this is ~2M rows. Per row:
    `pooled` 2048 floats, `context` 1024 floats, `targets` 20x3, `pose` 3. In float32
    that is 12.6 KB/row, ~25 GB total, of which `pooled` is two thirds.

    That rules out the prior work's `.npz`: it is a zip container, so it cannot be
    memory-mapped and every read decompresses whole arrays into RAM. It also rules out a
    HuggingFace/Arrow dataset, which would store each 2048-float row as a variable-length
    list with offset overhead and rebuild a numpy array per access -- correct, but the
    wrong shape for a fixed-width float matrix that will be randomly indexed a few
    hundred thousand times per training run.

    So: plain `.npy` per array per shard, opened with `mmap_mode="r"`. Self-describing
    dtype and shape, zero-copy random access, no dependency beyond numpy, and the page
    cache does the caching. Sharding buys resumability -- each shard is written to a
    temporary name and renamed, and its manifest entry is appended only once the arrays
    are on disk, so a crash costs at most one shard and never leaves a half-row.

    Measured: see `docs/HEAD_ONLY_TRAINING.md` for the pilot's on-disk size.
    """

    def __init__(self, root: os.PathLike | str):
        self.root = Path(root)
        self.manifest_path = self.root / MANIFEST_FILE

    # -- writing -----------------------------------------------------------------------
    def init(self, provenance: HarvestProvenance) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        if self.manifest_path.exists():
            existing = self.read_manifest()
            old, new = existing["provenance"], provenance.to_json()
            differing = [k for k in new if old.get(k) != new[k] and k != "notes"]
            if differing:
                raise RuntimeError(
                    f"{self.root} already holds a harvest from a different configuration; "
                    f"these fields differ: {differing}. Appending would mix two trunks' "
                    "context into one dataset, and nothing downstream could tell. Use a "
                    "fresh --out directory."
                )
            return
        self._write_manifest({"provenance": provenance.to_json(), "shards": []})

    def _write_manifest(self, blob: Dict[str, Any]) -> None:
        tmp = self.manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(blob, indent=2))
        tmp.replace(self.manifest_path)      # atomic: a reader never sees a partial file

    def read_manifest(self) -> Dict[str, Any]:
        return json.loads(self.manifest_path.read_text())

    def write_shard(self, arrays: Dict[str, np.ndarray], episodes: Sequence[str],
                    episode_rows: Sequence[int]) -> ShardMeta:
        """Append one shard. Arrays first, manifest entry last."""
        blob = self.read_manifest()
        index = max((s["index"] for s in blob["shards"]), default=-1) + 1
        rows = len(next(iter(arrays.values())))
        for name, arr in arrays.items():
            if len(arr) != rows:
                raise ValueError(f"array {name!r} has {len(arr)} rows, expected {rows}")
            # The temporary name still ends in `.npy`: `np.save` appends that suffix
            # itself when the path lacks it, so a `.npy.tmp` name would be written to
            # `.npy.tmp.npy` and the rename would miss it.
            tmp = self.root / f"_writing.shard_{index:05d}.{name}.npy"
            np.save(tmp, np.ascontiguousarray(arr))
            tmp.replace(self.root / f"shard_{index:05d}.{name}.npy")
        meta = ShardMeta(index=index, rows=rows, episodes=list(episodes),
                         episode_rows=list(episode_rows))
        blob["shards"].append(meta.to_json())
        self._write_manifest(blob)
        return meta

    def done_episodes(self) -> set:
        """Episode ids already harvested. The resume key -- 39k episodes is not redoable."""
        if not self.manifest_path.exists():
            return set()
        out = set()
        for s in self.read_manifest()["shards"]:
            out.update(s["episodes"])
        return out

    # -- reading -----------------------------------------------------------------------
    @classmethod
    def open(cls, root: os.PathLike | str) -> "OpenedContextStore":
        store = cls(root)
        if not store.manifest_path.exists():
            raise FileNotFoundError(
                f"{store.manifest_path} does not exist -- {root} is not a context store "
                "(run data_scripts/harvest_context.py first)"
            )
        blob = store.read_manifest()
        prov = HarvestProvenance.from_json(blob["provenance"])
        shards = sorted(blob["shards"], key=lambda s: s["index"])
        if not shards:
            raise RuntimeError(f"{root} holds a manifest but no shards")
        names = [n for n in ARRAY_SPECS
                 if (store.root / f"shard_{shards[0]['index']:05d}.{n}.npy").exists()]
        arrays = {
            n: [np.load(store.root / f"shard_{s['index']:05d}.{n}.npy", mmap_mode="r")
                for s in shards]
            for n in names
        }
        episode_ids: List[str] = []
        for s in shards:
            for ep, k in zip(s["episodes"], s["episode_rows"]):
                episode_ids.extend([ep] * int(k))
        return OpenedContextStore(root=store.root, provenance=prov, arrays=arrays,
                                  episode_ids=np.asarray(episode_ids), shards=shards)

    @classmethod
    def open_many(cls, roots: Sequence[os.PathLike | str]) -> "OpenedContextStore":
        """Concatenate several stores harvested from the *same* trunk in parallel.

        Harvesting is one forward pass per episode and embarrassingly parallel, but a
        single manifest written by four processes would race. So each worker takes a
        disjoint episode range into its own directory and the parts are joined here,
        after checking that they describe the same trunk and the same corpus -- mixing
        two trunks' context into one training set is the one mistake this whole pipeline
        exists to prevent, and it would not be visible in any downstream number.
        """
        parts = [cls.open(r) for r in roots]
        if len(parts) == 1:
            return parts[0]
        first = parts[0].provenance
        for p in parts[1:]:
            for field_name in ("source_hash", "corpus", "split", "pooled_dim",
                               "context_dim", "chunk_shape", "max_turns_per_sample"):
                if getattr(p.provenance, field_name) != getattr(first, field_name):
                    raise RuntimeError(
                        f"refusing to join {p.root} to {parts[0].root}: {field_name} "
                        f"differs ({getattr(p.provenance, field_name)!r} vs "
                        f"{getattr(first, field_name)!r})"
                    )
        names = sorted(set.intersection(*(set(p.arrays) for p in parts)))
        overlap = set()
        seen = set()
        for p in parts:
            ids = set(p.episode_ids.tolist())
            overlap |= seen & ids
            seen |= ids
        if overlap:
            raise RuntimeError(
                f"the parts share {len(overlap)} episode id(s) (e.g. "
                f"{sorted(overlap)[:5]}); duplicated rows would leak across the "
                "episode split and flatter every held-out number"
            )
        return OpenedContextStore(
            root=parts[0].root,
            provenance=first,
            arrays={n: [a for p in parts for a in p.arrays[n]] for n in names},
            episode_ids=np.concatenate([p.episode_ids for p in parts]),
            shards=[s for p in parts for s in p.shards],
        )


@dataclass
class OpenedContextStore:
    """A read-only view over the shards, concatenated logically but not in memory."""

    root: Path
    provenance: HarvestProvenance
    arrays: Dict[str, List[np.ndarray]]
    episode_ids: np.ndarray
    shards: List[Dict[str, Any]]

    def __post_init__(self):
        self._offsets = np.cumsum([0] + [len(a) for a in next(iter(self.arrays.values()))])

    def __len__(self) -> int:
        return int(self._offsets[-1])

    @property
    def n_episodes(self) -> int:
        return len(set(self.episode_ids.tolist()))

    def get(self, name: str, rows: np.ndarray) -> np.ndarray:
        """Gather `rows` (global indices) from array `name`, across shards."""
        rows = np.asarray(rows)
        shard_of = np.searchsorted(self._offsets, rows, side="right") - 1
        out = None
        for s in np.unique(shard_of):
            sel = shard_of == s
            block = np.asarray(self.arrays[name][s][rows[sel] - self._offsets[s]])
            if out is None:
                out = np.empty((len(rows), *block.shape[1:]), dtype=block.dtype)
            out[sel] = block
        return out

    def whole(self, name: str) -> np.ndarray:
        """The full array, materialised. Only for diagnostics on a modest split."""
        return np.concatenate([np.asarray(a) for a in self.arrays[name]])

    def split_by_episode(self, val_frac: float = 0.1, test_frac: float = 0.0,
                         seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Row indices for train/val/test, split by EPISODE.

        By episode and never by row: neighbouring observations of one episode are
        near-duplicates, so a random row split leaks the answer across the boundary and
        every held-out number comes back flattering. `dump/cnf_sigma_ablation` made the
        same choice for the same reason.
        """
        eps = np.array(sorted(set(self.episode_ids.tolist())))
        rng = np.random.default_rng(seed)
        rng.shuffle(eps)
        n_val = int(round(val_frac * len(eps)))
        n_test = int(round(test_frac * len(eps)))
        val, test = set(eps[:n_val]), set(eps[n_val:n_val + n_test])
        idx = np.arange(len(self))
        in_val = np.array([e in val for e in self.episode_ids])
        in_test = np.array([e in test for e in self.episode_ids])
        return idx[~(in_val | in_test)], idx[in_val], idx[in_test]


# =======================================================================================
# The dataset the trainer sees
# =======================================================================================
class FrozenContextDataset(torch.utils.data.Dataset):
    """One harvested observation per item. `num_turns = 1` so turn accounting is exact.

    `TurnVectorSFTTrainer._get_num_items_in_batch` sums `num_turns` over the raw items,
    which is how the loss stays a per-turn mean across accumulation and ranks. With one
    turn per item that sum is the batch size, and `turn_loss` here means exactly what
    `turn_loss` means in a full run.
    """

    def __init__(self, store: OpenedContextStore, rows: np.ndarray,
                 context_key: str = "pooled"):
        if context_key not in store.arrays:
            raise KeyError(
                f"the store has no {context_key!r} array (has {sorted(store.arrays)}). "
                "'pooled' is the frozen trunk state and trains the head; 'context' is "
                "the source head's own output and trains only the flow."
            )
        self.store, self.rows, self.context_key = store, np.asarray(rows), context_key

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        return {"row": int(self.rows[i]), "num_turns": 1}


@dataclass
class FrozenContextCollator:
    """Row indices -> `{"context": (B, D), "targets": (B, T, C), "num_turns": ...}`.

    The gather happens here rather than in `__getitem__` so a shard is touched once per
    batch instead of once per row; with `dataloader_num_workers > 0` this is the only
    part that touches the memmaps.
    """

    store: OpenedContextStore
    context_key: str = "pooled"

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        rows = np.array([e["row"] for e in examples])
        ctx = torch.from_numpy(np.ascontiguousarray(self.store.get(self.context_key, rows)))
        tgt = torch.from_numpy(np.ascontiguousarray(self.store.get("targets", rows)))
        return {
            "context": ctx.float(),
            "targets": tgt.float(),
            "num_turns": torch.tensor(len(rows), dtype=torch.long),
        }


# =======================================================================================
# The model: the head, and nothing else
# =======================================================================================
class FrozenContextFlowPolicy(FlowObjectiveMixin, nn.Module):
    """`TurnVectorHead.project` + `FlowActionDecoder`, trained on a cached trunk state.

    Exactly the trainable set of `train_flow_sft.py` minus the LoRA adapters and the
    modality encoders, both of which are trunk-side and frozen into the harvest. The
    slot names (`head`, `normalizer`) are the ones `TurnVectorRegressor.save_pretrained`
    and `load_head_state` use, so a head trained here is loadable by the ordinary path
    and `reattach_head.py` has nothing to translate.

    `context_key = "context"` degrades this to flow-only training on the source head's
    own output -- the `dump/cnf_sigma_ablation` setting -- by making the head an
    identity. Useful as a control: it isolates "can the flow use this context" from
    "can a fresh projection produce a usable one".
    """

    def __init__(
        self,
        pooled_dim: int,
        context_dim: int,
        chunk_shape: Sequence[int],
        head_hidden_dims: Sequence[int] = (1024,),
        head_dropout: float = 0.0,
        head_layer_norm: bool = True,
        flow_kwargs: Optional[Dict[str, Any]] = None,
        decode: str = "best_of_k",
        decode_k: int = 16,
        identity_head: bool = False,
    ):
        super().__init__()
        chunk_shape = tuple(int(d) for d in chunk_shape)
        if identity_head and pooled_dim != context_dim:
            raise ValueError(
                f"identity_head needs pooled_dim == context_dim, got {pooled_dim} vs "
                f"{context_dim}: with the source head frozen there is nothing to resize "
                "the vector with"
            )
        self.identity_head = bool(identity_head)
        # `hidden_size = pooled_dim` and pooling never runs: this head is only ever
        # entered through `project`, which starts at `pre_norm`. Building the real class
        # rather than an equivalent MLP is what makes the weights transfer to a real
        # checkpoint without a key rename.
        self.head = TurnVectorHead(
            hidden_size=pooled_dim,
            out_dim=context_dim,
            mode="mean",
            hidden_dims=tuple(head_hidden_dims),
            dropout=head_dropout,
            layer_norm=head_layer_norm,
        )
        flow = ConditionalFlow(
            context_dim=context_dim, chunk_len=chunk_shape[0],
            n_channels=chunk_shape[1], **(flow_kwargs or {}),
        )
        self.normalizer = FlowActionDecoder(flow, decode=decode, k=decode_k)
        self.pooled_dim = int(pooled_dim)
        self.context_dim = int(context_dim)
        self.chunk_shape = chunk_shape
        self.flow_kwargs = dict(flow_kwargs or {})
        self.target_shape = (int(context_dim),)
        self.sigma_mult = 1.0

    def encode(self, context: torch.Tensor) -> torch.Tensor:
        """Frozen trunk state -> the raw conditioning vector, i.e. what the trunk emits."""
        if self.identity_head:
            return context.float()
        return self.head.project(context).float()

    def forward(
        self,
        context: torch.Tensor,
        targets: torch.Tensor,
        num_turns: Optional[torch.Tensor] = None,
        num_items_in_batch: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        return self.flow_objective(self.encode(context), targets, num_items_in_batch)

    # -- checkpointing -----------------------------------------------------------------
    def head_state_blob(self) -> Dict[str, Any]:
        """The `turn_vector_head.pt` payload for the head half. See `reattach_head.py`.

        The keys are `TurnVectorRegressor.save_pretrained`'s, so the reattached
        checkpoint is byte-compatible with `load_head_state` and needs no special case.
        """
        return {"head": self.head.state_dict(), "normalizer": self.normalizer.state_dict()}

    def save_pretrained(self, output_dir: os.PathLike | str):
        """Head-only checkpoint: weights plus enough config to rebuild this object.

        Not a policy checkpoint -- there is no trunk here. `reattach_head.py` turns one
        of these plus its source trunk into something the ordinary loader accepts.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.head_state_blob(), output_dir / HEAD_WEIGHTS_FILE)
        (output_dir / "head_only_config.json").write_text(json.dumps({
            "pooled_dim": self.pooled_dim,
            "context_dim": self.context_dim,
            "chunk_shape": list(self.chunk_shape),
            "head_hidden_dims": [m.out_features for m in self.head.mlp
                                 if isinstance(m, nn.Linear)][:-1],
            "identity_head": self.identity_head,
            "flow": self.flow.config(),
            "decode": self.decoder.decode,
            "decode_k": self.decoder.k,
        }, indent=2))

    @classmethod
    def from_pretrained(cls, checkpoint_dir: os.PathLike | str,
                        map_location: str = "cpu") -> "FrozenContextFlowPolicy":
        checkpoint_dir = Path(checkpoint_dir)
        cfg = json.loads((checkpoint_dir / "head_only_config.json").read_text())
        flow_cfg = dict(cfg["flow"])
        for key in ("context_dim", "chunk_len", "n_channels"):
            flow_cfg.pop(key, None)
        model = cls(
            pooled_dim=cfg["pooled_dim"], context_dim=cfg["context_dim"],
            chunk_shape=cfg["chunk_shape"], head_hidden_dims=cfg["head_hidden_dims"],
            flow_kwargs=flow_cfg, decode=cfg.get("decode", "best_of_k"),
            decode_k=cfg.get("decode_k", 16), identity_head=cfg.get("identity_head", False),
        )
        blob = torch.load(checkpoint_dir / HEAD_WEIGHTS_FILE, map_location=map_location,
                          weights_only=False)
        model.head.load_state_dict(blob["head"])
        model.normalizer.load_state_dict(blob["normalizer"])
        return model


# =======================================================================================
# Loading the trunk out of a policy checkpoint
# =======================================================================================
def load_trunk(checkpoint_dir: os.PathLike | str, processor,
               dtype: torch.dtype = torch.bfloat16,
               device: Optional[str] = None) -> Any:
    """A checkpoint's **trunk**: backbone + LoRA + modality encoders + pooling head.

    Deliberately not `TurnVectorRegressor.from_pretrained`. That loads the whole policy,
    including the `normalizer` slot, and each head type puts something different in it --
    a `TargetNormalizer` for the regression head, a `FlowActionDecoder` for the CNF, a
    flow-matching decoder for the flow-matching head. Rebuilding the right one means
    knowing the head type, and the whole premise of a frozen-context dataset is that the
    context is **head-independent**: it is what the trunk emits, and the harvest should
    work off any checkpoint that has one.

    So the normalizer is skipped, by name, and what it was is reported. Everything else
    is loaded strictly -- especially the modality encoders, which are trunk-side and
    whose absence would silently change what the context means.

    Returns a `TurnVectorRegressor` in eval mode whose `head.project` is the source
    head's, so `head(states, mask)` reproduces the source policy's context vector exactly.
    """
    from longnav.utils.vector_sft import (
        ADAPTER_SUBDIR, LossConfig, TurnVectorRegressor, migrate_model_config,
    )

    checkpoint_dir = Path(checkpoint_dir)
    meta = json.loads((checkpoint_dir / HEAD_CONFIG_FILE).read_text())
    model_cfg = ModelConfig(**migrate_model_config(meta["model"]))
    # `normalize_targets=False`: the parent's build would otherwise construct a
    # normalizer that raises until it is fitted, and nothing here regresses anything.
    loss_cfg = LossConfig(**{**meta["loss"], "normalize_targets": False})
    model = TurnVectorRegressor.build(
        model_cfg, loss_cfg, None, meta["target_shape"], processor, dtype
    )
    model.train_content_len = meta.get("train_content_len")
    adapter_dir = checkpoint_dir / ADAPTER_SUBDIR
    if adapter_dir.exists():
        from peft import PeftModel

        model.backbone = PeftModel.from_pretrained(model.backbone, str(adapter_dir))
        model.attach_modality_hooks()   # the hooks live under the new wrapper
    else:
        raise RuntimeError(
            f"{checkpoint_dir} has no {ADAPTER_SUBDIR}/ -- the trunk would be the stock "
            "base model, not the trained one. Harvesting that is almost certainly a "
            "mistake, so it is refused rather than warned about."
        )
    blob = torch.load(checkpoint_dir / HEAD_WEIGHTS_FILE, map_location="cpu",
                      weights_only=False)
    model.head.load_state_dict(blob["head"], strict=True)
    model.modality_embedder.load_state_blob(blob.get("modality"))
    skipped = sorted(blob.get("normalizer", {}).keys())
    model.skipped_normalizer_keys = skipped
    if device:
        model.to(device)
    return model.eval()


# =======================================================================================
# Collapse diagnostics -- the reason this pipeline exists
# =======================================================================================
def context_collapse_stats(ctx: np.ndarray, near_mean_tol: float = 0.1) -> Dict[str, float]:
    """Is this conditioning vector carrying information, or is it the dataset mean?

    Reproduces `dump/cnf_sigma_ablation/ctx_stats.py`'s table, which is the reference for
    what healthy and collapsed look like:

        vector                       row norm  resid/row  participation  mean pairwise cos
        AR v3 ckpt2200 (healthy)        35.55      86.6%          14.54     +0.246 +- 0.200
        dead flow run, step 3000         13.88       1.9%           1.20     +0.9998 +- 0.0003

    `resid_frac` -- the mean residual norm after removing the dataset mean, over the mean
    row norm -- is the headline: 1.9% means 98% of every row is one constant vector, so
    permuting the context is nearly a no-op and every conditional measurement downstream
    is vacuous. `participation_ratio` is `(sum s^2)^2 / sum s^4` over the singular values
    of the **mean-removed** matrix, matching `ctx_stats.py` exactly, i.e. how many
    directions the surviving variation occupies out of the nominal width.

    Both are reported because neither subsumes the other. A constant vector plus isotropic
    noise has `resid_frac` ~ 0 and a full-width participation ratio: collapsed by any
    useful definition, invisible to PR alone. The dead run happened to be collapsed in
    both senses at once (1.9% and 1.20), which is why one number sufficed there and does
    not in general.
    """
    x = np.asarray(ctx, dtype=np.float64)
    if x.ndim != 2 or len(x) < 2:
        raise ValueError(f"expected a (N >= 2, D) matrix, got shape {x.shape}")
    mu = x.mean(0)
    resid = x - mu
    row_norm = float(np.linalg.norm(x, axis=1).mean())
    resid_norm = np.linalg.norm(resid, axis=1)
    # Eigenvalues of the covariance, via the residual's singular values -- no D x D
    # matrix is ever formed, which matters at D = 2048.
    sv = np.linalg.svd(resid / np.sqrt(max(1, len(x) - 1)), compute_uv=False)
    lam = sv ** 2
    denom = float((lam ** 2).sum())
    n = np.linalg.norm(x, axis=1, keepdims=True)
    unit = x / np.clip(n, 1e-12, None)
    # Mean pairwise cosine without an N x N matrix: E[<u_i, u_j>] over i != j is
    # (||sum u||^2 - N) / (N (N-1)).
    s = unit.sum(0)
    n_rows = len(x)
    mean_cos = float((float(s @ s) - n_rows) / max(1, n_rows * (n_rows - 1)))
    return {
        "ctx_row_norm": row_norm,
        "ctx_resid_frac": float(resid_norm.mean() / max(1e-12, row_norm)),
        "ctx_participation_ratio": float(lam.sum() ** 2 / denom) if denom > 0 else 0.0,
        "ctx_frac_near_mean": float((resid_norm < near_mean_tol * row_norm).mean()),
        "ctx_mean_pairwise_cos": mean_cos,
        "ctx_std": float(x.std()),
    }


def coupling_context_ratio(flow: ConditionalFlow) -> Dict[str, float]:
    """`mean|W_ctx| / mean|W_x|` over the couplings' first linear layer.

    The signature FINDINGS tracked. Each coupling's conditioner consumes
    `cat([x_normalised, ctx])`, so the two column blocks of `net[0].weight` are directly
    comparable and their ratio says how much of the pre-activation the context is worth.

    Read it with the caveat FINDINGS attaches: healthy cold-start cells sit at 0.85-0.91
    and healthy *warm-started* ones at 0.10-0.18 early, 0.33-0.40 by step 8000 -- the
    warm start loads large trained x-columns, so a low ratio is not by itself evidence of
    failure. What is diagnostic is the ratio failing to grow while the loss falls.
    """
    if flow.context_dim == 0:
        return {"w_ctx_over_w_x": 0.0, "w_x_abs_mean": 0.0, "w_ctx_abs_mean": 0.0}
    with torch.no_grad():
        wx = float(np.mean([float(c.net[0].weight[:, :flow.dim].abs().mean())
                            for c in flow.couplings]))
        wc = float(np.mean([float(c.net[0].weight[:, flow.dim:].abs().mean())
                            for c in flow.couplings]))
    return {"w_x_abs_mean": wx, "w_ctx_abs_mean": wc,
            "w_ctx_over_w_x": wc / max(1e-12, wx)}


# =======================================================================================
# The trainer
# =======================================================================================
class HeadOnlyFlowTrainer(FlowSFTTrainer):
    """`FlowSFTTrainer` on cached context. Inherits the metrics; that is the whole point.

    Nothing about the logging is redefined here. `_accumulate` and `_drain_metrics` are
    the parent's, so `eval_nll_per_dim`, `eval_nll_headroom`, `eval_logdet_absmax`,
    `eval_scale_saturation`, `eval_creep_*`, `eval_rmse_*`, `eval_mae_*`, `loss` and
    `grad_norm` are produced by the same code that produces them in `train_flow_sft.py`
    and are comparable digit for digit. A parallel implementation would drift and nothing
    would raise.

    Two deliberate departures:

      * `requires_unit_batch = False`. Both reasons for B == 1 are properties of the
        sparse backbone, and there is no backbone here.
      * the collapse probes, added to every eval. They are the diagnosis this pipeline
        was built to test, so they are logged next to the metrics rather than computed
        by a separate script afterwards.
    """

    requires_unit_batch = False

    def __init__(self, *args, probe_rows: Optional[np.ndarray] = None,
                 probe_store: Optional[OpenedContextStore] = None,
                 probe_context_key: str = "pooled", probe_max_rows: int = 4096,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.probe_store = probe_store
        self.probe_context_key = probe_context_key
        if probe_rows is not None and len(probe_rows) > probe_max_rows:
            # Deterministic subsample: the probe must not move between evals for reasons
            # that are about which rows it saw.
            probe_rows = np.asarray(probe_rows)[
                np.linspace(0, len(probe_rows) - 1, probe_max_rows).astype(int)]
        self.probe_rows = probe_rows

    def logdet_budget(self) -> float:
        flow = self.accelerator.unwrap_model(self.model).flow
        return float(torch.log(flow.unit_scale / flow.noise_std_raw).sum())

    @torch.no_grad()
    def collapse_metrics(self, prefix: str = "eval_") -> Dict[str, float]:
        """The FINDINGS diagnostics on the current head, on a fixed probe split."""
        model = self.accelerator.unwrap_model(self.model)
        out = {f"{prefix}{k}": v for k, v in coupling_context_ratio(model.flow).items()}
        if self.probe_store is None or self.probe_rows is None or len(self.probe_rows) < 2:
            return out
        was_training = model.training
        model.eval()
        raw = torch.from_numpy(np.ascontiguousarray(
            self.probe_store.get(self.probe_context_key, self.probe_rows))
        ).float().to(model.flow.unit_scale.device)
        ctx_raw = model.encode(raw)
        # Both vectors, because they answer different questions: `ctx_*` is what the
        # trunk hands over (`in_*`, fixed by construction here) versus what the head
        # emits and the flow is conditioned on (`ctx_*`, which is what can collapse).
        out.update({f"{prefix}{k}": v for k, v in
                    context_collapse_stats(ctx_raw.cpu().numpy()).items()})
        out.update({f"{prefix}in_{k}": v for k, v in
                    context_collapse_stats(raw.cpu().numpy()).items()})
        model.train(was_training)
        return out

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        extra = self.collapse_metrics(prefix=f"{metric_key_prefix}_")
        if extra:
            metrics.update(extra)
            self.log(extra)
        return metrics
