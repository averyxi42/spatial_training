"""Sampling a training stream from several datasets at a stated ratio.

The alternative is ``concatenate_datasets``, and it is wrong for this. Concatenation makes
the mixture a function of corpus *size*: ObjectNav is 39,061 rows against PointNav's ~5,000
and rising, so a concatenated table trains ~89% ObjectNav whatever anyone intended, and the
proportion drifts every time the generator adds an episode. Here the ratio is stated and
the sizes are irrelevant -- which is the property that survives a corpus growing overnight.

Concatenation is also, for these two corpora specifically, not possible: their ``messages``
columns are arrow-incompatible (``{content, role}`` against ``{role, content}`` -- pure
field ordering, identical as Python dicts) and PointNav carries ``obs_goals`` /
``segment_indices`` that ObjectNav has not got. Sampling per row sidesteps both, because a
``__getitem__`` hands back Python objects and the collator reads four columns by name:
``messages``, ``images``, ``action_chunks`` and the modality spec's column (``obs_poses``
-- a misnomer, since it holds agent and goal poses alike; see ``docs/column_naming.md``).
Nothing else reaches the model, so two corpora that agree on those four are
indistinguishable to it.

------------------------------------------------------------------------
Determinism
------------------------------------------------------------------------
The draw is a pure function of the index, not of call order:

    rng = default_rng([seed, index]) -> which source, then which row

This matters more than it looks. ``Trainer`` shuffles indices, and with
``dataloader_num_workers > 0`` the dataset object is pickled into each worker, so a
generator advanced per call would give a different mixture per worker and a different one
on every resume. Keyed on the index, every worker and every run agrees.

------------------------------------------------------------------------
"Epoch" stops meaning what it usually means
------------------------------------------------------------------------
Sampling is with replacement across sources of different sizes, so ``__len__`` is a
declared stream length rather than a fact about the data: at a 1:1 ratio the smaller corpus
is revisited several times per "epoch" and the larger one is not exhausted. That is
inherent to ratio mixing, not a defect -- but it means ``--max-steps`` is the only honest
budget, and an epoch count should not be read as a pass over the data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

#: The fixed columns `TurnVectorCollator` reads. It reads one more -- the modality
#: spec's own column -- which is not listed here because it is named by the spec at run
#: time and so cannot be checked without one. Everything else in a row -- `task`,
#: `obs_goals`, `segment_indices`, the whole stamped timing block -- is provenance or an
#: input to the *formatter*, and never reaches the model. Verified by inspection of
#: `vector_sft.TurnVectorCollator.__call__`, which has exactly four `ex[...]` reads.
COLLATOR_COLUMNS = ("messages", "images", "action_chunks")

#: Stamped fields that must agree across mixed sources. A disagreement here is not a
#: preference: two corpora at different `action_chunk_len` give the velocity field two
#: different `n_ticks`, which fails deep inside the head instead of at startup.
COMPATIBILITY_KEYS = (
    "native_fps", "dt_native", "obs_stride_frames", "action_chunk_len",
    "obs_interval", "chunk_duration", "v_max_mps", "w_max_radps",
)


class MixtureError(RuntimeError):
    """Sources that cannot be mixed. Always names the field and both values."""


@dataclass(frozen=True)
class MixtureSpec:
    """One source: where it is, which split, and its **unnormalised** weight."""

    name: str
    path: str
    ratio: float
    split: Optional[str] = None

    @classmethod
    def parse(cls, text: str, default_split: Optional[str] = None) -> "MixtureSpec":
        """``name=path:ratio`` -> a spec. ``path`` may contain ``:``, the ratio may not.

        Split on the LAST colon so an absolute path with a drive or a URL scheme in it
        still parses; a name is required because it is what the logs and any per-source
        metric are keyed on.
        """
        if "=" not in text:
            raise MixtureError(
                f"--mixture-datasets wants 'name=path:ratio', got {text!r} (no '=')"
            )
        name, rest = text.split("=", 1)
        if ":" not in rest:
            raise MixtureError(
                f"--mixture-datasets wants 'name=path:ratio', got {text!r} (no ':ratio')"
            )
        path, ratio = rest.rsplit(":", 1)
        try:
            weight = float(ratio)
        except ValueError:
            raise MixtureError(f"ratio {ratio!r} in {text!r} is not a number") from None
        if weight <= 0:
            raise MixtureError(f"ratio in {text!r} must be positive, got {weight}")
        return cls(name=name.strip(), path=path.strip(), ratio=weight, split=default_split)


def check_compatible(sources: Dict[str, Any], keys: Sequence[str] = COMPATIBILITY_KEYS) -> None:
    """Raise unless every source agrees on the stamped timing fields, and has the columns.

    Checked up front and named, because the alternative is a shape error thrown from inside
    the velocity field several minutes into a run, where nothing points at the dataset.
    A source missing a stamped field is *not* an error -- an older corpus may predate it --
    only a disagreement between two that both have it.
    """
    names = list(sources)
    missing = {
        n: [c for c in COLLATOR_COLUMNS if c not in sources[n].column_names]
        for n in names
    }
    bad = {n: cols for n, cols in missing.items() if cols}
    if bad:
        raise MixtureError(
            "these sources are missing columns the collator reads: "
            + "; ".join(f"{n} lacks {cols}" for n, cols in bad.items())
        )

    rows = {n: sources[n][0] for n in names}
    problems: List[str] = []
    for key in keys:
        seen = {n: rows[n][key] for n in names if key in sources[n].column_names}
        distinct = {}
        for n, v in seen.items():
            distinct.setdefault(v, []).append(n)
        if len(distinct) > 1:
            rendered = ", ".join(f"{v!r} ({'+'.join(ns)})" for v, ns in distinct.items())
            problems.append(f"  {key}: {rendered}")
    if problems:
        raise MixtureError(
            "REFUSING TO MIX: these sources were built with different chunking "
            "parameters, so their action chunks do not mean the same thing:\n"
            + "\n".join(problems)
            + "\nRebuild them with matching --obs-stride-frames / --action-chunk-len."
        )


class MixtureDataset:
    """Draw each example from one of several datasets, at a stated ratio.

    Implements only ``__len__`` / ``__getitem__``, which is all ``Trainer`` needs, so no
    trainer or collator change is required. Rows come back exactly as their source dataset
    yields them -- no column is added, removed or re-typed, so a source is byte-identical
    to how it trains alone.
    """

    def __init__(self, sources: Dict[str, Any], ratios: Dict[str, float],
                 length: Optional[int] = None, seed: int = 0,
                 check: bool = True) -> None:
        if not sources:
            raise MixtureError("a mixture needs at least one source")
        if set(sources) != set(ratios):
            raise MixtureError(
                f"sources {sorted(sources)} and ratios {sorted(ratios)} disagree"
            )
        if check:
            check_compatible(sources)
        self.names = sorted(sources)
        self.sources = sources
        total = float(sum(ratios[n] for n in self.names))
        if total <= 0:
            raise MixtureError("ratios must sum to something positive")
        self.ratios = {n: ratios[n] / total for n in self.names}
        self._p = np.array([self.ratios[n] for n in self.names], dtype=float)
        self._length = int(length) if length else sum(len(sources[n]) for n in self.names)
        self.seed = int(seed)

    def __len__(self) -> int:
        return self._length

    def _draw(self, index: int):
        # Keyed on the index, never on call order: `Trainer` shuffles, and the dataset is
        # pickled into every dataloader worker. A generator advanced per call would give a
        # different mixture per worker and a different one on resume.
        rng = np.random.default_rng([self.seed, int(index)])
        name = self.names[int(rng.choice(len(self.names), p=self._p))]
        source = self.sources[name]
        return name, int(rng.integers(0, len(source)))

    def __getitem__(self, index: int) -> Dict[str, Any]:
        name, row = self._draw(index)
        return self.sources[name][row]

    def source_of(self, index: int) -> str:
        """Which source index ``index`` comes from. For logging and tests."""
        return self._draw(index)[0]

    def realised_ratios(self, n: Optional[int] = None) -> Dict[str, float]:
        """The mixture actually produced over the first ``n`` indices.

        The requested ratio is a probability, so a short run will not match it exactly.
        Reporting the realised figure is how a 90/10 that was meant to be 50/50 gets
        noticed at step 0 rather than after a day of training.
        """
        n = int(n or min(len(self), 10000))
        counts: Dict[str, int] = {name: 0 for name in self.names}
        for i in range(n):
            counts[self.source_of(i)] += 1
        return {name: counts[name] / n for name in self.names}

    def describe(self) -> str:
        realised = self.realised_ratios()
        parts = [
            f"{n}: {len(self.sources[n])} rows, ratio {self.ratios[n]:.3f} "
            f"(realised {realised[n]:.3f})"
            for n in self.names
        ]
        return (f"MixtureDataset over {len(self.names)} source(s), stream length "
                f"{len(self)} (nominal -- sampling is with replacement)\n  "
                + "\n  ".join(parts))


def build_mixture(specs: Sequence[MixtureSpec], loader, split: str,
                  length: Optional[int] = None, seed: int = 0) -> MixtureDataset:
    """Load every spec through ``loader(path, split)`` and mix them."""
    if not specs:
        raise MixtureError("no --mixture-datasets given")
    names = [s.name for s in specs]
    if len(set(names)) != len(names):
        raise MixtureError(f"duplicate source names: {names}")
    sources = {s.name: loader(s.path, s.split or split) for s in specs}
    return MixtureDataset(sources, {s.name: s.ratio for s in specs},
                          length=length, seed=seed)
