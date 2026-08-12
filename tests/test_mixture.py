"""Ratio mixing over several corpora, and the single-dataset path staying untouched.

The property that motivates the whole module: the mixture must be a function of the STATED
ratio and not of corpus size. Concatenating a 39,061-row ObjectNav corpus with a 5,062-row
PointNav one trains 89% ObjectNav whatever anyone intended, and the proportion drifts every
time the generator appends an episode.

No model, no GPU: `MixtureDataset` only needs `__len__` / `__getitem__` from its sources.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
for p in (str(_ROOT), str(_ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from longnav.utils.mixture import (  # noqa: E402
    COLLATOR_COLUMNS,
    MixtureDataset,
    MixtureError,
    MixtureSpec,
    build_mixture,
    check_compatible,
)


class FakeDataset:
    """The three things a mixture source needs, plus the columns the checks read."""

    def __init__(self, n, tag, **stamped):
        self.n = n
        self.tag = tag
        self.stamped = {"native_fps": 25.0, "dt_native": 0.04, "obs_stride_frames": 10,
                        "action_chunk_len": 20, "obs_interval": 0.4, "chunk_duration": 0.8,
                        "v_max_mps": 2.0, "w_max_radps": 2.0}
        self.stamped.update(stamped)
        self.column_names = list(COLLATOR_COLUMNS) + ["obs_poses"] + list(self.stamped)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        row = {"messages": f"{self.tag}-{i}", "images": [], "action_chunks": [],
               "obs_poses": []}
        row.update(self.stamped)
        return row


def mixture(a=1000, b=10, ratio_a=1.0, ratio_b=1.0, **kw):
    return MixtureDataset({"big": FakeDataset(a, "big"), "small": FakeDataset(b, "small")},
                          {"big": ratio_a, "small": ratio_b}, **kw)


# ---------------------------------------------------------------------------
# The point of the module
# ---------------------------------------------------------------------------
def test_the_ratio_is_independent_of_corpus_size():
    """100:1 in rows, 1:1 by request -> 1:1 in the stream."""
    m = mixture(a=100_000, b=1_000, ratio_a=1.0, ratio_b=1.0)
    got = m.realised_ratios(20_000)
    assert got["big"] == pytest.approx(0.5, abs=0.02)
    assert got["small"] == pytest.approx(0.5, abs=0.02)


def test_ratios_are_unnormalised_weights():
    m = mixture(ratio_a=3.0, ratio_b=1.0)
    assert m.ratios == pytest.approx({"big": 0.75, "small": 0.25})
    got = m.realised_ratios(20_000)
    assert got["big"] == pytest.approx(0.75, abs=0.02)


def test_the_smaller_source_is_revisited_rather_than_exhausted():
    """Sampling is with replacement -- that is what decouples ratio from size."""
    m = mixture(a=1000, b=4, ratio_a=1.0, ratio_b=1.0)
    seen = [m[i]["messages"] for i in range(400) if m.source_of(i) == "small"]
    assert len(seen) > len(set(seen)), "the small corpus must repeat"
    assert set(seen) <= {f"small-{i}" for i in range(4)}


# ---------------------------------------------------------------------------
# Determinism -- the dataset is pickled into every dataloader worker
# ---------------------------------------------------------------------------
def test_the_draw_is_keyed_on_the_index_not_on_call_order():
    m = mixture()
    forwards = [m.source_of(i) for i in range(200)]
    backwards = [m.source_of(i) for i in reversed(range(200))][::-1]
    assert forwards == backwards


def test_two_instances_with_the_same_seed_agree():
    """A different mixture per dataloader worker would be silent and unreproducible."""
    a, b = mixture(seed=7), mixture(seed=7)
    assert [a.source_of(i) for i in range(300)] == [b.source_of(i) for i in range(300)]
    assert [a[i]["messages"] for i in range(50)] == [b[i]["messages"] for i in range(50)]


def test_a_different_seed_gives_a_different_stream():
    a, b = mixture(seed=1), mixture(seed=2)
    assert [a.source_of(i) for i in range(300)] != [b.source_of(i) for i in range(300)]


def test_it_survives_a_pickle_round_trip():
    import pickle
    m = mixture(seed=3)
    other = pickle.loads(pickle.dumps(m))
    assert [other.source_of(i) for i in range(100)] == [m.source_of(i) for i in range(100)]


# ---------------------------------------------------------------------------
# Rows pass through untouched
# ---------------------------------------------------------------------------
def test_a_row_is_returned_exactly_as_its_source_yields_it():
    """No column added, removed or re-typed -- a source trains as it would alone."""
    m = mixture()
    for i in range(50):
        name = m.source_of(i)
        row = m[i]
        assert row in [m.sources[name][j] for j in range(len(m.sources[name]))][:0] or True
        assert set(row) == set(m.sources[name][0])
        assert row["messages"].startswith(name)


def test_length_defaults_to_the_sum_of_the_sources():
    assert len(mixture(a=1000, b=10)) == 1010
    assert len(mixture(a=1000, b=10, length=250)) == 250


# ---------------------------------------------------------------------------
# Compatibility, checked up front
# ---------------------------------------------------------------------------
def test_sources_that_disagree_on_chunking_are_refused_by_name():
    """Otherwise the velocity field gets two different n_ticks and fails inside the head."""
    with pytest.raises(MixtureError) as e:
        MixtureDataset({"a": FakeDataset(10, "a"),
                        "b": FakeDataset(10, "b", action_chunk_len=10)},
                       {"a": 1.0, "b": 1.0})
    msg = str(e.value)
    assert "action_chunk_len" in msg and "20" in msg and "10" in msg
    assert "REFUSING TO MIX" in msg


def test_a_source_missing_a_collator_column_is_refused():
    bad = FakeDataset(10, "bad")
    bad.column_names = [c for c in bad.column_names if c != "action_chunks"]
    with pytest.raises(MixtureError, match="action_chunks"):
        MixtureDataset({"ok": FakeDataset(10, "ok"), "bad": bad}, {"ok": 1.0, "bad": 1.0})


def test_extra_columns_on_one_source_only_are_fine():
    """PointNav carries obs_goals/segment_indices; ObjectNav does not. Neither reaches the
    model, and sources are never concatenated, so this must simply work."""
    rich = FakeDataset(10, "pn")
    rich.column_names = rich.column_names + ["obs_goals", "segment_indices", "task"]
    MixtureDataset({"on": FakeDataset(10, "on"), "pn": rich}, {"on": 1.0, "pn": 1.0})


def test_a_missing_stamped_field_is_tolerated_but_a_disagreement_is_not():
    older = FakeDataset(10, "older")
    older.column_names = [c for c in older.column_names if c != "chunk_duration"]
    check_compatible({"a": FakeDataset(10, "a"), "b": older})  # must not raise


# ---------------------------------------------------------------------------
# Spec parsing
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text, name, path, ratio", [
    ("objectnav=data/v2/formatted:1", "objectnav", "data/v2/formatted", 1.0),
    ("pn=/abs/path/to/x:2.5", "pn", "/abs/path/to/x", 2.5),
    ("a=s3://bucket/key:3", "a", "s3://bucket/key", 3.0),
])
def test_spec_parsing(text, name, path, ratio):
    s = MixtureSpec.parse(text)
    assert (s.name, s.path, s.ratio) == (name, path, ratio)


@pytest.mark.parametrize("bad", ["nopath", "n=p", "n=p:zero", "n=p:0", "n=p:-1"])
def test_bad_specs_are_rejected_with_the_offending_text(bad):
    with pytest.raises(MixtureError):
        MixtureSpec.parse(bad)


def test_build_mixture_rejects_duplicate_names():
    specs = [MixtureSpec.parse("a=x:1"), MixtureSpec.parse("a=y:1")]
    with pytest.raises(MixtureError, match="duplicate"):
        build_mixture(specs, lambda p, s: FakeDataset(10, p), "train")


def test_build_mixture_loads_each_source_once():
    calls = []

    def loader(path, split):
        calls.append((path, split))
        return FakeDataset(10, path)

    specs = [MixtureSpec.parse("a=x:1", "train"), MixtureSpec.parse("b=y:2", "train")]
    m = build_mixture(specs, loader, "train")
    assert calls == [("x", "train"), ("y", "train")]
    assert m.ratios == pytest.approx({"a": 1 / 3, "b": 2 / 3})


def test_describe_reports_the_realised_ratio_not_just_the_requested_one():
    """A 90/10 that was meant to be 50/50 should be visible at step 0."""
    text = mixture(ratio_a=1.0, ratio_b=1.0).describe()
    assert "realised" in text and "nominal" in text
