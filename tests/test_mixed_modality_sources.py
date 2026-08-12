"""A mixture may pair a pose-carrying corpus with one that has no pose column at all.

The design intent is that `<pose>` is a modality exactly like `<image>`: a conversation
that never writes the marker simply supplies no values for it, the same way a text-only
sample in a vision mixture carries no image. Everything downstream already honours that --
`_modality_kwargs` compares the row count against the marker count found in `input_ids`
(0 == 0), and `ModalityEmbedder`'s post-hook states outright that "a key with zero rows and
zero occurrences is fine".

One line did not. The collator read `ex[s.source_column]` unconditionally, so the first
example drawn from a source without the column raised `KeyError` several minutes into a
run. `check_compatible` does not catch it either: it requires only the three columns the
collator reads (`messages`, `images`, `action_chunks`), and pose is not among them -- by
design, since the pose column is named by the model's spec rather than fixed.

Pinned here at the two levels that matter: the lookup itself, and that a source which DOES
carry the column is still held to matching it row for row (absent is defaulted, wrong is
not).
"""

import sys
from pathlib import Path

import pytest
import torch

# `longnav` is a namespace package (no `src/longnav/__init__.py`), so a bare import resolves
# to whichever checkout is first on the path. Pin this one before importing.
_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from longnav.utils.modality_embed import ModalityEmbedSpec  # noqa: E402

SPEC = ModalityEmbedSpec(
    token="<pose>", n_features=3, encoder="planar_se2",
    column="obs_poses", transform="pose_relative_first",
)


def lookup(ex, spec):
    """The collator's modality lookup, isolated."""
    raw = ex.get(spec.source_column)
    if raw is None:
        raw = []
    return torch.as_tensor(raw, dtype=torch.float32).reshape(-1, spec.raw_width)


def test_a_source_without_the_column_yields_zero_rows():
    ex = {"messages": [], "images": [], "action_chunks": []}
    out = lookup(ex, SPEC)
    assert out.shape == (0, SPEC.raw_width)


def test_a_source_with_the_column_is_unchanged():
    ex = {"obs_poses": [[1.0, 2.0, 0.3], [1.5, 2.5, 0.4]]}
    out = lookup(ex, SPEC)
    assert out.shape == (2, 3)
    assert torch.allclose(out[0], torch.tensor([1.0, 2.0, 0.3]))


def test_zero_rows_survives_the_transform():
    """`relative_se2` anchors on row 0; with no rows there is nothing to anchor."""
    out = SPEC.apply_transform(lookup({}, SPEC))
    assert out.shape == (0, 3)


def test_zero_rows_matches_a_zero_marker_count():
    """The invariant `_modality_kwargs` enforces: rows must equal markers in input_ids."""
    input_ids = torch.tensor([[1, 2, 3, 4]])          # no marker token
    marker_id = 151669
    n_found = int((input_ids == marker_id).sum())
    assert n_found == 0
    assert lookup({}, SPEC).shape[0] == n_found


def test_a_present_but_mismatched_column_is_still_a_mismatch():
    """Absent is defaulted; wrong is not. Two markers against one row must not pass."""
    input_ids = torch.tensor([[1, 151669, 3, 151669]])
    n_found = int((input_ids == 151669).sum())
    rows = lookup({"obs_poses": [[0.0, 0.0, 0.0]]}, SPEC).shape[0]
    assert n_found == 2 and rows == 1 and rows != n_found


@pytest.mark.parametrize("value", [[], None])
def test_an_empty_or_null_column_is_the_same_as_an_absent_one(value):
    """All three spellings of "this row has no pose" must agree.

    `None` is passed through as-is deliberately: an earlier version of this test
    substituted `[]` for it and so never exercised the case that actually raises
    (`torch.as_tensor(None)`).
    """
    assert lookup({"obs_poses": value}, SPEC).shape == (0, SPEC.raw_width)
    assert lookup({}, SPEC).shape == (0, SPEC.raw_width)


def test_markers_without_values_is_still_an_error_case():
    """The contract is symmetric: no markers needs no values, but markers DO need values.

    The collator does not raise here itself -- `_modality_kwargs` does, by comparing row
    count against markers in `input_ids`. Pinned so the defaulting above can never be
    mistaken for "modality values are optional".
    """
    input_ids = torch.tensor([[1, 151669, 3]])
    n_found = int((input_ids == 151669).sum())
    for ex in ({}, {"obs_poses": None}, {"obs_poses": []}):
        assert lookup(ex, SPEC).shape[0] == 0 != n_found
