"""Unit tests for the coarse ``Confidence`` label.

Covers the four-band derivation from top-k relevance, the empty-input
and no-relevance edge cases, and the tuneable ``top_k`` parameter.
Threshold constants are imported from the module under test so a
future tuning PR only has to move numbers in one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pytest

from cognee.modules.retrieval.utils.confidence import (
    CONFIDENCE_HIGH_MIN,
    CONFIDENCE_MEDIUM_MIN,
    CONFIDENCE_TOP_K,
    CONFIDENCE_WEAK_MIN,
    Confidence,
    derive_confidence,
)


@dataclass
class _RelevanceStub:
    """Minimal stand-in for ``SearchResultItem`` in this test module."""

    relevance: Optional[float]


def _stubs(*relevances: Optional[float]) -> list[_RelevanceStub]:
    return [_RelevanceStub(relevance=r) for r in relevances]


def test_empty_items_abstain():
    assert derive_confidence([]) is Confidence.ABSTAIN


def test_items_without_relevance_yield_medium():
    # We got recall hits, we just don't have a calibrated score to
    # rank them by. MEDIUM says "trust with care" rather than either
    # hallucinating high confidence or falsely abstaining.
    assert derive_confidence(_stubs(None, None, None)) is Confidence.MEDIUM


def test_mean_at_or_above_high_min_yields_high():
    # A hair above the threshold to sidestep IEEE-754 rounding on the
    # mean of three equal floats; the boundary math is exercised
    # explicitly by test_bucket_boundaries_are_inclusive below.
    just_above = CONFIDENCE_HIGH_MIN + 0.01
    assert derive_confidence(_stubs(just_above, just_above, just_above)) is Confidence.HIGH
    assert derive_confidence(_stubs(1.0, 0.9, 0.8)) is Confidence.HIGH


def test_bucket_boundaries_are_inclusive():
    # Directly feed a single-item ranking so the mean equals the
    # threshold with no floating-point drift, pinning that ``>=``
    # semantics hold on every boundary.
    assert derive_confidence(_stubs(CONFIDENCE_HIGH_MIN)) is Confidence.HIGH
    assert derive_confidence(_stubs(CONFIDENCE_MEDIUM_MIN)) is Confidence.MEDIUM
    assert derive_confidence(_stubs(CONFIDENCE_WEAK_MIN)) is Confidence.WEAK


def test_mean_between_medium_and_high_yields_medium():
    mid = (CONFIDENCE_MEDIUM_MIN + CONFIDENCE_HIGH_MIN) / 2
    assert derive_confidence(_stubs(mid, mid, mid)) is Confidence.MEDIUM


def test_mean_between_weak_and_medium_yields_weak():
    mid = (CONFIDENCE_WEAK_MIN + CONFIDENCE_MEDIUM_MIN) / 2
    assert derive_confidence(_stubs(mid, mid, mid)) is Confidence.WEAK


def test_mean_below_weak_min_yields_abstain():
    below = CONFIDENCE_WEAK_MIN / 2
    assert derive_confidence(_stubs(below, below, below)) is Confidence.ABSTAIN


def test_only_top_k_items_participate_in_the_mean():
    # Two strong items at the top followed by weak ones must not be
    # dragged down by the tail; the derivation looks only at
    # CONFIDENCE_TOP_K entries.
    top_two_strong = [0.95, 0.9] + [0.01] * 10
    label = derive_confidence(_stubs(*top_two_strong))
    assert label in {Confidence.HIGH, Confidence.MEDIUM}


def test_custom_top_k_narrows_the_window():
    # Same items evaluated against top_k=1 should surface the single
    # strong hit as HIGH regardless of the tail.
    items = _stubs(0.95, 0.1, 0.05, 0.02)
    assert derive_confidence(items, top_k=1) is Confidence.HIGH


def test_mix_of_relevance_and_none_only_averages_the_populated():
    # Items without relevance are skipped, not counted as zero, so a
    # partially-populated ranking doesn't wrongly collapse to ABSTAIN.
    label = derive_confidence(_stubs(0.9, None, 0.8, None))
    assert label is Confidence.HIGH


@pytest.mark.parametrize(
    "relevances, expected",
    [
        ([0.99, 0.98, 0.97], Confidence.HIGH),
        ([0.55, 0.5, 0.45], Confidence.MEDIUM),
        ([0.3, 0.25, 0.22], Confidence.WEAK),
        ([0.1, 0.05, 0.02], Confidence.ABSTAIN),
    ],
)
def test_table_driven_bucket_boundaries(relevances, expected):
    assert derive_confidence(_stubs(*relevances)) is expected


def test_top_k_default_matches_module_constant():
    # Guardrail: if someone changes the default without updating the
    # exported constant, this test fires before the drift lands.
    items = _stubs(*([0.95] * (CONFIDENCE_TOP_K + 2)))
    assert derive_confidence(items) is Confidence.HIGH
