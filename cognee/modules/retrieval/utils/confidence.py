"""Coarse confidence signal derived from top-k relevance scores.

An agent that consumes cognee memory needs a machine-readable answer
to "should I trust this recall?". A raw relevance number alone is hard
to threshold across retrievers and datasets, so we roll top-k
relevance into a small enum that any caller can branch on without
tuning constants themselves.

The derivation is deliberately conservative:

* No items in the result → :attr:`Confidence.ABSTAIN`.
* Items with no relevance signal at all → :attr:`Confidence.MEDIUM`.
  Retrieval succeeded (a hit came back), but nothing gave us a
  calibrated score to rank on. Emitting ``MEDIUM`` says "trust with
  care" rather than falsely projecting either high confidence or
  outright abstention.
* Items with relevance → label from the mean of the top-``k`` scores
  against the thresholds below.

Thresholds are module-level constants so a follow-up PR can tune them
against a benchmark without touching call sites, and so tests can
parameterize against the same values the runtime uses.
"""

from __future__ import annotations

from enum import Enum
from typing import Sequence

# Top-k window used to derive confidence. Small so a single strong hit
# still counts; large enough that a single outlier does not.
CONFIDENCE_TOP_K = 3

# Lower bounds for each confidence bucket (mean relevance of the top-k).
# Anything below CONFIDENCE_WEAK_MIN is treated as ABSTAIN so an agent
# can branch on "don't answer" instead of hallucinating from weak
# support.
CONFIDENCE_HIGH_MIN = 0.70
CONFIDENCE_MEDIUM_MIN = 0.40
CONFIDENCE_WEAK_MIN = 0.20


class Confidence(str, Enum):
    """Coarse confidence label emitted by ``derive_confidence``.

    Ordered by strength of support. Consumers should branch on the
    named values rather than the string representation to stay
    resilient to future additions.
    """

    HIGH = "high"
    MEDIUM = "medium"
    WEAK = "weak"
    ABSTAIN = "abstain"


class _RelevanceCarrier:
    """Duck-typed protocol for objects that expose ``.relevance``.

    Kept as a private structural hint rather than a Protocol import so
    the module has no runtime dependency on typing extensions. Any
    object with a ``relevance`` attribute (``SearchResultItem``,
    ``ScoredResult`` once we add it, or a caller-supplied surrogate for
    tests) is accepted by :func:`derive_confidence`.
    """

    relevance: float | None


def derive_confidence(
    items: Sequence[_RelevanceCarrier],
    top_k: int = CONFIDENCE_TOP_K,
) -> Confidence:
    """Return a coarse confidence label for a ranked item list.

    Parameters
    ----------
    items
        The ranked items. The function reads ``.relevance`` on each
        one; anything without a relevance signal is skipped when
        computing the mean.
    top_k
        Window size for the mean. Callers can override the default to
        tune sensitivity (a longer window smooths out a single strong
        hit; a shorter one amplifies it).
    """

    if not items:
        return Confidence.ABSTAIN

    window = list(items[:top_k])
    relevances = [
        getattr(item, "relevance", None)
        for item in window
        if getattr(item, "relevance", None) is not None
    ]
    if not relevances:
        return Confidence.MEDIUM

    mean = sum(relevances) / len(relevances)
    if mean >= CONFIDENCE_HIGH_MIN:
        return Confidence.HIGH
    if mean >= CONFIDENCE_MEDIUM_MIN:
        return Confidence.MEDIUM
    if mean >= CONFIDENCE_WEAK_MIN:
        return Confidence.WEAK
    return Confidence.ABSTAIN
