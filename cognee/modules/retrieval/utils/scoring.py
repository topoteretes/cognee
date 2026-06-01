"""Helpers for exposing and filtering vector-retrieval distance scores.

cognee's vector adapters return ``ScoredResult`` objects whose ``score`` is a raw
backend distance (cosine distance for the built-in adapters) where a lower value
means a closer match. The chunk and summary retrievers historically returned only
the payloads, so that distance was dropped before it reached the caller. These
helpers let a retriever keep the signal: attach the distance to each returned
payload, and drop matches that are farther away than a caller-supplied cutoff.
"""

from typing import Any, Dict, List, Optional

from cognee.infrastructure.databases.vector.models.ScoredResult import ScoredResult


def filter_by_max_distance(
    results: List[ScoredResult], max_distance: Optional[float]
) -> List[ScoredResult]:
    """Drop results whose distance is greater than ``max_distance``.

    Relies on the ``ScoredResult`` contract that a lower score is a closer match
    (cognee's built-in adapters report cosine distance), so a result is kept when
    its score is less than or equal to the cutoff. ``max_distance=None`` disables
    filtering and the input list is returned unchanged.
    """
    if max_distance is None:
        return results
    return [result for result in results if result.score <= max_distance]


def attach_scores(results: List[ScoredResult]) -> List[Dict[str, Any]]:
    """Return each result's payload with its retrieval distance added as ``score``.

    This matches the ``"score"`` key that ``SearchResultItem`` already reads when it
    normalizes search output, so the distance reaches the caller. A result with no
    payload yields a dict carrying just the ``score`` key. If a payload already has a
    ``score`` key it is overwritten with the retrieval distance.
    """
    return [{**(result.payload or {}), "score": result.score} for result in results]
