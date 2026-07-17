from typing import Any, Dict, Optional
from uuid import UUID
from pydantic import BaseModel


def normalize_distance_to_relevance(score: float) -> float:
    """Map a raw distance-based score onto normalized relevance in ``[0, 1]``.

    Uses ``1 / (1 + max(0, score))`` because it is:

    * monotonically decreasing in ``score`` (lower distance always gives
      higher relevance),
    * defined for every finite non-negative score,
    * bounded on ``(0, 1]`` regardless of the distance metric, so
      thresholds set against relevance are portable across cosine,
      euclidean, and hybrid retrievers.

    Adapters that ship higher-is-better scores (dot product, similarity
    weights) should not call this helper; they can populate
    :attr:`ScoredResult.relevance` directly using a metric-appropriate
    mapping.
    """

    if score < 0:
        score = 0.0
    return 1.0 / (1.0 + score)


class ScoredResult(BaseModel):
    """
    Represents a vector retrieval result with an identification and associated data.

    Attributes:

    - id (UUID): Unique identifier for the scored result.
    - score (float): Raw backend distance score (cosine distance for built-in adapters), where a
    lower score indicates a better match.
    - relevance (Optional[float]): Normalized relevance in ``[0, 1]``, higher-is-better, comparable
    across retrievers and backends. Adapters can populate this directly or leave it unset for the
    consumer to fill in via :func:`normalize_distance_to_relevance` (or a metric-appropriate helper).
    - payload (Optional[Dict[str, Any]]): Additional information related to the score, stored as
    key-value pairs in a dictionary.
    """

    id: UUID
    score: float  # Lower score is better
    relevance: Optional[float] = None
    payload: Optional[Dict[str, Any]] = None
