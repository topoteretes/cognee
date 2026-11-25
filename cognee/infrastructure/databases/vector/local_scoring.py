"""Cosine distances of fetched vectors against several query vectors, computed locally.

LanceDB's ``score_by_ids_batch`` fetches the rows of the requested ids once and hands
them here, so a batch of queries costs one filtered scan instead of one per vector.
The distance is ``1 - cosine similarity``, what every adapter's ``score_by_ids``
reports. The SQL adapters compute their distances in the engine instead, one column
per query, because shipping the vectors out costs more than the statements it saves.
"""

from typing import Any

import numpy as np

from cognee.infrastructure.engine.utils import parse_id

from .models.ScoredResult import ScoredResult


def score_rows_against_queries(
    rows: list[tuple[Any, list[float]]], query_vectors: list[list[float]]
) -> list[list[ScoredResult]]:
    """One result list per query vector, in order; each row scored once per query."""
    scores: list[list[ScoredResult]] = [[] for _ in query_vectors]
    if not rows or not query_vectors:
        return scores
    queries = np.asarray(query_vectors, dtype=np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    row_ids = [parse_id(str(row_id)) for row_id, _ in rows]
    vectors = np.asarray([vector for _, vector in rows], dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    distances = 1.0 - queries @ vectors.T  # (queries, rows)
    for query_scores, row_distances in zip(scores, distances):
        query_scores.extend(
            ScoredResult(id=row_id, score=float(distance), payload=None)
            for row_id, distance in zip(row_ids, row_distances)
        )
    return scores
