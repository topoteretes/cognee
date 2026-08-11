from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from ..graph.scoring import cosine_distance
from ...models import SummaryNode


async def embed_items_for_divisive_split(
    items: list[SummaryNode],
    vector_engine: Any,
) -> dict[str, list[float]]:
    """
    Batch-embed every item's text once, so pole-selection/side-assignment can
    compare arbitrary pairs. The existing "vector" strategy only does ANN
    ``search()`` for top-K neighbors, which can't give a pairwise distance
    between two arbitrary items -- this re-embeds directly via the embedding
    engine instead, the same primitive ``bucketing/graph/inputs.py``'s
    relationship-name embedding already uses.
    """
    if not items:
        return {}

    texts = [item.text for item in items]
    vectors = await vector_engine.embedding_engine.embed_text(texts)
    return {item.id: vector for item, vector in zip(items, vectors)}


def build_vector_pole_a_fn(
    vectors_by_id: Mapping[str, list[float]],
) -> Callable[[list[str]], str]:
    """
    Pole A: the item farthest (by cosine distance) from the current
    subgroup's centroid -- the vector-space analog of "most distinctive".
    Ties break on ascending id.
    """

    def pole_a_fn(ids: list[str]) -> str:
        dimensions = len(vectors_by_id[ids[0]])
        centroid = [
            sum(vectors_by_id[item_id][dimension] for item_id in ids) / len(ids)
            for dimension in range(dimensions)
        ]
        return sorted(
            ids,
            key=lambda item_id: (-cosine_distance(vectors_by_id[item_id], centroid), item_id),
        )[0]

    return pole_a_fn


def build_vector_similarity_fn(
    vectors_by_id: Mapping[str, list[float]],
) -> Callable[[str, str], float]:
    def similarity_fn(left_id: str, right_id: str) -> float:
        return 1.0 - cosine_distance(vectors_by_id[left_id], vectors_by_id[right_id])

    return similarity_fn
