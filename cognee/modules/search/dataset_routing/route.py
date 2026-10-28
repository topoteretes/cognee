"""Pick the datasets a query should be searched in, from the routing collection.

Every readable dataset is scored against the query by its best routing row:
its root (the dataset's vocabulary) or one of its document buckets (one
document's distinctive entities and relations), so a specific question is
carried by the document that matches and a broad one by the root. The rows
are addressed by id, derived from the caller's datasets and their documents,
and scored with ``score_by_ids``, so no other tenant's row is ever read and
nothing is cut off by a result limit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError
from cognee.modules.data.methods.get_data_ids_by_dataset import get_data_ids_by_dataset
from cognee.shared.logging_utils import get_logger

from .index import dataset_route_ids
from .store import DATASET_ROUTE_COLLECTION, get_dataset_route_engine, routing_width_matches

logger = get_logger("dataset_routing")


@dataclass(frozen=True)
class DatasetRoutingDecision:
    """Which of the candidate datasets a recall searches, and why."""

    candidates: tuple[UUID, ...]  # every dataset that was scored or could have been
    routed: tuple[UUID, ...]  # the best-scoring datasets, best first
    unindexed: tuple[UUID, ...]  # datasets with no routing row: always searched
    scores: dict[UUID, float] = field(default_factory=dict)  # best cosine distance per dataset

    @property
    def dataset_ids(self) -> list[UUID]:
        return [*self.routed, *self.unindexed]

    @property
    def narrowed(self) -> bool:
        return len(self.dataset_ids) < len(self.candidates)


async def route_datasets(
    query_text: str, dataset_ids: list[UUID], top_k: int
) -> DatasetRoutingDecision:
    """Keep the ``top_k`` best-scoring datasets, plus every dataset that has no routing row.

    A dataset without a row has never been through the global context index
    (or lost it), so routing knows nothing about it and never drops it. An
    adapter that cannot score rows by id, or a routing collection built with
    another embedding width, leaves every dataset unindexed. So does a
    deployment with no routing collection at all, before the query is
    embedded: a search type that needs no embedding (CODE, lexical, Cypher)
    must not start needing one for a routing step that cannot narrow anything.
    An embedding failure is a warning and leaves every dataset unindexed too,
    for the same reason: routing is a narrowing, never the search itself.
    """
    if top_k < 1:
        raise ValueError("Dataset routing needs top_k >= 1.")
    candidates = tuple(dataset_ids)

    engine = get_dataset_route_engine()
    if not await engine.has_collection(DATASET_ROUTE_COLLECTION):
        return DatasetRoutingDecision(candidates=candidates, routed=(), unindexed=candidates)
    if not await routing_width_matches(engine):
        return DatasetRoutingDecision(candidates=candidates, routed=(), unindexed=candidates)

    documents = await get_data_ids_by_dataset(list(candidates))
    dataset_by_row: dict[str, UUID] = {}
    for dataset_id in candidates:
        for row_id in dataset_route_ids(dataset_id, documents.get(dataset_id, [])):
            dataset_by_row[row_id] = dataset_id

    try:
        query_vector = (await engine.embedding_engine.embed_text([query_text]))[0]
    except Exception:
        logger.warning(
            "Dataset routing is off for this search: the query could not be embedded. "
            "Searching every readable dataset.",
            exc_info=True,
        )
        return DatasetRoutingDecision(candidates=candidates, routed=(), unindexed=candidates)
    try:
        scored = await engine.score_by_ids(
            DATASET_ROUTE_COLLECTION, list(dataset_by_row), query_vector
        )
    except CollectionNotFoundError:
        scored = []
    except NotImplementedError:
        logger.debug("Vector adapter cannot score rows by id; dataset routing is off")
        scored = []

    best: dict[UUID, float] = {}
    for result in scored:
        dataset_id = dataset_by_row.get(str(result.id))
        if dataset_id is None:
            continue
        if dataset_id not in best or result.score < best[dataset_id]:
            best[dataset_id] = result.score

    ranked = sorted(best, key=lambda dataset_id: (best[dataset_id], str(dataset_id)))
    return DatasetRoutingDecision(
        candidates=candidates,
        routed=tuple(ranked[:top_k]),
        unindexed=tuple(dataset_id for dataset_id in candidates if dataset_id not in best),
        scores=best,
    )
