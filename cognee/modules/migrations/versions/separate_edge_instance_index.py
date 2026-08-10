"""Rebuild edge type and edge instance vector indexes from graph state."""

from collections import Counter
from collections.abc import AsyncIterator
from copy import deepcopy
from typing import Any

from cognee.modules.engine.utils import generate_edge_object_id
from cognee.modules.graph.models.EdgeType import EdgeType
from cognee.modules.graph.utils.edge_index_points import build_edge_index_points
from cognee.modules.graph.utils.prepare_edges_for_storage import get_edge_retrieval_text
from cognee.modules.migrations.migration import MigrationContext

_BATCH_SIZE = 1000
_Edge = tuple[Any, Any, str, dict[str, Any]]


def _triplet_to_edge(triplet: dict[str, Any]) -> _Edge:
    relationship = dict(triplet["relationship_properties"] or {})
    relationship_name = relationship.pop("relationship_name")
    nested_properties = relationship.pop("properties", None)
    properties = dict(nested_properties) if isinstance(nested_properties, dict) else {}
    properties.update(relationship)
    return (
        triplet["start_node"]["id"],
        triplet["end_node"]["id"],
        relationship_name,
        properties,
    )


def _is_indexable_edge(edge: _Edge) -> bool:
    source_id, target_id, relationship_name, _ = edge
    return not (relationship_name == "SELF" and str(source_id) == str(target_id))


class _GraphEdgeSource:
    """Replay graph edges for two passes, using one full-graph fallback at most."""

    def __init__(self, graph_engine: Any):
        self.graph_engine = graph_engine
        self._fallback_edges: list | None = None
        self._pagination_supported: bool | None = None

    async def batches(self) -> AsyncIterator[list[_Edge]]:
        if self._pagination_supported is not False:
            offset = 0
            try:
                while True:
                    triplets = await self.graph_engine.get_triplets_batch(offset, _BATCH_SIZE)
                    self._pagination_supported = True
                    if not triplets:
                        return
                    yield [
                        edge
                        for triplet in triplets
                        if _is_indexable_edge(edge := _triplet_to_edge(triplet))
                    ]
                    offset += len(triplets)
            except NotImplementedError:
                self._pagination_supported = False

        if self._fallback_edges is None:
            _, self._fallback_edges = await self.graph_engine.get_graph_data()
        for offset in range(0, len(self._fallback_edges), _BATCH_SIZE):
            yield [
                (source, target, relationship_name, dict(properties or {}))
                for source, target, relationship_name, properties in self._fallback_edges[
                    offset : offset + _BATCH_SIZE
                ]
                if not (relationship_name == "SELF" and str(source) == str(target))
            ]


async def _one_batch(points: list) -> AsyncIterator[list]:
    yield points


async def _empty_batches() -> AsyncIterator[list]:
    if False:
        yield []


async def _repair_and_count(source: _GraphEdgeSource) -> Counter[str]:
    relationship_counts: Counter[str] = Counter()
    async for batch in source.batches():
        repaired_edges = []
        for source_id, target_id, relationship_name, properties in batch:
            relationship_counts[relationship_name] += 1
            if not properties.get("edge_object_id"):
                repaired_properties = deepcopy(properties)
                repaired_properties["edge_object_id"] = generate_edge_object_id(
                    source_id, target_id, relationship_name
                )
                repaired_edges.append(
                    (source_id, target_id, relationship_name, repaired_properties)
                )
        if repaired_edges:
            await source.graph_engine.add_edges(repaired_edges)
    return relationship_counts


async def _instance_batches(source: _GraphEdgeSource) -> AsyncIterator[list]:
    async for batch in source.batches():
        yield build_edge_index_points(batch).edge_instances


async def migrate(context: MigrationContext) -> None:
    """Replace both edge indexes with the exact state derived from the graph."""
    source = _GraphEdgeSource(context.graph_engine)
    relationship_counts = await _repair_and_count(source)
    edge_types = [
        EdgeType(relationship_name=name, number_of_edges=count)
        for name, count in relationship_counts.items()
    ]

    await context.vector_engine.replace_index_data_points(
        "EdgeType", "relationship_name", _one_batch(edge_types)
    )
    await context.vector_engine.replace_index_data_points(
        "EdgeInstance", "text", _instance_batches(source)
    )


async def _legacy_edge_type_points(source: _GraphEdgeSource) -> list[EdgeType]:
    legacy_counts: Counter[str] = Counter()
    async for batch in source.batches():
        for _, _, relationship_name, properties in batch:
            legacy_counts[
                get_edge_retrieval_text(properties.get("edge_text"), relationship_name)
            ] += 1
    return [
        EdgeType(relationship_name=text, number_of_edges=count)
        for text, count in legacy_counts.items()
        if text
    ]


async def downgrade(context: MigrationContext) -> None:
    """Remove instance points and restore the legacy prose-keyed EdgeType index."""
    source = _GraphEdgeSource(context.graph_engine)
    legacy_edge_types = await _legacy_edge_type_points(source)
    await context.vector_engine.replace_index_data_points("EdgeInstance", "text", _empty_batches())
    await context.vector_engine.replace_index_data_points(
        "EdgeType", "relationship_name", _one_batch(legacy_edge_types)
    )
