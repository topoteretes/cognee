"""Live Graphiti (OSS) memory source."""

from __future__ import annotations

from typing import Any, AsyncIterator, Dict, List, Optional

from cognee.modules.migration.cogx import COGXRecord
from cognee.modules.migration.sources.base import MemorySource
from cognee.modules.migration.sources.live._utils import call_maybe_async
from cognee.modules.migration.sources.zep import iter_zep_graph_records


def _serialize_graphiti_node(node: Any) -> Dict[str, Any]:
    if hasattr(node, "model_dump"):
        data = node.model_dump(mode="json")
    elif isinstance(node, dict):
        data = dict(node)
    elif hasattr(node, "__dict__"):
        data = {key: value for key, value in vars(node).items() if not key.startswith("_")}
    else:
        data = {key: getattr(node, key) for key in dir(node) if not key.startswith("_")}
    source = data.get("source")
    if hasattr(source, "value"):
        data["source"] = source.value
    return data


async def _paginate_graphiti_nodes(
    node_cls: Any, driver: Any, group_ids: List[str], page_size: int
):
    items: List[Any] = []
    cursor = None
    while True:
        kwargs: Dict[str, Any] = {
            "driver": driver,
            "group_ids": group_ids,
            "limit": page_size,
        }
        if cursor is not None:
            kwargs["uuid_cursor"] = cursor
        try:
            batch = await call_maybe_async(node_cls.get_by_group_ids, **kwargs)
        except Exception as error:
            error_name = type(error).__name__
            if error_name in ("GroupsEdgesNotFoundError", "GroupsNodesNotFoundError"):
                return []
            raise
        if not batch:
            break
        items.extend(batch)
        if len(batch) < page_size:
            break
        last = batch[-1]
        cursor = getattr(last, "uuid", None)
        if cursor is None:
            break
    return items


async def fetch_graphiti_snapshot(
    graphiti: Any,
    group_ids: Optional[List[str]] = None,
    page_size: int = 500,
) -> Dict[str, Any]:
    """Export a Graphiti knowledge graph into the ZepSource dict shape."""
    from cognee.modules.migration.sources.live._utils import require_extra

    graphiti_core = require_extra("graphiti", "graphiti_core")
    EpisodicNode = graphiti_core.nodes.EpisodicNode
    EntityNode = graphiti_core.nodes.EntityNode
    EntityEdge = graphiti_core.edges.EntityEdge

    driver = getattr(graphiti, "driver", None)
    if driver is None and hasattr(graphiti, "clients"):
        driver = graphiti.clients.driver
    if driver is None:
        raise ValueError("Graphiti instance must expose a graph driver.")
    resolved_group_ids = group_ids
    if resolved_group_ids is None:
        resolved_group_ids = await _discover_group_ids(driver)

    episodes = await _paginate_graphiti_nodes(EpisodicNode, driver, resolved_group_ids, page_size)
    nodes = await _paginate_graphiti_nodes(EntityNode, driver, resolved_group_ids, page_size)
    edges = await _paginate_graphiti_nodes(EntityEdge, driver, resolved_group_ids, page_size)

    return {
        "episodes": [_serialize_graphiti_node(episode) for episode in episodes],
        "nodes": [_serialize_graphiti_node(node) for node in nodes],
        "edges": [_serialize_graphiti_node(edge) for edge in edges],
    }


async def _discover_group_ids(driver: Any) -> List[str]:
    query = "MATCH (n) WHERE n.group_id IS NOT NULL RETURN DISTINCT n.group_id AS group_id"
    try:
        result = await call_maybe_async(driver.execute_query, query)
    except Exception:
        return [""]
    records = result
    if isinstance(result, tuple):
        records = result[0]
    group_ids = []
    for row in records or []:
        if isinstance(row, dict):
            value = row.get("group_id")
        else:
            value = row[0] if row else None
        if value is not None and value not in group_ids:
            group_ids.append(value)
    return group_ids or [""]


class GraphitiLiveSource(MemorySource):
    """Fetch a Graphiti knowledge graph from a running graph database.

    Pass an injected ``graphiti_core.Graphiti`` instance (credentials and
    connection are configured by the caller). On first ``records()`` the full
    graph is snapshotted into memory and replayed on subsequent calls.

    Communities, sagas, and non-entity edge types are not exported.

    Args:
        graphiti: Connected ``graphiti_core.Graphiti`` instance.
        group_ids: Graph partitions to export; ``None`` discovers distinct
            ``group_id`` values via Cypher.
        page_size: Batch size for ``get_by_group_ids`` pagination.
        mode: Import mode (default ``hybrid``).
        close_on_fetch: Close the Graphiti client after the snapshot is taken.
    """

    source_system = "graphiti"
    replayable = True

    def __init__(
        self,
        graphiti: Any,
        group_ids: Optional[List[str]] = None,
        page_size: int = 500,
        mode: str = "hybrid",
        close_on_fetch: bool = False,
    ):
        super().__init__(mode=mode)
        self._graphiti = graphiti
        self._group_ids = group_ids
        self._page_size = page_size
        self._close_on_fetch = close_on_fetch
        self._snapshot: Optional[Dict[str, Any]] = None

    async def _ensure_snapshot(self) -> Dict[str, Any]:
        if self._snapshot is None:
            try:
                self._snapshot = await fetch_graphiti_snapshot(
                    self._graphiti,
                    group_ids=self._group_ids,
                    page_size=self._page_size,
                )
            finally:
                if self._close_on_fetch:
                    close = getattr(self._graphiti, "close", None)
                    if close is not None:
                        await call_maybe_async(close)
        return self._snapshot

    async def records(self) -> AsyncIterator[COGXRecord]:
        snapshot = await self._ensure_snapshot()
        async for record in iter_zep_graph_records(snapshot, source_system=self.source_system):
            yield record
