"""Live Zep Cloud memory source."""

from __future__ import annotations

from datetime import datetime
from typing import Any, AsyncIterator, Dict, List, Optional

from cognee.modules.migration.cogx import COGXRecord
from cognee.modules.migration.sources.base import MemorySource
from cognee.modules.migration.sources.live._utils import call_maybe_async, paginate_uuid_cursor
from cognee.modules.migration.sources.zep import iter_zep_graph_records


def _isoformat(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _zep_episode_dict(episode: Any, user_id: Optional[str] = None) -> Dict[str, Any]:
    return {
        "uuid": getattr(episode, "uuid_", None) or getattr(episode, "uuid", None),
        "name": getattr(episode, "name", None),
        "content": getattr(episode, "content", None) or getattr(episode, "episode_body", None),
        "created_at": _isoformat(getattr(episode, "created_at", None)),
        "valid_at": _isoformat(getattr(episode, "valid_at", None)),
        "group_id": getattr(episode, "group_id", None) or getattr(episode, "session_id", None),
        "user_id": user_id,
        "source_description": getattr(episode, "source_description", None),
    }


def _zep_node_dict(node: Any) -> Dict[str, Any]:
    labels = getattr(node, "labels", None) or getattr(node, "label", None)
    if labels is not None and not isinstance(labels, list):
        labels = [labels]
    return {
        "uuid": getattr(node, "uuid_", None) or getattr(node, "uuid", None),
        "name": getattr(node, "name", None),
        "labels": labels,
        "summary": getattr(node, "summary", None) or getattr(node, "description", None),
        "attributes": getattr(node, "attributes", None) or {},
        "created_at": _isoformat(getattr(node, "created_at", None)),
        "group_id": getattr(node, "group_id", None),
    }


def _zep_edge_dict(edge: Any) -> Dict[str, Any]:
    return {
        "uuid": getattr(edge, "uuid_", None) or getattr(edge, "uuid", None),
        "source_node_uuid": getattr(edge, "source_node_uuid", None),
        "target_node_uuid": getattr(edge, "target_node_uuid", None),
        "name": getattr(edge, "name", None) or getattr(edge, "relation", None),
        "fact": getattr(edge, "fact", None),
        "valid_at": _isoformat(getattr(edge, "valid_at", None)),
        "invalid_at": _isoformat(
            getattr(edge, "invalid_at", None) or getattr(edge, "expired_at", None)
        ),
        "created_at": _isoformat(getattr(edge, "created_at", None)),
        "episodes": getattr(edge, "episodes", None) or [],
        "group_id": getattr(edge, "group_id", None),
    }


async def fetch_zep_snapshot(
    client: Any,
    *,
    user_id: Optional[str] = None,
    graph_id: Optional[str] = None,
    episode_lastn: int = 10_000,
    page_size: int = 100,
) -> Dict[str, Any]:
    """Export Zep Cloud graph data into the ZepSource dict shape."""
    from cognee.modules.migration.sources.live._utils import require_extra

    require_extra("zep", "zep_cloud")

    if not user_id and not graph_id:
        raise ValueError("ZepLiveSource requires user_id or graph_id.")

    graph = client.graph
    if user_id:
        episode_resp = await call_maybe_async(
            graph.episode.get_by_user_id, user_id=user_id, lastn=episode_lastn
        )
        nodes = await paginate_uuid_cursor(
            lambda **kwargs: graph.node.get_by_user_id(user_id=user_id, **kwargs),
            page_size=page_size,
        )
        edges = await paginate_uuid_cursor(
            lambda **kwargs: graph.edge.get_by_user_id(user_id=user_id, **kwargs),
            page_size=page_size,
        )
    else:
        episode_resp = await call_maybe_async(
            graph.episode.get_by_graph_id, graph_id=graph_id, lastn=episode_lastn
        )
        nodes = await paginate_uuid_cursor(
            lambda **kwargs: graph.node.get_by_graph_id(graph_id=graph_id, **kwargs),
            page_size=page_size,
        )
        edges = await paginate_uuid_cursor(
            lambda **kwargs: graph.edge.get_by_graph_id(graph_id=graph_id, **kwargs),
            page_size=page_size,
        )

    episodes = getattr(episode_resp, "episodes", None) or episode_resp or []
    if not isinstance(episodes, list):
        episodes = []

    episode_dicts = [_zep_episode_dict(episode, user_id=user_id) for episode in episodes]
    edge_dicts = [_zep_edge_dict(edge) for edge in edges]

    # Fetch episodes referenced on edges but missing from the lastn window.
    known_uuids = {item["uuid"] for item in episode_dicts if item.get("uuid")}
    for edge in edge_dicts:
        for episode_uuid in edge.get("episodes") or []:
            if not episode_uuid or episode_uuid in known_uuids:
                continue
            try:
                episode = await call_maybe_async(graph.episode.get, uuid_=episode_uuid)
            except Exception:
                continue
            episode_dicts.append(_zep_episode_dict(episode, user_id=user_id))
            known_uuids.add(episode_uuid)

    return {
        "episodes": episode_dicts,
        "nodes": [_zep_node_dict(node) for node in nodes],
        "edges": edge_dicts,
    }


class ZepLiveSource(MemorySource):
    """Fetch Zep Cloud graph memory via an injected Zep client.

      Episode history is limited to the most recent ``episode_lastn`` entries
      because the Zep Cloud API does not expose episode cursor pagination.
    Nodes and edges are fully paginated.

      Args:
          client: ``zep_cloud.Zep`` or ``zep_cloud.client.AsyncZep`` instance.
          user_id: Zep user id to export (mutually exclusive with graph_id).
          graph_id: Zep graph id to export (mutually exclusive with user_id).
          episode_lastn: Maximum recent episodes to retrieve.
          page_size: Page size for node/edge uuid-cursor pagination.
          mode: Import mode (default ``hybrid``).
    """

    source_system = "zep"
    replayable = True

    def __init__(
        self,
        client: Any,
        user_id: Optional[str] = None,
        graph_id: Optional[str] = None,
        episode_lastn: int = 10_000,
        page_size: int = 100,
        mode: str = "hybrid",
    ):
        super().__init__(mode=mode)
        self._client = client
        self._user_id = user_id
        self._graph_id = graph_id
        self._episode_lastn = episode_lastn
        self._page_size = page_size
        self._snapshot: Optional[Dict[str, Any]] = None

    async def _ensure_snapshot(self) -> Dict[str, Any]:
        if self._snapshot is None:
            self._snapshot = await fetch_zep_snapshot(
                self._client,
                user_id=self._user_id,
                graph_id=self._graph_id,
                episode_lastn=self._episode_lastn,
                page_size=self._page_size,
            )
        return self._snapshot

    async def records(self) -> AsyncIterator[COGXRecord]:
        snapshot = await self._ensure_snapshot()
        async for record in iter_zep_graph_records(snapshot, source_system=self.source_system):
            yield record
