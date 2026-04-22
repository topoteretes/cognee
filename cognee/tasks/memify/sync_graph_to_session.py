"""Sync recent graph knowledge into the session cache for fast retrieval.

Queries the relational Edge/Node tables incrementally (by created_at
timestamp) so only *new* edges since the last sync are processed.
The result is stored as a dedicated graph knowledge key — separate
from the QA conversation history — so it is always available to
session completions regardless of conversation length.

A ``max_lines`` cap prevents unbounded growth: when the merged snapshot
exceeds the limit, only the most recently created edges are kept.
"""

from datetime import datetime
from typing import Optional
from uuid import UUID

from sqlalchemy import select, and_

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.graph.models.Edge import Edge
from cognee.modules.graph.models.Node import Node
from cognee.shared.logging_utils import get_logger
from cognee.modules.observability import (
    new_span,
    COGNEE_SESSION_ID,
    COGNEE_DATASET_NAME,
    COGNEE_GRAPH_EDGES_SYNCED,
)

logger = get_logger("sync_graph_to_session")

_CHECKPOINT_KEY_PREFIX = "graph_sync_checkpoint"

BATCH_SIZE = 500
DEFAULT_MAX_LINES = 500


def _checkpoint_key(user_id: str, dataset_id: str, session_id: str) -> str:
    return f"{_CHECKPOINT_KEY_PREFIX}:{user_id}:{dataset_id}:{session_id}"


async def _load_checkpoint(cache_engine, key: str) -> Optional[datetime]:
    """Read the last-synced timestamp from the cache engine."""
    if cache_engine is None:
        return None
    try:
        raw = await cache_engine.async_redis.get(key)
        if raw:
            return datetime.fromisoformat(raw.decode() if isinstance(raw, bytes) else raw)
    except AttributeError:
        try:
            raw = cache_engine._cache.get(key)
            if raw:
                return datetime.fromisoformat(raw)
        except Exception:
            logger.debug("_load_checkpoint: FsCache read failed for key %s", key)
    except Exception:
        logger.debug("_load_checkpoint: Redis read failed for key %s", key)
    return None


async def _save_checkpoint(cache_engine, key: str, ts: datetime) -> None:
    """Persist the high-water mark timestamp."""
    value = ts.isoformat()
    try:
        await cache_engine.async_redis.set(key, value)
    except AttributeError:
        try:
            cache_engine._cache.set(key, value)
        except Exception:
            logger.debug("_save_checkpoint: FsCache write failed for key %s", key)


async def _fetch_new_edges(
    db_engine,
    dataset_id: UUID,
    since: Optional[datetime],
    limit: int = BATCH_SIZE,
):
    """Fetch edges created after *since* for the given dataset.

    Returns (edges, node_map) where node_map contains all referenced nodes.
    """
    async with db_engine.get_async_session() as session:
        conditions = [Edge.dataset_id == dataset_id]
        if since is not None:
            conditions.append(Edge.created_at > since)

        edge_query = (
            select(Edge).where(and_(*conditions)).order_by(Edge.created_at.asc()).limit(limit)
        )
        edges = (await session.scalars(edge_query)).all()

        if not edges:
            return [], {}

        node_ids = set()
        for e in edges:
            node_ids.add(e.source_node_id)
            node_ids.add(e.destination_node_id)

        node_query = select(Node).where(Node.id.in_(node_ids))
        nodes = (await session.scalars(node_query)).all()
        node_map = {n.id: n for n in nodes}

        return edges, node_map


def _edge_to_text(edge: Edge, node_map: dict) -> Optional[str]:
    """Render an edge as a structured JSON-line with node metadata.

    Preserves node type and description alongside the triplet relationship,
    giving LLMs richer context than plain ``src —[rel]→ dst`` strings.
    """
    import json

    src = node_map.get(edge.source_node_id)
    dst = node_map.get(edge.destination_node_id)
    if not src or not dst:
        return None

    src_entry = {"label": src.label or src.type or str(src.id)}
    dst_entry = {"label": dst.label or dst.type or str(dst.id)}

    if getattr(src, "type", None):
        src_entry["type"] = src.type
    if getattr(dst, "type", None):
        dst_entry["type"] = dst.type
    if getattr(src, "description", None):
        src_entry["description"] = src.description
    if getattr(dst, "description", None):
        dst_entry["description"] = dst.description

    entry = {
        "source": src_entry,
        "relationship": edge.relationship_name or "related_to",
        "target": dst_entry,
    }
    return json.dumps(entry, ensure_ascii=False)


async def sync_graph_to_session(
    *,
    user_id: str,
    session_id: str,
    dataset_id: UUID,
    dataset_name: str = "main_dataset",
    max_lines: int = DEFAULT_MAX_LINES,
) -> dict:
    """Incrementally sync recent graph edges into the session's graph knowledge context.

    Maintains a single deduplicated knowledge snapshot per session via
    ``SessionManager.set_graph_context()``. New triplets are merged with
    the existing snapshot. When the total exceeds ``max_lines``, the
    oldest lines are dropped to stay within budget.

    Returns a summary dict with the count of newly synced edges.
    """
    from cognee.infrastructure.databases.cache.get_cache_engine import get_cache_engine
    from cognee.infrastructure.session.get_session_manager import get_session_manager

    cache_engine = get_cache_engine()
    sm = get_session_manager()
    if not sm.is_available:
        logger.warning("sync_graph_to_session: session cache not available, skipping")
        return {"synced": 0}

    ck = _checkpoint_key(user_id, str(dataset_id), session_id)
    since = await _load_checkpoint(cache_engine, ck)
    logger.info(
        "sync_graph_to_session: dataset=%s since=%s",
        dataset_name,
        since.isoformat() if since else "beginning",
    )

    # Collect new triplet lines (ordered by created_at ascending)
    db_engine = get_relational_engine()
    new_lines: list[str] = []
    latest_ts = since

    while True:
        edges, node_map = await _fetch_new_edges(db_engine, dataset_id, latest_ts, BATCH_SIZE)
        if not edges:
            break

        for edge in edges:
            text = _edge_to_text(edge, node_map)
            if text:
                new_lines.append(text)
            if latest_ts is None or edge.created_at > latest_ts:
                latest_ts = edge.created_at

        if len(edges) < BATCH_SIZE:
            break

    if not new_lines:
        logger.info("sync_graph_to_session: no new edges to sync")
        return {"synced": 0}

    # Merge: existing (older) + new (newer), then cap at max_lines
    # keeping the tail (most recent) when over budget
    existing = await sm.get_graph_context(user_id=user_id, session_id=session_id)
    existing_lines = [line for line in existing.split("\n") if line] if existing else []
    merged = existing_lines + new_lines
    if len(merged) > max_lines:
        logger.info(
            "sync_graph_to_session: capping at %d lines (had %d), dropping %d oldest",
            max_lines,
            len(merged),
            len(merged) - max_lines,
        )
        merged = merged[-max_lines:]

    await sm.set_graph_context(
        user_id=user_id,
        session_id=session_id,
        context="\n".join(merged),
    )

    if latest_ts and latest_ts != since:
        await _save_checkpoint(cache_engine, ck, latest_ts)

    with new_span("cognee.task.sync_graph_to_session") as span:
        span.set_attribute(COGNEE_SESSION_ID, session_id)
        span.set_attribute(COGNEE_DATASET_NAME, dataset_name)
        span.set_attribute(COGNEE_GRAPH_EDGES_SYNCED, len(new_lines))

    logger.info(
        "sync_graph_to_session: synced %d new edges (total %d, cap %d)",
        len(new_lines),
        len(merged),
        max_lines,
    )
    return {"synced": len(new_lines), "total": len(merged)}
