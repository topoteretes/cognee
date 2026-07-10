"""Sync recent graph knowledge into the session cache for fast retrieval.

Reads new edges incrementally (by created_at timestamp) so only edges added
since the last sync are processed. On ledger-backed graphs this walks the
relational Edge/Node tables; on graph-provenance graphs (provenance stored in the
graph, empty ledger) it reads the edges straight from the graph adapter via
``get_edges_created_since``. Either way the result is stored as a dedicated
graph knowledge key — separate from the QA conversation history — so it is
always available to session completions regardless of conversation length.

A ``max_lines`` cap prevents unbounded growth: when the merged snapshot
exceeds the limit, only the most recently created edges are kept.
"""

from datetime import datetime
from typing import Optional
from uuid import UUID

from sqlalchemy import select, and_

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.infrastructure.databases.unified import get_unified_engine
from cognee.infrastructure.databases.provenance.markers import stores_provenance_in_graph
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
        raw = await cache_engine.get_value(key)
        if raw:
            return datetime.fromisoformat(raw)
        return None
    except (NotImplementedError, AttributeError, TypeError):
        # Adapter predates the KV interface (missing, non-async, or
        # different-signature get_value), fall back to legacy duck-typing
        pass
    except Exception:
        logger.debug("_load_checkpoint: cache read failed for key %s", key)
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
        await cache_engine.set_value(key, value)
        return
    except (NotImplementedError, AttributeError, TypeError):
        # Adapter predates the KV interface (missing, non-async, or
        # different-signature set_value), fall back to legacy duck-typing
        pass
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


def _node_entry(meta: dict) -> dict:
    """Build the per-endpoint JSON entry from a node metadata dict.

    Works for both a relational ``Node`` (label/type) and a graph node
    (name/type), preferring the most human-readable label available.
    """
    label = meta.get("label") or meta.get("name") or meta.get("type") or str(meta.get("id"))
    entry = {"label": label}
    if meta.get("type"):
        entry["type"] = meta["type"]
    if meta.get("description"):
        entry["description"] = meta["description"]
    return entry


def _triplet_line(src_meta: dict, relationship: Optional[str], dst_meta: dict) -> str:
    """Render a triplet as a structured JSON-line with node metadata.

    Preserves node type and description alongside the relationship, giving LLMs
    richer context than plain ``src —[rel]→ dst`` strings.
    """
    import json

    entry = {
        "source": _node_entry(src_meta),
        "relationship": relationship or "related_to",
        "target": _node_entry(dst_meta),
    }
    return json.dumps(entry, ensure_ascii=False)


def _relational_node_meta(node: Node) -> dict:
    return {
        "id": node.id,
        "label": node.label,
        "type": getattr(node, "type", None),
        "description": getattr(node, "description", None),
    }


def _edge_to_text(edge: Edge, node_map: dict) -> Optional[str]:
    """Render a relational ``Edge`` as a triplet JSON-line, or None if an
    endpoint node is missing."""
    src = node_map.get(edge.source_node_id)
    dst = node_map.get(edge.destination_node_id)
    if not src or not dst:
        return None
    return _triplet_line(
        _relational_node_meta(src), edge.relationship_name, _relational_node_meta(dst)
    )


async def _collect_relational_lines(db_engine, dataset_id: UUID, since: Optional[datetime]):
    """Walk the relational Edge ledger in created_at order, returning
    (triplet_lines, latest_timestamp)."""
    lines: list[str] = []
    latest = since
    while True:
        edges, node_map = await _fetch_new_edges(db_engine, dataset_id, latest, BATCH_SIZE)
        if not edges:
            break
        for edge in edges:
            text = _edge_to_text(edge, node_map)
            if text:
                lines.append(text)
            if latest is None or edge.created_at > latest:
                latest = edge.created_at
        if len(edges) < BATCH_SIZE:
            break
    return lines, latest


async def _collect_graph_provenance_lines(graph_engine, since: Optional[datetime]):
    """Walk new graph edges (by created_at) on a graph-provenance graph, returning
    (triplet_lines, latest_timestamp). Mirrors the relational collector but reads
    provenance-carrying edges straight from the graph instead of the ledger.

    Pages with a keyset cursor (created_at plus edge identity), not the timestamp
    alone: the graph adapters stamp every edge of one add_edges batch with the
    same created_at, so a timestamp-only cursor would permanently skip the rest
    of a tie group whenever a page boundary lands inside one."""
    lines: list[str] = []
    latest = since
    after_key = None
    while True:
        edges, node_map = await graph_engine.get_edges_created_since(
            latest, BATCH_SIZE, after_key=after_key
        )
        if not edges:
            break
        for source_id, target_id, relationship, created_at in edges:
            src = node_map.get(source_id)
            dst = node_map.get(target_id)
            if src and dst:
                lines.append(_triplet_line(src, relationship, dst))
        # Pages are totally ordered, so the last row is both the newest timestamp
        # and the exact resume point inside its tie group.
        last_source, last_target, last_relationship, latest = edges[-1]
        after_key = (last_source, last_target, last_relationship)
        if len(edges) < BATCH_SIZE:
            break
    return lines, latest


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

    # Collect new triplet lines (ordered by created_at ascending). Graph-provenance
    # graphs carry their edges in the graph (the relational Edge ledger is empty),
    # so read new edges straight from the graph adapter; otherwise walk the ledger.
    unified = await get_unified_engine()
    stores_provenance = (
        unified.supports_graph_provenance_delete()
        and await stores_provenance_in_graph(unified.graph)
    )
    if stores_provenance:
        new_lines, latest_ts = await _collect_graph_provenance_lines(unified.graph, since)
    else:
        db_engine = get_relational_engine()
        new_lines, latest_ts = await _collect_relational_lines(db_engine, dataset_id, since)

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
