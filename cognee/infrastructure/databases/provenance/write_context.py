from __future__ import annotations

from typing import Any
from uuid import UUID

from cognee.infrastructure.databases.provenance.markers import stores_provenance_in_graph
from cognee.infrastructure.databases.provenance.source_refs import make_source_ref_key


def _value_id(value: Any) -> Any:
    if value is None:
        return None
    return getattr(value, "id", value)


def _context_value(ctx: Any, name: str) -> Any:
    if ctx is None:
        return None
    return getattr(ctx, name, None)


def _context_data_id(ctx: Any) -> Any:
    data_item = _context_value(ctx, "data_item")
    if data_item is None:
        return None
    return getattr(data_item, "id", None) or getattr(data_item, "data_id", None)


def _coerce_uuid(value: Any) -> UUID:
    if isinstance(value, UUID):
        return value
    return UUID(str(value))


def source_ref_from_context(
    ctx: Any = None,
    *,
    dataset_id: Any = None,
    data_id: Any = None,
    fallback_data_id: Any = None,
    pipeline_run_id: Any = None,
) -> tuple[str | None, str | None]:
    """Build write-time provenance args from pipeline context or explicit ids."""
    dataset_id = dataset_id if dataset_id is not None else _value_id(_context_value(ctx, "dataset"))
    data_id = data_id if data_id is not None else (_context_data_id(ctx) or fallback_data_id)
    pipeline_run_id = (
        pipeline_run_id if pipeline_run_id is not None else _context_value(ctx, "pipeline_run_id")
    )

    if dataset_id is None or data_id is None:
        return None, None

    return make_source_ref_key(_coerce_uuid(dataset_id), _coerce_uuid(data_id)), (
        str(pipeline_run_id) if pipeline_run_id else None
    )


async def graph_provenance_write_kwargs(
    graph_engine: Any,
    ctx: Any = None,
    *,
    dataset_id: Any = None,
    data_id: Any = None,
    fallback_data_id: Any = None,
    pipeline_run_id: Any = None,
) -> dict[str, str | None]:
    """Return folded provenance kwargs for graph writes, but only when the graph
    *already* stores its provenance in-graph.

    This never marks or otherwise mutates the graph: deciding a graph's mode is
    the ingestion path's job (``add_data_points`` marks an empty graph on the
    first write). A secondary write site only stamps when the graph is already
    graph-provenance; on a ledger graph (or before the graph has been marked) it
    returns no provenance and the write is left unstamped.
    """
    source_ref_key, run_id = source_ref_from_context(
        ctx,
        dataset_id=dataset_id,
        data_id=data_id,
        fallback_data_id=fallback_data_id,
        pipeline_run_id=pipeline_run_id,
    )
    if source_ref_key is None:
        return {"source_ref_key": None, "pipeline_run_id": None}

    if not await stores_provenance_in_graph(graph_engine):
        return {"source_ref_key": None, "pipeline_run_id": None}

    return {"source_ref_key": source_ref_key, "pipeline_run_id": run_id}
