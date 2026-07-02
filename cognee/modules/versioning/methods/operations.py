"""Reversible destructive operations: forget capture, rollback-to-T, and undo.

Every destructive operation follows the same write-ahead discipline:

1. compute exactly what the existing delete path will remove (same inputs the
   planner receives — never a forked delete),
2. commit the inverse to the ``version_ops`` ledger,
3. run the *existing* destructive primitive
   (``UnifiedStoreEngine.delete_by_* `` / ``rollback_by_pipeline_run_id`` via
   their established callers),
4. mark the op APPLIED.

Undo replays the committed inverse steps in reverse application order. All
restore primitives are idempotent, so a crash anywhere in 2-4 or during undo
is recovered by replaying.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import attributes as orm_attributes

from cognee.infrastructure.databases.exceptions import UnsupportedProvenanceCapability
from cognee.infrastructure.databases.provenance import make_source_ref_key
from cognee.infrastructure.databases.provenance.markers import stores_provenance_in_graph
from cognee.infrastructure.databases.provenance.source_refs import (
    get_data_id_from_source_ref_key,
)
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.infrastructure.databases.unified import get_unified_engine
from cognee.modules.data.models import Data
from cognee.modules.versioning.methods.inverse import (
    capture_source_ref_removal_inverse,
    restore_inverse_step,
)
from cognee.modules.versioning.methods.ledger import (
    append_op_step,
    assert_within_retention,
    create_version_op,
    get_version_op,
    set_op_status,
)
from cognee.modules.versioning.methods.timeline import (
    get_run_ids_after,
    resolve_as_of_time,
)
from cognee.modules.versioning.models import VersionOpStatus
from cognee.shared.logging_utils import get_logger

logger = get_logger("versioning.operations")


async def _get_provenance_engines():
    """The unified engine's graph/vector pair, or raise when unsupported.

    Reversibility requires graph-provenance mode: the inverse is captured from
    (and restored into) the provenance columns the delete path maintains.
    Backends without that capability fail closed with a typed error instead of
    silently losing data.
    """
    unified = await get_unified_engine()
    if not unified.supports_graph_provenance_delete():
        raise UnsupportedProvenanceCapability()
    if not await stores_provenance_in_graph(unified.graph):
        raise UnsupportedProvenanceCapability()
    return unified.graph, unified.vector


def _data_ids_from_refs(
    refs_by_node: Dict[str, List[str]], refs_by_edge: Dict[Any, List[str]]
) -> set:
    data_ids = set()
    for refs in list(refs_by_node.values()) + list(refs_by_edge.values()):
        for key in refs:
            data_ids.add(get_data_id_from_source_ref_key(key))
    return data_ids


async def _snapshot_pipeline_status(data_ids: set, dataset_id: UUID) -> Dict[str, Dict[str, Any]]:
    """Capture each data item's per-dataset pipeline status before it is reset."""
    if not data_ids:
        return {}

    dataset_id_str = str(dataset_id)
    snapshot: Dict[str, Dict[str, Any]] = {}

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        data_records = (
            (await session.execute(select(Data).where(Data.id.in_(list(data_ids))))).scalars().all()
        )
        for record in data_records:
            statuses = {}
            for pipeline_name, per_dataset in (record.pipeline_status or {}).items():
                if dataset_id_str in per_dataset:
                    statuses[pipeline_name] = per_dataset[dataset_id_str]
            if statuses:
                snapshot[str(record.id)] = statuses

    return snapshot


async def _restore_pipeline_status(snapshot: Dict[str, Dict[str, Any]], dataset_id: UUID) -> None:
    if not snapshot:
        return

    dataset_id_str = str(dataset_id)

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        data_records = (
            (
                await session.execute(
                    select(Data).where(Data.id.in_([UUID(key) for key in snapshot.keys()]))
                )
            )
            .scalars()
            .all()
        )
        for record in data_records:
            statuses = snapshot.get(str(record.id), {})
            if not statuses:
                continue
            pipeline_status = record.pipeline_status or {}
            for pipeline_name, value in statuses.items():
                pipeline_status.setdefault(pipeline_name, {})[dataset_id_str] = value
            record.pipeline_status = pipeline_status
            orm_attributes.flag_modified(record, "pipeline_status")
        await session.commit()


async def capture_forget_inverse(dataset_id: UUID, data_id: Optional[UUID] = None) -> UUID:
    """Commit the inverse of an imminent memory-forget; returns the op id.

    Must be called *before* the forget executes. Mirrors the exact discovery
    the unified delete path performs (``find_*_by_source_ref`` for a data
    item, ``find_*_source_refs_by_dataset`` for a whole dataset), so the
    captured inverse covers precisely what the delete will remove.
    """
    graph, vector = await _get_provenance_engines()

    if data_id is not None:
        source_ref_key = make_source_ref_key(dataset_id, data_id)
        node_ids = await graph.find_nodes_by_source_ref(source_ref_key)
        edges = await graph.find_edges_by_source_ref(source_ref_key)
        refs_by_node = {node_id: [source_ref_key] for node_id in node_ids}
        refs_by_edge = {edge: [source_ref_key] for edge in edges}
    else:
        refs_by_node = await graph.find_node_source_refs_by_dataset(str(dataset_id))
        refs_by_edge = await graph.find_edge_source_refs_by_dataset(str(dataset_id))

    step = await capture_source_ref_removal_inverse(
        graph, vector, refs_by_node=refs_by_node, refs_by_edge=refs_by_edge
    )
    step["pipeline_status"] = await _snapshot_pipeline_status(
        _data_ids_from_refs(refs_by_node, refs_by_edge), dataset_id
    )

    op_id = await create_version_op(
        dataset_id,
        "forget",
        steps=[step],
        extra={
            "target": "data_item_memory" if data_id is not None else "dataset_memory",
            "data_id": str(data_id) if data_id is not None else None,
        },
    )

    logger.info(
        "Captured forget inverse op=%s (dataset=%s, data_id=%s): %d nodes, %d edges.",
        op_id,
        dataset_id,
        data_id,
        len(refs_by_node),
        len(refs_by_edge),
    )
    return op_id


async def mark_forget_applied(op_id: UUID) -> None:
    await set_op_status(op_id, VersionOpStatus.APPLIED)


async def rollback_dataset_to(
    dataset: Any, as_of: Union[str, datetime], user: Any = None
) -> Dict[str, Any]:
    """Reverse every run completed after ``as_of``, newest first, reversibly.

    Each run is undone with the existing rollback primitive
    (``cognify_rollback_handler`` -> ``rollback_by_pipeline_run_id``); this
    orchestration adds only the write-ahead inverse capture so the rollback is
    itself an undoable ledger entry. Note the Approach-1 boundary: rollback
    reverses *run contributions*; a forget executed after T is a separate
    ledger op and is undone via its own op id, not by rollback.
    """
    # Imported here: modules.cognify pulls the graph engine stack at import
    # time, which must not become an import-time dependency of versioning.
    from cognee.modules.cognify.rollback import cognify_rollback_handler

    dataset_id = dataset.id

    as_of_time = await resolve_as_of_time(dataset_id, as_of)
    run_ids = await get_run_ids_after(dataset_id, as_of_time)

    if not run_ids:
        return {"operation_id": None, "rolled_back_runs": [], "status": "noop"}

    graph, vector = await _get_provenance_engines()

    op_id = await create_version_op(
        dataset_id,
        "rollback",
        extra={"as_of_time": as_of_time.isoformat(), "candidate_run_ids": run_ids},
    )

    rolled_back_runs: List[str] = []
    for run_id in run_ids:  # newest first
        refs_by_node = await graph.find_node_source_refs_by_pipeline_run(run_id)
        refs_by_edge = await graph.find_edge_source_refs_by_pipeline_run(run_id)

        if not refs_by_node and not refs_by_edge:
            # Runs that attached no graph artifacts (e.g. add-pipeline runs, or
            # runs whose ownership was later superseded) have nothing to
            # reverse; skip them rather than record empty ledger steps.
            continue

        step = await capture_source_ref_removal_inverse(
            graph, vector, refs_by_node=refs_by_node, refs_by_edge=refs_by_edge
        )
        step["run_id"] = run_id
        step["pipeline_status"] = await _snapshot_pipeline_status(
            _data_ids_from_refs(refs_by_node, refs_by_edge), dataset_id
        )

        # Write-ahead: the step is committed before its destructive execution.
        await append_op_step(op_id, step)

        await cognify_rollback_handler(UUID(run_id), dataset, user)
        rolled_back_runs.append(run_id)

    await set_op_status(op_id, VersionOpStatus.APPLIED)

    logger.info(
        "Rolled back dataset %s to %s (%d of %d candidate runs) as op=%s.",
        dataset_id,
        as_of_time.isoformat(),
        len(rolled_back_runs),
        len(run_ids),
        op_id,
    )
    return {
        "operation_id": str(op_id),
        "rolled_back_runs": rolled_back_runs,
        "status": "success",
    }


async def undo_version_op(op_id: UUID) -> Dict[str, Any]:
    """Replay a ledgered op's inverse (newest step first) and mark it UNDONE."""
    op = await get_version_op(op_id)

    if op.status == VersionOpStatus.UNDONE:
        raise ValueError(f"Version op {op_id} has already been undone.")
    assert_within_retention(op)

    graph, vector = await _get_provenance_engines()

    steps = op.payload.get("steps", [])
    # Reverse application order: the last-applied step is restored first.
    for step in reversed(steps):
        await restore_inverse_step(graph, vector, step)
        await _restore_pipeline_status(step.get("pipeline_status", {}), op.dataset_id)

    await set_op_status(op_id, VersionOpStatus.UNDONE)

    logger.info("Undid version op %s (%s, %d steps).", op_id, op.op_type, len(steps))
    return {"operation_id": str(op_id), "op_type": op.op_type, "status": "undone"}
