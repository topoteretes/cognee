"""Public dataset-versioning API (issue #3650, Approach 1: run-ledger time-travel).

Additive surface over the COG-5522 provenance substrate:

    await cognee.snapshot("before-import", dataset="my_dataset")
    nodes, edges = await cognee.graph_as_of("before-import", dataset="my_dataset")
    hits = await cognee.search_as_of("query", "before-import", dataset="my_dataset")
    result = await cognee.rollback("before-import", dataset="my_dataset")
    await cognee.undo(result["operation_id"], dataset="my_dataset")

Reversible forget is exposed on ``cognee.forget(..., memory_only=True,
reversible=True)``, which returns the ledger ``operation_id`` consumed by
``cognee.undo``. Nothing here changes existing behavior when unused.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import UUID

from cognee.context_global_variables import set_database_global_context_variables
from cognee.modules import versioning
from cognee.shared.logging_utils import get_logger

logger = get_logger("version")


async def _resolve_user(user: Any):
    if user is not None:
        return user
    from cognee.modules.users.methods import get_default_user

    return await get_default_user()


async def _resolve_dataset(
    dataset: Optional[str], dataset_id: Optional[UUID], user: Any, permission: str
):
    """Resolve a dataset name or UUID to an authorized Dataset object."""
    if dataset and dataset_id:
        raise ValueError("Provide either dataset or dataset_id, not both.")
    if not dataset and not dataset_id:
        raise ValueError("Provide dataset or dataset_id.")

    if dataset_id is not None:
        from cognee.modules.data.methods.get_authorized_dataset import get_authorized_dataset

        resolved = await get_authorized_dataset(user, dataset_id, permission)
        if not resolved:
            raise ValueError(f"Dataset {dataset_id} not found or not accessible.")
        return resolved

    from cognee.modules.data.methods import get_authorized_dataset_by_name

    return await get_authorized_dataset_by_name(dataset, user, permission)


async def snapshot(
    name: str,
    *,
    dataset: Optional[str] = None,
    dataset_id: Optional[UUID] = None,
    user: Any = None,
) -> Dict[str, Any]:
    """Label the dataset's current position in the run ledger. Copies nothing."""
    user = await _resolve_user(user)
    resolved = await _resolve_dataset(dataset, dataset_id, user, "write")

    created = await versioning.create_snapshot(resolved.id, name)
    return {
        "id": str(created.id),
        "name": created.name,
        "dataset_id": str(created.dataset_id),
        "as_of_time": created.as_of_time.isoformat(),
        "latest_pipeline_run_id": (
            str(created.latest_pipeline_run_id) if created.latest_pipeline_run_id else None
        ),
    }


async def list_snapshots(
    *,
    dataset: Optional[str] = None,
    dataset_id: Optional[UUID] = None,
    user: Any = None,
) -> List[Dict[str, Any]]:
    user = await _resolve_user(user)
    resolved = await _resolve_dataset(dataset, dataset_id, user, "read")

    return [
        {
            "id": str(item.id),
            "name": item.name,
            "dataset_id": str(item.dataset_id),
            "as_of_time": item.as_of_time.isoformat(),
            "latest_pipeline_run_id": (
                str(item.latest_pipeline_run_id) if item.latest_pipeline_run_id else None
            ),
        }
        for item in await versioning.list_snapshots(resolved.id)
    ]


async def graph_as_of(
    as_of: Union[str, datetime],
    *,
    dataset: Optional[str] = None,
    dataset_id: Optional[UUID] = None,
    user: Any = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """The dataset's visible subgraph at ``as_of`` (snapshot name or datetime).

    Forward-faithful read: artifacts destroyed by an un-undone forget or
    rollback executed *after* ``as_of`` are gone from the live store and are
    not resurrected by this filter (see modules/versioning/methods/as_of_read).
    """
    from cognee.infrastructure.databases.graph import get_graph_engine

    user = await _resolve_user(user)
    resolved = await _resolve_dataset(dataset, dataset_id, user, "read")

    async with set_database_global_context_variables(resolved.id, resolved.owner_id):
        graph_engine = await get_graph_engine()
        return await versioning.get_graph_as_of(graph_engine, resolved.id, as_of)


async def search_as_of(
    query_text: str,
    as_of: Union[str, datetime],
    *,
    dataset: Optional[str] = None,
    dataset_id: Optional[UUID] = None,
    top_k: int = 15,
    user: Any = None,
) -> List[Any]:
    """Chunk search over the dataset as it existed at ``as_of``."""
    from cognee.infrastructure.databases.graph import get_graph_engine
    from cognee.infrastructure.databases.vector import get_vector_engine

    user = await _resolve_user(user)
    resolved = await _resolve_dataset(dataset, dataset_id, user, "read")

    async with set_database_global_context_variables(resolved.id, resolved.owner_id):
        graph_engine = await get_graph_engine()
        vector_engine = get_vector_engine()
        return await versioning.search_chunks_as_of(
            graph_engine, vector_engine, resolved.id, query_text, as_of, top_k=top_k
        )


async def rollback(
    to: Union[str, datetime],
    *,
    dataset: Optional[str] = None,
    dataset_id: Optional[UUID] = None,
    user: Any = None,
) -> Dict[str, Any]:
    """Reverse every run completed after ``to`` (snapshot name or datetime).

    Uses the existing run rollback primitive, newest run first, and records
    the whole rollback as one undoable ledger operation; the returned
    ``operation_id`` feeds ``cognee.undo``.
    """
    user = await _resolve_user(user)
    resolved = await _resolve_dataset(dataset, dataset_id, user, "delete")

    async with set_database_global_context_variables(resolved.id, resolved.owner_id):
        return await versioning.rollback_dataset_to(resolved, to, user)


async def undo(
    operation_id: Union[str, UUID],
    *,
    dataset: Optional[str] = None,
    dataset_id: Optional[UUID] = None,
    user: Any = None,
) -> Dict[str, Any]:
    """Undo a ledgered forget or rollback within the retention window."""
    user = await _resolve_user(user)

    op_id = operation_id if isinstance(operation_id, UUID) else UUID(str(operation_id))

    if dataset is None and dataset_id is None:
        # The ledger row knows its dataset; authorize against that.
        op = await versioning.get_version_op(op_id)
        dataset_id = op.dataset_id

    resolved = await _resolve_dataset(dataset, dataset_id, user, "delete")

    async with set_database_global_context_variables(resolved.id, resolved.owner_id):
        return await versioning.undo_version_op(op_id)
