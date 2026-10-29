"""Dataset-authorized conflict decisions, serialized with ingestion and improve.

Like the memory API and the other remote-aware entry points, both functions go to
the remote instance once ``cognee.serve()`` is connected: the dataset and its
conflicts live there. On that path the API key identifies the caller, so ``user``
is ignored, and HTTP errors surface as RuntimeError carrying the status.
"""

from datetime import datetime, timezone
from typing import Literal
from uuid import UUID

from cognee.context_global_variables import set_database_global_context_variables
from cognee.exceptions import CogneeValidationError
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.locks.dataset_lock import dataset_lock
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.graph.exceptions.exceptions import EntityNotFoundError
from cognee.modules.graph.utils.fact_conflicts import read_conflict_marks
from cognee.modules.operations import record_operation
from cognee.modules.pipelines.models import PipelineContext
from cognee.modules.session_lifecycle.invalidate_sessions import (
    invalidate_sessions_for_deleted_data,
)
from cognee.modules.users.exceptions import PermissionDeniedError
from cognee.modules.users.methods import get_default_user
from cognee.modules.users.models import User
from cognee.tasks.memify.review_conflicts.models import (
    AcceptedConflict,
    ConflictsToWrite,
)
from cognee.tasks.memify.review_conflicts.read_facts import _to_conflict_record, read_entity_facts
from cognee.tasks.memify.review_conflicts.write_conflicts import (
    _rebuild_stored_conflict,
    write_conflicts,
)

from .models import ConflictDetail, ConflictFact


def _fact_text(fact: dict, nodes: dict) -> str:
    return fact["properties"].get("edge_text") or " ".join(
        (
            nodes.get(fact["source"], {}).get("name", fact["source"]),
            fact["relationship"].replace("_", " "),
            nodes.get(fact["target"], {}).get("name", fact["target"]),
        )
    )


def _conflict_facts(conflict_id: str, scope) -> dict[str, str]:
    return {
        fact_id: mark["status"]
        for fact_id, fact in scope.facts.items()
        for mark in read_conflict_marks(fact["properties"])
        if str(mark.get("conflict_id")) == conflict_id
    }


def conflict_detail(stored: dict, scope) -> ConflictDetail:
    facts = [
        ConflictFact(
            id=UUID(fact_id),
            text=_fact_text(scope.facts[fact_id], scope.nodes),
            source_id=scope.facts[fact_id]["source"],
            target_id=scope.facts[fact_id]["target"],
            relationship=scope.facts[fact_id]["relationship"],
            status=status,
            sources=scope.facts[fact_id]["sources"],
        )
        for fact_id, status in sorted(_conflict_facts(str(stored["id"]), scope).items())
    ]
    return ConflictDetail.model_validate({**stored, "facts": facts})


async def _authorized_dataset(dataset_id: UUID, user: User | None, permission: str):
    user = user or await get_default_user()
    datasets = await get_authorized_existing_datasets([dataset_id], permission, user)
    if not datasets:
        raise PermissionDeniedError(message="Not authorized for this dataset")
    return user, datasets[0]


async def list_conflicts(
    dataset_id: UUID,
    *,
    status: Literal["resolved", "unresolved"] | None = None,
    user: User | None = None,
) -> list[ConflictDetail]:
    """List a readable dataset's conflicts with selectable fact ids and source evidence."""
    if status not in (None, "resolved", "unresolved"):
        raise CogneeValidationError(
            message="status must be resolved or unresolved", status_code=422
        )
    from cognee.api.v1.serve.state import get_remote_client

    client = get_remote_client()
    if client is not None:
        return await client.list_conflicts(dataset_id, status=status)
    user, dataset = await _authorized_dataset(dataset_id, user, "read")
    async with set_database_global_context_variables(dataset.id, dataset.owner_id):
        graph = await get_graph_engine()
        rows, _ = await graph.get_filtered_graph_data(
            [{"type": ["FactConflict"]}], include_edges=False
        )
        conflicts = [
            _to_conflict_record(node_id, props)
            for node_id, props in rows
            if str(props.get("dataset_id")) == str(dataset.id)
            and (status is None or props.get("status") == status)
        ]
        if not conflicts:
            return []
        scope = await read_entity_facts(
            None,
            entity_ids=sorted({stored["about_id"] for stored in conflicts}),
            ctx=PipelineContext(user=user, dataset=dataset),
        )
        return [
            conflict_detail(stored, scope) for stored in sorted(conflicts, key=lambda c: c["id"])
        ]


async def resolve_conflict(
    conflict_id: UUID,
    *,
    dataset_id: UUID,
    selected_fact_id: UUID,
    reason: str | None = None,
    user: User | None = None,
) -> ConflictDetail:
    """Choose an existing competing fact; keep rejected facts as historical evidence.

    Repeating a choice over the same facts is idempotent; repeated after a review
    listed new evidence, it also rejects that evidence. A new choice explicitly
    revises the prior decision. Automatic improve preserves the choice while its
    selected fact exists, and never rejects evidence the user did not see.
    """
    from cognee.api.v1.serve.state import get_remote_client

    client = get_remote_client()
    if client is not None:
        return await client.resolve_conflict(
            conflict_id,
            dataset_id=dataset_id,
            selected_fact_id=selected_fact_id,
            reason=reason,
        )
    user, dataset = await _authorized_dataset(dataset_id, user, "write")
    conflict_id, selected_fact_id = str(conflict_id), str(selected_fact_id)
    async with record_operation("resolve_conflict") as operation:
        operation.set_user(user)
        operation.set_dataset(dataset.id)
        async with (
            dataset_lock(dataset.id),
            set_database_global_context_variables(dataset.id, dataset.owner_id),
        ):
            graph = await get_graph_engine()
            rows, _ = await graph.get_filtered_graph_data(
                [{"id": [conflict_id]}], include_edges=False
            )
            stored = next(
                (
                    _to_conflict_record(node_id, props)
                    for node_id, props in rows
                    if str(node_id) == conflict_id
                    and props.get("type") == "FactConflict"
                    and str(props.get("dataset_id")) == str(dataset.id)
                ),
                None,
            )
            if stored is None:
                raise EntityNotFoundError(message="Conflict not found in this dataset")
            ctx = PipelineContext(user=user, dataset=dataset)
            scope = await read_entity_facts(None, entity_ids=[stored["about_id"]], ctx=ctx)
            statuses = _conflict_facts(conflict_id, scope)
            if selected_fact_id not in statuses:
                raise CogneeValidationError(
                    message="selected_fact_id must be an existing fact belonging to this conflict",
                    status_code=422,
                )
            rejected_fact_ids = sorted(fid for fid in statuses if fid != selected_fact_id)
            # The same choice over the same competing facts changes nothing. A fact a review
            # listed since makes it a new decision, one that rejects that fact as well.
            same_choice = (
                stored.get("resolution_source") == "user"
                and stored.get("selected_fact_id") == selected_fact_id
                and stored.get("resolution_reason") == reason
                and sorted(stored.get("rejected_fact_ids", [])) == rejected_fact_ids
            )
            selected_text = _fact_text(scope.facts[selected_fact_id], scope.nodes)
            resolved = _rebuild_stored_conflict(
                stored,
                status="resolved",
                resolution_source="user",
                cache_cleanup_pending=True,
                selected_fact_id=selected_fact_id,
                selected_fact=selected_text,
                rejected_fact_ids=rejected_fact_ids,
                rejected_fact_texts=sorted(
                    {_fact_text(scope.facts[fid], scope.nodes) for fid in rejected_fact_ids}
                ),
                resolved_by=stored.get("resolved_by") if same_choice else str(user.id),
                resolved_at=stored.get("resolved_at")
                if same_choice
                else datetime.now(timezone.utc).isoformat(),
                resolution_reason=reason,
                text=f"Resolved by user: {selected_text}. Use this accepted fact; competing claims are superseded.",
            )
            accepted = AcceptedConflict(
                resolved,
                {fid: "current" if fid == selected_fact_id else "superseded" for fid in statuses},
            )
            if not same_choice or stored.get("write_pending"):
                await write_conflicts(
                    [ConflictsToWrite(scope, [accepted], shown_fact_ids=set(statuses))], ctx
                )
            else:
                await graph.add_nodes([resolved.model_copy(update={"write_pending": True})])
            result = conflict_detail(resolved.model_dump(mode="json"), scope)

        # Session storage is independent of the dataset lock and database queue slot.
        # The pending marker survives a cleanup failure, so an identical request retries it.
        # Only answers that used a competing fact, the conflict, or a passage cited for
        # it are stale; the subject entity is a hub every unrelated answer about it cites.
        await invalidate_sessions_for_deleted_data(
            dataset.id,
            deleted_node_ids={conflict_id, *stored.get("sources", [])},
            deleted_edge_ids=set(statuses),
            user_id=user.id,
            strict=True,
        )
        async with (
            dataset_lock(dataset.id),
            set_database_global_context_variables(dataset.id, dataset.owner_id),
        ):
            graph = await get_graph_engine()
            rows, _ = await graph.get_filtered_graph_data(
                [{"id": [conflict_id]}], include_edges=False
            )
            current = next((props for node_id, props in rows if str(node_id) == conflict_id), None)
            # Another resolution, review or forget may have run during cache cleanup.
            # Finalize only this decision, using the fresh node so new evidence survives.
            if (
                current
                and current.get("resolution_source") == "user"
                and current.get("selected_fact_id") == resolved.selected_fact_id
                and current.get("resolved_at") == resolved.resolved_at
            ):
                await graph.add_nodes(
                    [
                        _rebuild_stored_conflict(
                            _to_conflict_record(conflict_id, current),
                            write_pending=False,
                            cache_cleanup_pending=False,
                        )
                    ]
                )
        return result.model_copy(update={"write_pending": False})
