"""Dataset-authorized conflict decisions, serialized with ingestion and improve."""

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
    ConflictsToFinalize,
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
    user, dataset = await _authorized_dataset(dataset_id, user, "read")
    async with (
        dataset_lock(dataset.id),
        set_database_global_context_variables(dataset.id, dataset.owner_id),
    ):
        scope = await read_entity_facts(None, ctx=PipelineContext(user=user, dataset=dataset))
        return [
            conflict_detail(stored, scope)
            for _, stored in sorted(scope.conflicts.items())
            if status is None or stored["status"] == status
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

    Repeating a choice is idempotent. A new choice explicitly revises the prior
    decision. Automatic improve runs cannot replace a user decision.
    """
    user, dataset = await _authorized_dataset(dataset_id, user, "write")
    conflict_id, selected_fact_id = str(conflict_id), str(selected_fact_id)
    async with (
        dataset_lock(dataset.id),
        set_database_global_context_variables(dataset.id, dataset.owner_id),
    ):
        graph = await get_graph_engine()
        rows, _ = await graph.get_filtered_graph_data([{"id": [conflict_id]}])
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
        same_choice = (
            stored.get("resolution_source") == "user"
            and stored.get("selected_fact_id") == selected_fact_id
            and stored.get("resolution_reason") == reason
        )
        selected_text = _fact_text(scope.facts[selected_fact_id], scope.nodes)
        resolved = _rebuild_stored_conflict(
            stored,
            status="resolved",
            resolution_source="user",
            selected_fact_id=selected_fact_id,
            selected_fact=selected_text,
            rejected_fact_ids=sorted(fid for fid in statuses if fid != selected_fact_id),
            rejected_fact_texts=sorted(
                {
                    _fact_text(scope.facts[fid], scope.nodes)
                    for fid in statuses
                    if fid != selected_fact_id
                }
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
        async with record_operation("resolve_conflict") as operation:
            operation.set_user(user)
            operation.set_dataset(dataset.id)
            if not same_choice or stored.get("write_pending"):
                await write_conflicts(
                    [ConflictsToWrite(scope, [accepted], shown_fact_ids=set(statuses))], ctx
                )
            else:
                await graph.add_nodes([resolved.model_copy(update={"write_pending": True})])
            # Remove cached answers/lessons based on either side, not unrelated sessions.
            # Retry this on an idempotent request too, including after a partial failure.
            await invalidate_sessions_for_deleted_data(
                dataset.id,
                deleted_node_ids={conflict_id, stored["about_id"], *stored.get("sources", [])},
                deleted_edge_ids=set(statuses),
                user_id=user.id,
                strict=True,
            )
            await write_conflicts([ConflictsToFinalize(accepted_conflicts=[accepted])], ctx)
        return conflict_detail(
            resolved.model_copy(update={"write_pending": False}).model_dump(mode="json"), scope
        )
