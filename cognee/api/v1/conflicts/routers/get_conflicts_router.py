"""Inspect competing facts and record the dataset's accepted information."""

from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from pydantic import Field

from cognee.api.DTO import InDTO
from cognee.modules.conflicts.methods import list_conflicts, resolve_conflict
from cognee.modules.conflicts.models import ConflictDetail
from cognee.modules.users.methods import get_authenticated_user
from cognee.modules.users.models import User


class ResolveConflictRequest(InDTO):
    dataset_id: UUID = Field(
        description="Dataset containing the conflict. Requires write permission."
    )
    selected_fact_id: UUID = Field(description="Fact id from the conflict's facts list to accept.")
    reason: str | None = Field(default=None, description="Optional explanation of the decision.")


def get_conflicts_router() -> APIRouter:
    router = APIRouter()

    @router.get("", response_model=list[ConflictDetail])
    async def get_conflicts(
        dataset_id: UUID = Query(description="Dataset to inspect. Requires read permission."),
        status: Literal["resolved", "unresolved"] | None = Query(default=None),
        user: User = Depends(get_authenticated_user),
    ):
        """List conflicts, candidate fact ids, source evidence and previous decisions."""
        return await list_conflicts(dataset_id, status=status, user=user)

    @router.post("/{conflict_id}/resolve", response_model=ConflictDetail)
    async def choose_fact(
        conflict_id: UUID,
        request: ResolveConflictRequest,
        user: User = Depends(get_authenticated_user),
    ):
        """Accept one competing fact and mark the conflict resolved.

        Repeating a choice over the same facts is idempotent; repeated after a review
        listed new evidence, it also rejects that evidence. Choosing another fact
        revises the decision.
        Automatic conflict review preserves the user decision. Rejected source facts
        remain stored for inspection, but normal retrieval uses the selected information.
        """
        return await resolve_conflict(
            conflict_id,
            dataset_id=request.dataset_id,
            selected_fact_id=request.selected_fact_id,
            reason=request.reason,
            user=user,
        )

    return router
