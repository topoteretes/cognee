"""Authenticated, user-and-dataset-scoped answer evaluation API."""

import os
from contextlib import asynccontextmanager
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.routing import APIRoute
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.answer_evaluations import service
from cognee.modules.answer_evaluations.models import EvaluationRun, EvaluationScope
from cognee.modules.answer_evaluations.privacy import Protection
from cognee.modules.answer_evaluations.schema import Trace
from cognee.modules.users.methods.get_authenticated_user import get_authenticated_user
from cognee.modules.users.permissions.methods.get_permitted_dataset_ids import (
    get_permitted_dataset_ids,
)


class BoundedRoute(APIRoute):
    def get_route_handler(self):
        original = super().get_route_handler()

        async def bounded(request):
            if request.method in {"POST", "PUT"}:
                body = bytearray()
                async for chunk in request.stream():
                    body.extend(chunk)
                    if len(body) > 64000:
                        raise HTTPException(413, "Evaluation request exceeds 64 KB")
                request._body = bytes(body)
            return await original(request)

        return bounded


class Consent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool


@asynccontextmanager
async def transaction():
    try:
        async with get_relational_engine().get_async_session() as session:
            yield session
            await session.commit()
    except service.EvaluationError as exc:
        raise HTTPException(exc.status, exc.message) from None
    except SQLAlchemyError:
        raise HTTPException(
            503, "Evaluation storage unavailable; retry with the same event ID"
        ) from None


def protection():
    try:
        return Protection()
    except (ValueError, TypeError):
        raise HTTPException(503, "Evaluation protection is not configured") from None


async def authorize(user, dataset_id):
    if dataset_id not in await get_permitted_dataset_ids(user.id):
        raise HTTPException(403, "Dataset is not readable")
    return str(getattr(user, "parent_user_id", None) or user.id)


def require_human(user):
    if getattr(user, "parent_user_id", None):
        raise HTTPException(
            403, "Use the owning user's credential for evaluation history and privacy controls"
        )


def get_evaluations_router():
    router = APIRouter(route_class=BoundedRoute)

    @router.post("/traces", status_code=202)
    async def capture(trace: Trace, user=Depends(get_authenticated_user)):
        """Accept an explicitly opted-in, bounded trace after durable commit."""
        if os.environ.get("COGNEE_EVALUATIONS_ENABLED", "false").lower() != "true":
            raise HTTPException(503, "Evaluation capture is disabled")
        owner = await authorize(user, trace.dataset_id)
        async with transaction() as session:
            result = await service.ingest(session, protection(), owner, trace)
        return result

    @router.get("/summary")
    async def read_summary(dataset_id: UUID, user=Depends(get_authenticated_user)):
        """Read only this user's evaluation summary for the requested dataset."""
        owner = await authorize(user, dataset_id)
        async with transaction() as session:
            return await service.summary(session, protection().identity(owner, str(dataset_id)))

    @router.get("/runs")
    async def runs(
        dataset_id: UUID, limit: int = Query(20, ge=1, le=100), user=Depends(get_authenticated_user)
    ):
        """List bounded run history without trace bodies."""
        require_human(user)
        owner = await authorize(user, dataset_id)
        async with transaction() as session:
            rows = (
                await session.scalars(
                    select(EvaluationRun)
                    .where(EvaluationRun.scope_id == protection().identity(owner, str(dataset_id)))
                    .order_by(EvaluationRun.created_at.desc())
                    .limit(limit)
                )
            ).all()
            return [
                {
                    "id": r.id,
                    "created_at": r.created_at,
                    "status": r.status,
                    "summary": r.result.get("summary"),
                }
                for r in rows
            ]

    @router.get("/runs/{run_id}")
    async def run_detail(run_id: UUID, dataset_id: UUID, user=Depends(get_authenticated_user)):
        """Return selection provenance and numeric verdicts for an owned run."""
        require_human(user)
        owner = await authorize(user, dataset_id)
        async with transaction() as session:
            row = await session.get(EvaluationRun, str(run_id))
            if row is None or row.scope_id != protection().identity(owner, str(dataset_id)):
                raise HTTPException(404, "Evaluation run not found")
            return {"id": row.id, "created_at": row.created_at, "status": row.status, **row.result}

    @router.get("/consent")
    async def read_consent(dataset_id: UUID, user=Depends(get_authenticated_user)):
        """Report whether warehouse reconciliation has acknowledged deletion."""
        require_human(user)
        owner = await authorize(user, dataset_id)
        async with transaction() as session:
            scope = await session.get(
                EvaluationScope, protection().identity(owner, str(dataset_id))
            )
            return {
                "enabled": bool(scope and scope.enabled),
                "warehouse_pending": bool(scope and scope.synced_revision != scope.revision),
            }

    @router.put("/consent")
    async def consent(body: Consent, dataset_id: UUID, user=Depends(get_authenticated_user)):
        """Revoke capture and erase content, or explicitly re-enable it."""
        require_human(user)
        owner = await authorize(user, dataset_id)
        async with transaction() as session:
            scope = await service.lock_scope(
                session,
                protection().identity(owner, str(dataset_id)),
                owner,
                str(dataset_id),
                create=True,
            )
            scope.enabled = body.enabled
            if not body.enabled:
                return await service.erase(session, scope)
            return {"enabled": True}

    @router.delete("/traces/{receipt}", status_code=202)
    async def delete_trace(receipt: str, dataset_id: UUID, user=Depends(get_authenticated_user)):
        """Erase source content and request warehouse reconciliation."""
        require_human(user)
        owner = await authorize(user, dataset_id)
        async with transaction() as session:
            scope = await service.lock_scope(
                session, protection().identity(owner, str(dataset_id)), owner, str(dataset_id)
            )
            if scope is None:
                raise HTTPException(404, "Evaluation scope not found")
            return await service.erase(session, scope, receipt)

    return router
