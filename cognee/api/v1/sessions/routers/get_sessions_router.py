"""Sessions HTTP router.

Backs the dashboard: list/detail of sessions, aggregate stats, cost
by model. Effective status is computed in SQL with the
abandonment-by-idle rule so no sweeper is needed.
"""

from datetime import datetime, timedelta, timezone
from typing import Literal, Optional
from uuid import UUID as UUIDType

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from sqlalchemy import and_, func, select

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.session_lifecycle.metrics import (
    SessionStatus,
    get_effective_status_sql,
    get_session_row,
    list_session_rows,
)
from cognee.modules.session_lifecycle.models import SessionRecord
from cognee.modules.users.methods import get_authenticated_user
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger

logger = get_logger("sessions_api")


_RangeLiteral = Literal["24h", "7d", "30d", "all"]


def _range_since(range_key: _RangeLiteral) -> Optional[datetime]:
    now = datetime.now(timezone.utc)
    if range_key == "24h":
        return now - timedelta(hours=24)
    if range_key == "7d":
        return now - timedelta(days=7)
    if range_key == "30d":
        return now - timedelta(days=30)
    return None  # "all"


def _record_to_row_dict(rec: SessionRecord) -> dict:
    d = rec.to_dict()
    # Caller sets effective_status attr on the instance
    eff = getattr(rec, "effective_status", None)
    if eff is not None:
        d["effective_status"] = eff
    return d


def get_sessions_router() -> APIRouter:
    router = APIRouter()

    @router.get("")
    async def list_sessions(
        range: _RangeLiteral = Query("30d"),
        status: Optional[str] = Query(None),
        limit: int = Query(50, ge=1, le=500),
        offset: int = Query(0, ge=0),
        order_by: str = Query("last_activity_at"),
        descending: bool = Query(True),
        user: User = Depends(get_authenticated_user),
    ):
        since = _range_since(range)
        try:
            rows = await list_session_rows(
                user_id=user.id,
                since=since,
                status_filter=status,
                limit=limit,
                offset=offset,
                order_by=order_by,
                descending=descending,
            )
            return jsonable_encoder([_record_to_row_dict(r) for r in rows])
        except Exception as exc:
            logger.error("list_sessions failed: %s", exc, exc_info=True)
            return JSONResponse(status_code=500, content={"error": "list failed"})

    @router.get("/stats")
    async def get_stats(
        range: _RangeLiteral = Query("30d"),
        user: User = Depends(get_authenticated_user),
    ):
        """Aggregate counters for the dashboard stat cards + status bar."""
        since = _range_since(range)
        eff = get_effective_status_sql()

        engine = get_relational_engine()
        async with engine.get_async_session() as session:
            base_filter = [SessionRecord.user_id == user.id]
            if since is not None:
                base_filter.append(SessionRecord.last_activity_at >= since)

            # Totals
            totals_stmt = select(
                func.count().label("sessions"),
                func.coalesce(func.sum(SessionRecord.tokens_in), 0).label("tokens_in"),
                func.coalesce(func.sum(SessionRecord.tokens_out), 0).label("tokens_out"),
                func.coalesce(func.sum(SessionRecord.cost_usd), 0).label("cost_usd"),
            ).where(and_(*base_filter))
            totals = (await session.execute(totals_stmt)).one()

            # Duration: sum(ended_at or last_activity_at - started_at)
            dur_stmt = select(
                SessionRecord.started_at,
                SessionRecord.ended_at,
                SessionRecord.last_activity_at,
            ).where(and_(*base_filter))
            durs = (await session.execute(dur_stmt)).all()
            total_seconds = 0.0
            session_count = 0
            for started, ended, last_act in durs:
                if started is None:
                    continue
                end = ended or last_act
                if end is None:
                    continue
                total_seconds += max(0.0, (end - started).total_seconds())
                session_count += 1
            avg_seconds = (total_seconds / session_count) if session_count else 0.0

            # Status buckets (using effective status)
            status_stmt = (
                select(eff.label("s"), func.count().label("c"))
                .where(and_(*base_filter))
                .group_by("s")
            )
            status_rows = (await session.execute(status_stmt)).all()
            buckets = {s: c for s, c in status_rows}
            completed = buckets.get(SessionStatus.COMPLETED.value, 0)
            failed = buckets.get(SessionStatus.FAILED.value, 0)
            abandoned = buckets.get(SessionStatus.ABANDONED.value, 0)
            running = buckets.get(SessionStatus.RUNNING.value, 0)
            decided = completed + failed + abandoned
            success_rate = (completed / decided) if decided else 1.0

        sessions_count = totals.sessions or 0
        avg_spend = (totals.cost_usd / sessions_count) if sessions_count else 0.0

        return jsonable_encoder(
            {
                "range": range,
                "sessions": sessions_count,
                "total_spend_usd": float(totals.cost_usd or 0.0),
                "avg_spend_per_session_usd": float(avg_spend),
                "tokens_in": int(totals.tokens_in or 0),
                "tokens_out": int(totals.tokens_out or 0),
                "tokens_total": int((totals.tokens_in or 0) + (totals.tokens_out or 0)),
                "agent_time_s": float(total_seconds),
                "avg_session_s": float(avg_seconds),
                "success_rate": float(success_rate),
                "completed": int(completed),
                "failed": int(failed),
                "abandoned": int(abandoned),
                "running": int(running),
            }
        )

    @router.get("/cost-by-model")
    async def cost_by_model(
        range: _RangeLiteral = Query("30d"),
        user: User = Depends(get_authenticated_user),
    ):
        since = _range_since(range)
        engine = get_relational_engine()
        async with engine.get_async_session() as session:
            stmt = (
                select(
                    SessionRecord.last_model.label("model"),
                    func.count().label("session_count"),
                    func.coalesce(func.sum(SessionRecord.cost_usd), 0).label("cost_usd"),
                    func.coalesce(func.sum(SessionRecord.tokens_in), 0).label("tokens_in"),
                    func.coalesce(func.sum(SessionRecord.tokens_out), 0).label("tokens_out"),
                )
                .where(SessionRecord.user_id == user.id)
                .group_by(SessionRecord.last_model)
            )
            if since is not None:
                stmt = stmt.where(SessionRecord.last_activity_at >= since)

            rows = (await session.execute(stmt)).all()

        return jsonable_encoder(
            [
                {
                    "model": row.model or "unknown",
                    "session_count": int(row.session_count),
                    "cost_usd": float(row.cost_usd or 0.0),
                    "tokens_in": int(row.tokens_in or 0),
                    "tokens_out": int(row.tokens_out or 0),
                }
                for row in rows
            ]
        )

    @router.get("/{session_id}")
    async def get_session_detail(
        session_id: str,
        user: User = Depends(get_authenticated_user),
    ):
        row = await get_session_row(session_id=session_id, user_id=user.id)
        if row is None:
            raise HTTPException(status_code=404, detail="session not found")

        # Pull the rich content (QAs + trace steps) from the session cache.
        from cognee.infrastructure.session.get_session_manager import get_session_manager

        sm = get_session_manager()
        qas: list = []
        traces: list = []
        if sm.is_available:
            try:
                qas_raw = await sm.get_session(
                    user_id=str(user.id), session_id=session_id, formatted=False
                )
                qas = qas_raw if isinstance(qas_raw, list) else []
                traces = await sm.get_agent_trace_session(
                    user_id=str(user.id), session_id=session_id
                )
            except Exception:
                pass

        record = _record_to_row_dict(row)
        # Label = first QA's question, if any (for the dashboard's "LABEL" column).
        label = None
        for entry in qas:
            if isinstance(entry, dict) and entry.get("question"):
                label = str(entry["question"])[:120]
                break
        record["label"] = label
        record["msg_count"] = len(qas)
        record["tool_calls"] = len(traces)
        record["qas"] = qas[-20:]  # recent tail to cap payload
        record["traces"] = traces[-20:]
        return jsonable_encoder(record)

    return router
