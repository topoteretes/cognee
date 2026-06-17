"""Sessions HTTP router.

Backs the dashboard: list/detail of sessions, aggregate stats, cost
by model. Effective status is computed in SQL with the
abandonment-by-idle rule so no sweeper is needed.
"""

from datetime import datetime, timedelta, timezone
from typing import Literal, Optional
from uuid import UUID as UUIDType

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import and_, func, or_, select

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.session_lifecycle.metrics import (
    SessionStatus,
    get_effective_status_sql,
    get_session_row,
    list_session_rows,
)
from cognee.modules.session_lifecycle.models import SessionModelUsage, SessionRecord
from cognee.modules.users.exceptions import PermissionDeniedError
from cognee.modules.users.methods import get_authenticated_user
from cognee.modules.users.models import User
from cognee.modules.users.permissions.methods.get_specific_user_permission_datasets import (
    get_specific_user_permission_datasets,
)
from cognee.shared.logging_utils import get_logger

logger = get_logger("sessions_api")


_RangeLiteral = Literal["24h", "7d", "30d", "all"]


# --------------------------------------------------------------------------- #
# Response models
#
# These mirror the existing JSON contract exactly (snake_case keys). They use
# a plain ``BaseModel`` (NOT ``OutDTO``) so field names serialize as-is rather
# than being camelCased by ``OutDTO``'s alias generator.
# --------------------------------------------------------------------------- #


class SessionRowResponse(BaseModel):
    """A session list/detail row — mirrors ``SessionRecord.to_dict()`` plus
    the read-time computed ``effective_status``."""

    session_id: str
    user_id: str
    dataset_id: Optional[str] = None
    status: str
    started_at: Optional[str] = None
    last_activity_at: Optional[str] = None
    ended_at: Optional[str] = None
    tokens_in: int
    tokens_out: int
    cost_usd: float
    error_count: int
    last_model: Optional[str] = None
    effective_status: str


class SessionListResponse(BaseModel):
    """Paginated envelope returned by ``GET /api/v1/sessions``."""

    sessions: list[SessionRowResponse] = Field(default_factory=list)
    total: int
    limit: int
    offset: int
    has_more: bool


class SessionStatsResponse(BaseModel):
    """Aggregate counters returned by ``GET /api/v1/sessions/stats``."""

    range: _RangeLiteral
    sessions: int
    total_spend_usd: float
    avg_spend_per_session_usd: float
    tokens_in: int
    tokens_out: int
    tokens_total: int
    agent_time_s: float
    avg_session_s: float
    success_rate: float
    completed: int
    failed: int
    abandoned: int
    running: int


class CostByModelRow(BaseModel):
    """One row of ``GET /api/v1/sessions/cost-by-model``."""

    model: str
    session_count: int
    cost_usd: float
    tokens_in: int
    tokens_out: int


def _range_since(range_key: _RangeLiteral) -> Optional[datetime]:
    now = datetime.now(timezone.utc)
    if range_key == "24h":
        return now - timedelta(hours=24)
    if range_key == "7d":
        return now - timedelta(days=7)
    if range_key == "30d":
        return now - timedelta(days=30)
    return None  # "all"


async def _permitted_dataset_ids_for(user: User) -> list[UUIDType]:
    """Return the UUIDs of datasets this user can read (empty on none)."""
    try:
        datasets = await get_specific_user_permission_datasets(user.id, "read", None)
        return [ds.id for ds in datasets] if datasets else []
    except PermissionDeniedError:
        return []
    except Exception:
        return []


async def _child_agent_user_ids(user_id: UUIDType) -> list[UUIDType]:
    """Return user IDs of agents whose parent_user_id matches this user."""
    engine = get_relational_engine()
    async with engine.get_async_session() as session:
        from cognee.modules.users.models import User as UserModel

        rows = (
            await session.execute(select(UserModel.id).where(UserModel.parent_user_id == user_id))
        ).all()
        return [row.id for row in rows]


async def _visible_user_ids(user: User) -> list[UUIDType]:
    """User's own ID plus any child agent IDs."""
    ids = [user.id]
    ids.extend(await _child_agent_user_ids(user.id))
    return ids


def get_sessions_router() -> APIRouter:
    router = APIRouter()

    @router.get("", response_model=SessionListResponse)
    async def list_sessions(
        range: _RangeLiteral = Query(
            "30d",
            description=(
                "Time window filtered on last_activity_at: last 24 hours (24h), "
                "7 days (7d), 30 days (30d), or all time (all)."
            ),
            examples=["30d"],
        ),
        status: Optional[str] = Query(
            None,
            description=(
                "Filter by effective status: 'running', 'completed', 'failed', or 'abandoned'. "
                "'abandoned' is computed at read time: stored status 'running' with "
                "last_activity_at older than SESSION_ABANDON_AFTER_SECONDS (default 30 min). "
                "Any other value matches nothing and returns an empty list."
            ),
            examples=["completed"],
        ),
        limit: int = Query(50, ge=1, le=500, description="Page size (max 500)."),
        offset: int = Query(0, ge=0, description="Rows to skip for pagination."),
        order_by: str = Query(
            "last_activity_at",
            description=(
                "Column to sort by: last_activity_at, started_at, ended_at, cost_usd, "
                "tokens_in, or tokens_out. Unknown values silently fall back to "
                "last_activity_at."
            ),
            examples=["cost_usd"],
        ),
        descending: bool = Query(True, description="Sort descending (newest/largest first)."),
        user: User = Depends(get_authenticated_user),
    ):
        """Paginated list of sessions.

        ## Request Parameters
        - **range** (Literal): Time window on last_activity_at: 24h, 7d, 30d, or all
          (default: 30d).
        - **status** (Optional[str]): Effective-status filter: running, completed, failed,
          or abandoned.
        - **limit** (int): Page size, 1-500 (default: 50).
        - **offset** (int): Rows to skip for pagination (default: 0).
        - **order_by** (str): Sort column: last_activity_at, started_at, ended_at, cost_usd,
          tokens_in, or tokens_out (default: last_activity_at).
        - **descending** (bool): Sort newest/largest first (default: true).

        Response envelope::

            {
              "sessions": [...],
              "total": <int>,      # rows matching filters before pagination
              "limit":  <int>,
              "offset": <int>,
              "has_more": <bool>,
            }
        """
        since = _range_since(range)
        try:
            permitted = await _permitted_dataset_ids_for(user)
            visible_ids = await _visible_user_ids(user)
            page = await list_session_rows(
                user_ids=visible_ids,
                permitted_dataset_ids=permitted,
                since=since,
                status_filter=status,
                limit=limit,
                offset=offset,
                order_by=order_by,
                descending=descending,
            )
            return SessionListResponse(
                sessions=[SessionRowResponse(**r.to_dict()) for r in page.sessions],
                total=page.total,
                limit=page.limit,
                offset=page.offset,
                has_more=page.has_more,
            )
        except Exception as exc:
            logger.error("list_sessions failed: %s", exc, exc_info=True)
            return JSONResponse(status_code=500, content={"error": "list failed"})

    @router.get("/stats", response_model=SessionStatsResponse)
    async def get_stats(
        range: _RangeLiteral = Query(
            "30d",
            description="Time window filtered on last_activity_at: 24h, 7d, 30d, or all.",
            examples=["30d"],
        ),
        user: User = Depends(get_authenticated_user),
    ):
        """Aggregate counters for the dashboard stat cards + status bar.

        ## Request Parameters
        - **range** (Literal): Time window on last_activity_at: 24h, 7d, 30d, or all
          (default: 30d).

        ## Response
        Returns a JSON object with:
        - **sessions** (int): Number of sessions in the window.
        - **total_spend_usd** / **avg_spend_per_session_usd** (float): Cost totals.
        - **tokens_in** / **tokens_out** / **tokens_total** (int): Token totals.
        - **agent_time_s** / **avg_session_s** (float): Summed and average session duration
          in seconds.
        - **success_rate** (float): completed / (completed + failed + abandoned); 1.0 when
          no session has ended yet.
        - **completed** / **failed** / **abandoned** / **running** (int): Effective-status
          counts.
        """
        since = _range_since(range)
        eff = get_effective_status_sql()
        permitted = await _permitted_dataset_ids_for(user)
        visible_ids = await _visible_user_ids(user)

        engine = get_relational_engine()
        async with engine.get_async_session() as session:
            visibility_terms = [SessionRecord.user_id.in_(visible_ids)]
            if permitted:
                visibility_terms.append(SessionRecord.dataset_id.in_(permitted))
            base_filter = [
                or_(*visibility_terms) if len(visibility_terms) > 1 else visibility_terms[0]
            ]
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

        return SessionStatsResponse(
            range=range,
            sessions=sessions_count,
            total_spend_usd=float(totals.cost_usd or 0.0),
            avg_spend_per_session_usd=float(avg_spend),
            tokens_in=int(totals.tokens_in or 0),
            tokens_out=int(totals.tokens_out or 0),
            tokens_total=int((totals.tokens_in or 0) + (totals.tokens_out or 0)),
            agent_time_s=float(total_seconds),
            avg_session_s=float(avg_seconds),
            success_rate=float(success_rate),
            completed=int(completed),
            failed=int(failed),
            abandoned=int(abandoned),
            running=int(running),
        )

    @router.get("/cost-by-model", response_model=list[CostByModelRow])
    async def cost_by_model(
        range: _RangeLiteral = Query(
            "30d",
            description=(
                "Time window filtered on session_records.last_activity_at: 24h, 7d, 30d, or all."
            ),
            examples=["30d"],
        ),
        user: User = Depends(get_authenticated_user),
    ):
        """Cost + token totals grouped by the model that produced them.

        Aggregates ``session_model_usage`` rows (one per session × model),
        so a session that used multiple models splits its cost correctly.
        Filters on ``session_records.last_activity_at`` to scope by
        range — requires a join back to the session row.

        ## Request Parameters
        - **range** (Literal): Time window: 24h, 7d, 30d, or all (default: 30d).
        """
        since = _range_since(range)
        permitted = await _permitted_dataset_ids_for(user)
        visible_ids = await _visible_user_ids(user)
        engine = get_relational_engine()
        async with engine.get_async_session() as session:
            visibility_terms = [SessionRecord.user_id.in_(visible_ids)]
            if permitted:
                visibility_terms.append(SessionRecord.dataset_id.in_(permitted))
            stmt = (
                select(
                    SessionModelUsage.model.label("model"),
                    func.count(func.distinct(SessionModelUsage.session_id)).label("session_count"),
                    func.coalesce(func.sum(SessionModelUsage.cost_usd), 0).label("cost_usd"),
                    func.coalesce(func.sum(SessionModelUsage.tokens_in), 0).label("tokens_in"),
                    func.coalesce(func.sum(SessionModelUsage.tokens_out), 0).label("tokens_out"),
                )
                .join(
                    SessionRecord,
                    and_(
                        SessionModelUsage.session_id == SessionRecord.session_id,
                        SessionModelUsage.user_id == SessionRecord.user_id,
                    ),
                )
                .where(or_(*visibility_terms) if len(visibility_terms) > 1 else visibility_terms[0])
                .group_by(SessionModelUsage.model)
                .order_by(func.sum(SessionModelUsage.cost_usd).desc())
            )
            if since is not None:
                stmt = stmt.where(SessionRecord.last_activity_at >= since)

            rows = (await session.execute(stmt)).all()

        return [
            CostByModelRow(
                model=row.model or "unknown",
                session_count=int(row.session_count),
                cost_usd=float(row.cost_usd or 0.0),
                tokens_in=int(row.tokens_in or 0),
                tokens_out=int(row.tokens_out or 0),
            )
            for row in rows
        ]

    @router.get("/{session_id}")
    async def get_session_detail(
        session_id: str = Path(
            ...,
            description=(
                "Client-supplied session identifier; the same value passed as session_id "
                "to POST /api/v1/remember."
            ),
            examples=["claude-code-1718000000"],
        ),
        user: User = Depends(get_authenticated_user),
    ):
        permitted = await _permitted_dataset_ids_for(user)
        visible_ids = await _visible_user_ids(user)
        row = await get_session_row(
            session_id=session_id,
            user_id=user.id,
            user_ids=visible_ids,
            permitted_dataset_ids=permitted,
        )
        if row is None:
            raise HTTPException(status_code=404, detail="session not found")

        # Pull the rich content (QAs + trace steps) from the session cache.
        # Important: cache entries are keyed by the session's OWNER, not
        # the authenticated caller. A user with dataset-granted read
        # permission is viewing someone else's session, so we need to
        # query the cache under the owner's user_id from the row.
        from cognee.infrastructure.session.get_session_manager import get_session_manager

        owner_user_id = str(getattr(row, "user_id", ""))
        sm = get_session_manager()
        qas: list = []
        traces: list = []
        if sm.is_available and owner_user_id:
            try:
                qas_raw = await sm.get_session(
                    user_id=owner_user_id, session_id=session_id, formatted=False
                )
                qas = qas_raw if isinstance(qas_raw, list) else []
                traces = await sm.get_agent_trace_session(
                    user_id=owner_user_id, session_id=session_id
                )
            except Exception:
                pass

        record = row.to_dict()
        # Label = first QA's question, else first trace's origin_function
        # (so trace-only sessions — the plugin case — still have a label).
        label = None
        for entry in qas:
            if getattr(entry, "question", None):
                label = str(entry.question)[:120]
                break
        if label is None:
            for entry in traces:
                if getattr(entry, "origin_function", None):
                    label = str(entry.origin_function)
                    break
        record["label"] = label
        record["msg_count"] = len(qas)
        record["tool_calls"] = len(traces)
        record["qas"] = qas[-20:]  # recent tail to cap payload
        record["traces"] = traces[-20:]
        return jsonable_encoder(record)

    return router
