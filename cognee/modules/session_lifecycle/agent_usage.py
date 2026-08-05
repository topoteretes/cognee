"""Session usage joined with agent-connection metadata (CLO-434).

Token/cost aggregates live in ``session_records``
(``session_lifecycle.metrics``); which *client* produced them lives in
the agent-connections registry (``modules.agents``). Neither table
knows about the other, and ``AgentConnection`` isn't a SQL row (it's
in-memory + a JSON blob in ``principal_configuration``), so there's no
real database JOIN to write — these functions do the join in Python,
keyed on ``session_id``.

Kept out of the router so the join/aggregation logic is independently
unit-testable and reusable (e.g. from a CLI command or another router)
without spinning up FastAPI.
"""

from collections import defaultdict
from datetime import datetime, timezone
from typing import Iterable, Optional
from uuid import UUID as UUIDType

from sqlalchemy import or_, select

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.agents.models import AgentConnection
from cognee.modules.agents.operations import list_agent_connections
from cognee.modules.agents.registry import (
    derive_connection_type,
    list_persisted_agent_connections,
    list_registered_agent_connections,
)
from cognee.modules.session_lifecycle.metrics import list_session_rows
from cognee.modules.session_lifecycle.models import SessionRecord
from cognee.modules.users.exceptions import PermissionDeniedError
from cognee.modules.users.models import User
from cognee.modules.users.tenants.methods.get_users_in_tenant import get_users_in_tenant


def _latest_wins_agent_map(
    agents: Iterable[AgentConnection],
) -> dict[str, AgentConnection]:
    """Map session_id -> its most-recently-active AgentConnection.

    A session_id can be re-registered across reconnects; sorting
    ascending by ``last_active_at`` and building the dict last-write-wins
    keeps only the latest.
    """
    sorted_agents = sorted(
        agents, key=lambda a: a.last_active_at or datetime.min.replace(tzinfo=timezone.utc)
    )
    return {a.session_id: a for a in sorted_agents if a.session_id}


async def build_sessions_with_agent_info_page(
    *,
    user: User,
    permitted_dataset_ids: list[UUIDType],
    visible_user_ids: list[UUIDType],
    since: Optional[datetime],
    status_filter: Optional[str],
    limit: int,
    offset: int,
    order_by: str,
    descending: bool,
) -> dict:
    """Session records for ``visible_user_ids`` merged with agent-connection
    metadata, joined on ``session_id``.

    Mirrors the ``GET /sessions`` pagination envelope, with each session
    additionally carrying ``agent_type`` / ``agent_source`` /
    ``agent_session_name`` / ``origin_function``. When no registered
    connection matches a session, ``agent_type`` falls back to the
    session_id/origin_function prefix heuristic (``derive_connection_type``)
    so unregistered sessions (e.g. ``codex-...``) still get a sensible type.
    """
    page = await list_session_rows(
        user_ids=visible_user_ids,
        permitted_dataset_ids=permitted_dataset_ids,
        since=since,
        status_filter=status_filter,
        limit=limit,
        offset=offset,
        order_by=order_by,
        descending=descending,
    )

    agents_response = await list_agent_connections(
        user=user,
        include_sources=False,
        active_only=False,
        limit=10000,
        offset=0,
    )
    agents_by_session = _latest_wins_agent_map(agents_response.agents)

    sessions = []
    for row in page.sessions:
        record = row.to_dict()
        agent = agents_by_session.get(record["session_id"])
        if agent is not None:
            record["agent_type"] = agent.type
            record["agent_source"] = agent.source
            record["agent_session_name"] = agent.agent_session_name
            record["origin_function"] = agent.origin_function
        else:
            record["agent_type"] = derive_connection_type(session_id=record["session_id"])
            record["agent_source"] = None
            record["agent_session_name"] = None
            record["origin_function"] = None
        sessions.append(record)

    return {
        "sessions": sessions,
        "total": page.total,
        "limit": page.limit,
        "offset": page.offset,
        "has_more": page.has_more,
    }


async def _resolve_scope_emails(
    user: User, *, visible_user_ids: Iterable[UUIDType]
) -> dict[UUIDType, str]:
    """Return {user_id: email} for the callers this request may report on.

    Always includes ``visible_user_ids`` — the caller plus their child
    agents, the same base scope every other ``/sessions`` endpoint uses
    (see ``_visible_user_ids`` in the router). A tenant owner/admin (same
    check ``GET /tenants/{id}/users`` uses) additionally gets every
    tenant member unioned in on top of that base scope; a regular
    member — or a caller with no tenant at all (single-user/local mode)
    — just keeps the base scope rather than being denied outright.

    Child-agent ids without a resolvable email map to "unknown" here;
    the caller's own email and any tenant member emails (from
    ``get_users_in_tenant``) are always known.
    """
    email_by_id: dict[UUIDType, str] = {uid: "unknown" for uid in visible_user_ids}
    email_by_id[user.id] = user.email

    if user.tenant_id is not None:
        try:
            users_data = await get_users_in_tenant(user.tenant_id, user)
            for u in users_data:
                email_by_id[UUIDType(u["id"])] = u["email"]
        except PermissionDeniedError:
            pass

    return email_by_id


async def compute_cost_by_user_agent(
    *,
    user: User,
    visible_user_ids: list[UUIDType],
    permitted_dataset_ids: list[UUIDType],
    since: Optional[datetime],
) -> list[dict]:
    """Token/cost totals grouped by (user, agent type) for a spend chart.

    Visibility mirrors every other ``/sessions`` endpoint — the caller,
    their child agents (``visible_user_ids``), and sessions whose
    dataset is shared with them (``permitted_dataset_ids``) — unioned
    with every tenant member's sessions when the caller is a tenant
    owner/admin (see ``_resolve_scope_emails``). Returns rows sorted by
    ``cost_usd`` descending — ready to feed a "who spends the most, with
    which agent" chart without further client-side sorting.
    """
    email_by_user_id = await _resolve_scope_emails(user, visible_user_ids=visible_user_ids)
    scoped_user_ids = list(email_by_user_id.keys())
    if not scoped_user_ids and not permitted_dataset_ids:
        return []

    engine = get_relational_engine()
    async with engine.get_async_session() as session:
        visibility_terms = []
        if scoped_user_ids:
            visibility_terms.append(SessionRecord.user_id.in_(scoped_user_ids))
        if permitted_dataset_ids:
            visibility_terms.append(SessionRecord.dataset_id.in_(permitted_dataset_ids))
        stmt = select(
            SessionRecord.user_id,
            SessionRecord.session_id,
            SessionRecord.tokens_in,
            SessionRecord.tokens_out,
            SessionRecord.cost_usd,
        ).where(or_(*visibility_terms) if len(visibility_terms) > 1 else visibility_terms[0])
        if since is not None:
            stmt = stmt.where(SessionRecord.last_activity_at >= since)
        rows = (await session.execute(stmt)).all()

    persisted_agents = await list_persisted_agent_connections(scoped_user_ids, active_only=False)
    scoped_id_set = set(scoped_user_ids)
    registered_agents = [
        a for a in list_registered_agent_connections() if a.user_id in scoped_id_set
    ]
    agents_by_session = _latest_wins_agent_map([*persisted_agents, *registered_agents])

    aggregates: dict[tuple[UUIDType, str], dict] = defaultdict(
        lambda: {"session_count": 0, "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0}
    )
    for row in rows:
        agent = agents_by_session.get(row.session_id)
        agent_type = (
            agent.type if agent is not None else derive_connection_type(session_id=row.session_id)
        )
        bucket = aggregates[(row.user_id, agent_type)]
        bucket["session_count"] += 1
        bucket["tokens_in"] += row.tokens_in or 0
        bucket["tokens_out"] += row.tokens_out or 0
        bucket["cost_usd"] += row.cost_usd or 0.0

    result = [
        {
            "user_id": str(user_id),
            "user_email": email_by_user_id.get(user_id, "unknown"),
            "agent_type": agent_type,
            "session_count": vals["session_count"],
            "tokens_in": vals["tokens_in"],
            "tokens_out": vals["tokens_out"],
            "tokens_total": vals["tokens_in"] + vals["tokens_out"],
            "cost_usd": round(vals["cost_usd"], 6),
        }
        for (user_id, agent_type), vals in aggregates.items()
    ]
    result.sort(key=lambda r: r["cost_usd"], reverse=True)
    return result
