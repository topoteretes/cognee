"""Read a window of recalls out of ``queries``.

``get_queries`` used to have exactly one shape — one user's newest N rows — and
zero callers. Recall coverage needs the window instead: everything asked in the
tenant over the last N days, restricted to the search types that count as recall
and optionally to one agent's sessions. All of that arrives as additive optional
keyword arguments, and the ``user_id`` filter became opt-in rather than
mandatory.

``get_history`` is deliberately not reused: it unions questions with answers and
orders ASC before applying its limit, so truncating it keeps the *oldest* rows.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional, Sequence, Union
from uuid import UUID

from sqlalchemy import ColumnElement, and_, false, not_, or_, select, true

from cognee.infrastructure.databases.relational import get_relational_engine

# Both names come from the recall-coverage side because that is where they are
# defined and produced: ``AgentScopeMode`` is the vocabulary ``resolve_agent_scope``
# returns, and ``LIKE_ESCAPE_CHAR`` is the escape character it escaped the
# prefixes with — the two halves of one contract, so duplicating either here
# would let them drift apart. ``recall_coverage.types`` imports nothing but the
# standard library and pydantic, so this cannot cycle back into search.
from cognee.modules.recall_coverage.types import LIKE_ESCAPE_CHAR, AgentScope, AgentScopeMode

from ..models.Query import Query


@dataclass(frozen=True)
class QueryWindowRow:
    """One recall in the window.

    A value object rather than a ``Query`` instance: these rows outlive the
    session that read them (the caller collapses, embeds, replays and judges
    them), and a detached ORM instance raises on any attribute that was not
    already loaded.
    """

    query_id: UUID
    text: str
    query_type: str
    user_id: UUID
    dataset_id: Optional[UUID]
    created_at: datetime


def build_session_predicate(
    session_column: Any,
    mode: Union[AgentScopeMode, str] = AgentScopeMode.ALL,
    escaped_prefixes: Sequence[str] = (),
    excluded_prefixes: Sequence[str] = (),
) -> Optional[ColumnElement[bool]]:
    """Turn an agent scope into a ``session_id`` predicate, or ``None`` for no filter.

    Prefixes must already be escaped (``resolve_agent_scope`` does it); they are
    matched with ``ESCAPE '\\'``, without which ``cc_%`` would also match
    ``ccx...``.

    ``excluded_prefixes`` implements "longest prefix wins": ``claude_desktop_a1``
    is a genuine literal match for ``claude_``, so Claude Code's predicate only
    holds for sessions that match none of the longer prefixes Claude Desktop
    owns. Escaping cannot express that; subtraction can.

    The column is a parameter rather than being read from ``Query`` directly so
    both rules can be exercised against a purpose-built table.
    """
    if mode == AgentScopeMode.ALL:
        return None

    if mode == AgentScopeMode.NEGATED:
        if not escaped_prefixes:
            # Nothing is known, so everything is unknown.
            return true()
        # "api" is the complement of the map: no session at all, or a session
        # whose prefix belongs to no known tool. NOT LIKE never matches a NULL,
        # which is why the IS NULL arm is a separate term and not an oversight.
        return or_(
            session_column.is_(None),
            and_(
                *[
                    not_(session_column.like(f"{prefix}%", escape=LIKE_ESCAPE_CHAR))
                    for prefix in escaped_prefixes
                ]
            ),
        )

    if mode == AgentScopeMode.PREFIX:
        if not escaped_prefixes:
            return false()
        # One label owns one or more prefixes, hence an OR group.
        matches_own = or_(
            *[
                session_column.like(f"{prefix}%", escape=LIKE_ESCAPE_CHAR)
                for prefix in escaped_prefixes
            ]
        )
        if not excluded_prefixes:
            return matches_own
        return and_(
            matches_own,
            *[
                not_(session_column.like(f"{prefix}%", escape=LIKE_ESCAPE_CHAR))
                for prefix in excluded_prefixes
            ],
        )

    raise ValueError(f"Unknown session filter mode: {mode}")


def session_predicate_for_scope(scope: AgentScope) -> Optional[ColumnElement[bool]]:
    """The ``session_id`` predicate for a resolved scope, against the live column."""
    return build_session_predicate(
        Query.session_id, scope.mode, scope.prefixes, scope.excluded_prefixes
    )


async def get_queries(
    user_id: Optional[UUID] = None,
    limit: Optional[int] = None,
    *,
    since: Optional[datetime] = None,
    query_types: Optional[Sequence[str]] = None,
    session_scope: Optional[AgentScope] = None,
) -> list[QueryWindowRow]:
    """Return recalls in the window, newest first.

    Every filter is optional and off by default:

    * ``user_id`` — one user's rows. Omitted for a coverage run, see below.
    * ``since`` — lower bound on ``created_at``, i.e. the age window.
    * ``query_types`` — the stored ``Query.query_type`` strings that count as
      recall. ``None`` or empty means unfiltered.
    * ``session_scope`` — one agent, as returned by ``resolve_agent_scope``.
      ``None`` is the ``all`` scope: no session predicate. The whole value object
      is taken rather than loose prefix/mode arguments because a correct
      predicate needs its three fields to agree, and passing two of the three is
      how one agent's sessions end up counted as another's.
    """
    db_engine = get_relational_engine()

    statement = select(
        Query.id,
        Query.text,
        Query.query_type,
        Query.user_id,
        Query.dataset_id,
        Query.created_at,
    )

    # No user filter unless one is asked for. A coverage run is tenant-wide:
    # "analyse Claude Code" means every user's Claude Code traffic, not only the
    # caller's, and ``user_id`` is a column on the output for the UI to filter.
    #
    # That is safe because THE DATABASE IS THE TENANT BOUNDARY — cloud v2
    # provisions one Postgres project per tenant, so scanning every row cannot
    # cross a tenant. Filtering on ``User.tenant_id`` instead would be worse
    # than redundant: ``create_user`` never populates it, so the window would
    # come back empty. Caveat for OSS: a self-hosted deployment that puts
    # several unrelated users in one relational database has no boundary here,
    # and a coverage run there reads all of their question text.
    if user_id is not None:
        statement = statement.where(Query.user_id == user_id)

    if since is not None:
        statement = statement.where(Query.created_at >= since)

    if query_types:
        statement = statement.where(Query.query_type.in_(list(query_types)))

    if session_scope is not None:
        session_predicate = session_predicate_for_scope(session_scope)
        if session_predicate is not None:
            statement = statement.where(session_predicate)

    # Newest first: the window is "the last N asks", so a limit must drop the
    # oldest rows, not the newest.
    statement = statement.order_by(Query.created_at.desc())

    if limit is not None:
        statement = statement.limit(limit)

    async with db_engine.get_async_session() as session:
        rows = (await session.execute(statement)).all()

    return [
        QueryWindowRow(
            query_id=row.id,
            text=row.text,
            query_type=row.query_type,
            user_id=row.user_id,
            dataset_id=row.dataset_id,
            created_at=row.created_at,
        )
        for row in rows
    ]
