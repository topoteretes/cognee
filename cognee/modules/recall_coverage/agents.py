"""Discover which agents actually asked something in the window.

Spec section 5 route 4. **There is no agent registry** — an agent exists because
it asked something. So this counts ``queries`` rows per candidate label over the
same age window and search types a run would use, and drops every label whose
count is zero. A label that has never been seen is absent from ``GET /agents``
while remaining a perfectly valid ``agent_label`` on every other route, where it
answers "nothing asked yet" rather than 404.

``Query.session_id`` exists and the prefix predicates filter on it for real;
rows written before the column shipped keep it NULL, so on those history rows
prefix labels match nothing and ``api`` matches everything. Two label notes:

* ``all`` and ``api`` are included as candidates even though neither is minted
  from a prefix. ``api`` is genuinely traffic-derived (it is the complement of
  the prefix map), and ``all`` is the aggregate row a UI opens on — its count
  covers the same rows every other row does; it is the total, not a peer;
* counts are scoped by ``user_ids`` — the users the caller may analyse, same
  boundary as the run window — because the relational database is not a tenant
  boundary and unscoped counts would leak tenant-wide traffic volumes.

Counting is one ``SELECT count(*)`` per candidate label inside a single session,
rather than one ``GROUP BY``: a group-by would need to express "which label owns
this session id" in SQL, and that classification is exactly the longest-prefix-
wins rule that ``resolve_agent_scope`` owns and SQL cannot state.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Sequence
from uuid import UUID

from sqlalchemy import func, select

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.recall_coverage.agent_scope import (
    LABEL_ALL,
    LABEL_API,
    resolve_agent_scope,
)
from cognee.modules.recall_coverage.config import (
    RecallCoverageConfig,
    get_recall_coverage_config,
)
from cognee.modules.search.models.Query import Query
from cognee.modules.search.operations.get_queries import session_predicate_for_scope
from cognee.shared.logging_utils import get_logger

logger = get_logger("recall_coverage")


@dataclass(frozen=True)
class AgentWindow:
    """How much traffic one label has in the window."""

    label: str
    recall_row_count: int


def candidate_labels(config: Optional[RecallCoverageConfig] = None) -> tuple[str, ...]:
    """Every label worth counting: the prefix map's, plus ``api`` and ``all``.

    Ordered map-first so two labels with an equal count come back in a stable
    order rather than in dictionary-iteration luck.
    """
    if config is None:
        config = get_recall_coverage_config()
    return tuple(config.prefix_map().keys()) + (LABEL_API, LABEL_ALL)


async def agent_window_counts(
    *,
    since: Optional[datetime] = None,
    query_types: Optional[Sequence[str]] = None,
    labels: Optional[Sequence[str]] = None,
    config: Optional[RecallCoverageConfig] = None,
    user_ids: Optional[Sequence[UUID]] = None,
) -> list[AgentWindow]:
    """Count window rows per label, busiest first, zero-count labels dropped.

    ``since`` and ``query_types`` must be the same window a run would use, or the
    ranking would describe a different population than the runs it is displayed
    next to. ``user_ids`` must be too — the route passes ``visible_user_ids`` so
    the counts describe the same users a run would replay, and nobody else's.
    """
    if config is None:
        config = get_recall_coverage_config()

    wanted = tuple(labels) if labels else candidate_labels(config)

    db_engine = get_relational_engine()
    counted: list[AgentWindow] = []

    async with db_engine.get_async_session() as session:
        for label in wanted:
            scope = resolve_agent_scope(label, config=config)

            statement = select(func.count(Query.id))
            if user_ids is not None:
                statement = statement.where(Query.user_id.in_(tuple(user_ids)))
            if since is not None:
                statement = statement.where(Query.created_at >= since)
            if query_types:
                statement = statement.where(Query.query_type.in_(list(query_types)))

            predicate = session_predicate_for_scope(scope)
            if predicate is not None:
                statement = statement.where(predicate)

            count = int((await session.execute(statement)).scalar() or 0)
            if count > 0:
                counted.append(AgentWindow(label=scope.label, recall_row_count=count))

    counted.sort(key=lambda window: (-window.recall_row_count, window.label))
    logger.debug("recall_coverage: %s labels have traffic in the window", len(counted))
    return counted


__all__ = ["AgentWindow", "agent_window_counts", "candidate_labels"]
