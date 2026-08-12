"""Guards on agent discovery — spec section 5 route 4.

``GET /agents`` has no registry behind it: **an agent exists because it asked
something.** So this module counts ``queries`` rows per candidate label over the
same window a run would use and drops every label whose count is zero. The router
test fakes this away to check the HTTP shape; here the counting itself runs
against real SQLite, because the two properties that matter are SQL-level:

* a label with **no** traffic is absent, so the endpoint can never degrade into a
  dump of the configured prefix map;
* the window is the *same* population a run analyses — same age bound, same query
  types — or the ranking would describe different traffic from the runs it is
  displayed next to. That equivalence is asserted against ``get_queries`` itself
  rather than restated.

``Query.session_id`` does not exist yet, so every prefix-derived label counts zero
and today only ``all`` is reported. The per-label counting, ranking and
zero-dropping are still exercised, by substituting a predicate over a column that
*does* exist for the one ``session_predicate_for_scope`` will return once the
column ships — the escaping and longest-prefix-wins rules themselves are pinned in
``cognee/tests/unit/modules/search/test_get_queries_window.py``.
"""

import importlib
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import false

from cognee.infrastructure.databases.relational import Base
from cognee.infrastructure.databases.relational.create_relational_engine import (
    create_relational_engine,
)
from cognee.modules.recall_coverage.config import RecallCoverageConfig
from cognee.modules.recall_coverage.exceptions import UnknownAgentLabelError
from cognee.modules.recall_coverage.types import AgentScope, AgentScopeMode
from cognee.modules.search.models.Query import Query

agents = importlib.import_module("cognee.modules.recall_coverage.agents")
get_queries_module = importlib.import_module("cognee.modules.search.operations.get_queries")

BASE_TIME = datetime(2026, 8, 12, 12, 0, 0, tzinfo=timezone.utc)

RECALL_TYPE = "GRAPH_COMPLETION"
LOOKUP_TYPE = "CHUNKS"


def _config(**overrides) -> RecallCoverageConfig:
    return RecallCoverageConfig(_env_file=None, **overrides)


@pytest_asyncio.fixture
async def queries_engine(tmp_path, monkeypatch):
    """A SQLite engine holding only ``queries``, bound into both readers."""
    engine = create_relational_engine(
        db_path=str(tmp_path),
        db_name="recall_coverage_agents_test.db",
        db_host="",
        db_port="",
        db_username="",
        db_password="",
        db_provider="sqlite",
    )

    async with engine.engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all, tables=[Query.__table__])

    monkeypatch.setattr(agents, "get_relational_engine", lambda: engine)
    monkeypatch.setattr(get_queries_module, "get_relational_engine", lambda: engine)

    yield engine

    await engine.engine.dispose()


async def _insert(engine, rows):
    """``rows`` are ``(text, query_type, created_at)`` triples."""
    async with engine.get_async_session() as session:
        for text, query_type, created_at in rows:
            session.add(
                Query(
                    id=uuid4(),
                    text=text,
                    query_type=query_type,
                    user_id=uuid4(),
                    created_at=created_at,
                )
            )
        await session.commit()


# --- discovery ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_traffic_means_no_agents_rather_than_the_prefix_map(queries_engine):
    """The endpoint must never answer "these agents exist" from configuration."""
    assert await agents.agent_window_counts(config=_config()) == []


@pytest.mark.asyncio
async def test_a_label_with_traffic_is_reported_with_its_row_count(queries_engine):
    await _insert(
        queries_engine,
        [("Where are the runbooks?", RECALL_TYPE, BASE_TIME) for _ in range(3)],
    )

    counted = await agents.agent_window_counts(config=_config())

    # Until Query.session_id ships, "all" is the only label with a non-zero count:
    # every prefix predicate selects nothing, deliberately, rather than degrading
    # into "every row" and reporting one tenant as one agent.
    assert [(window.label, window.recall_row_count) for window in counted] == [("all", 3)]


@pytest.mark.asyncio
async def test_the_window_is_bounded_by_age_and_query_type(queries_engine):
    await _insert(
        queries_engine,
        [
            ("recent recall", RECALL_TYPE, BASE_TIME),
            ("old recall", RECALL_TYPE, BASE_TIME - timedelta(days=40)),
            ("a lookup, not a recall", LOOKUP_TYPE, BASE_TIME),
        ],
    )

    counted = await agents.agent_window_counts(
        since=BASE_TIME - timedelta(days=30),
        query_types=[RECALL_TYPE],
        config=_config(),
    )

    assert [(window.label, window.recall_row_count) for window in counted] == [("all", 1)]

    # Unbounded, all three rows are in scope — so the filters above did the work.
    unbounded = await agents.agent_window_counts(config=_config())
    assert unbounded[0].recall_row_count == 3


@pytest.mark.asyncio
async def test_the_count_matches_the_window_a_run_would_analyse(queries_engine):
    """The ranking and the runs beside it must describe the same population."""
    since = BASE_TIME - timedelta(days=30)
    await _insert(
        queries_engine,
        [
            ("in the window", RECALL_TYPE, BASE_TIME),
            ("also in the window", RECALL_TYPE, BASE_TIME - timedelta(days=1)),
            ("too old", RECALL_TYPE, since - timedelta(days=1)),
            ("wrong type", LOOKUP_TYPE, BASE_TIME),
        ],
    )
    scope = AgentScope(label="all", prefixes=(), mode=AgentScopeMode.ALL)

    counted = await agents.agent_window_counts(
        since=since, query_types=[RECALL_TYPE], config=_config()
    )
    window = await get_queries_module.get_queries(
        since=since, query_types=[RECALL_TYPE], session_scope=scope
    )

    assert counted[0].recall_row_count == len(window) == 2


# --- ranking and zero-dropping ------------------------------------------------


@pytest.mark.asyncio
async def test_labels_are_ranked_busiest_first_and_silent_ones_dropped(queries_engine, monkeypatch):
    """Substitute a predicate over a column that exists for the session one.

    ``session_predicate_for_scope`` is the single seam between "which label owns
    this row" and "count the rows", so replacing it exercises the counting,
    ranking and dropping exactly as they will run once ``Query.session_id`` lands
    — without this test pretending to own the escaping rules, which live in the
    predicate builder's own tests.
    """
    owners = {"claude-code": "cc:", "codex": "cx:", "cursor": "cu:"}

    def fake_predicate(scope):
        if scope.mode == AgentScopeMode.ALL:
            # The one mode with no session predicate at all, as in production.
            return None
        # Every other label owns a marker or nothing; nothing must select
        # nothing, never "every row".
        return Query.text.like(f"{owners[scope.label]}%") if scope.label in owners else false()

    monkeypatch.setattr(agents, "session_predicate_for_scope", fake_predicate)

    await _insert(
        queries_engine,
        # Two for claude-code, three for codex, none at all for cursor.
        [(f"cc:{index}", RECALL_TYPE, BASE_TIME) for index in range(2)]
        + [(f"cx:{index}", RECALL_TYPE, BASE_TIME) for index in range(3)],
    )

    counted = await agents.agent_window_counts(config=_config())

    assert [(window.label, window.recall_row_count) for window in counted] == [
        ("all", 5),
        ("codex", 3),
        ("claude-code", 2),
    ]
    # cursor is configured but silent, so it is absent — not a zero row.
    assert "cursor" not in {window.label for window in counted}


@pytest.mark.asyncio
async def test_an_equal_count_breaks_the_tie_by_label_for_a_stable_order(
    queries_engine, monkeypatch
):
    monkeypatch.setattr(
        agents,
        "session_predicate_for_scope",
        lambda scope: (
            None if scope.mode == AgentScopeMode.ALL else Query.text.like(f"{scope.label}:%")
        ),
    )

    await _insert(
        queries_engine,
        [("codex:1", RECALL_TYPE, BASE_TIME), ("claude-code:1", RECALL_TYPE, BASE_TIME)],
    )

    counted = await agents.agent_window_counts(labels=["codex", "claude-code"], config=_config())

    assert [window.label for window in counted] == ["claude-code", "codex"]


# --- candidates and validation ------------------------------------------------


def test_candidate_labels_are_the_prefix_map_plus_api_and_all():
    config = _config()
    candidates = agents.candidate_labels(config)

    assert set(config.prefix_map()) <= set(candidates)
    assert candidates[-2:] == ("api", "all")
    # No duplicates: a label counted twice would be listed twice.
    assert len(candidates) == len(set(candidates))


@pytest.mark.asyncio
async def test_an_unknown_label_is_rejected_rather_than_counted_as_zero(queries_engine):
    """A typo must not be indistinguishable from "this agent asked nothing"."""
    with pytest.raises(UnknownAgentLabelError):
        await agents.agent_window_counts(labels=["claude-codex"], config=_config())
