"""Guards on the recall window read out of ``queries``.

Two halves:

* the window filters against the real ``queries`` table on SQLite — tenant-wide
  (no ``user_id`` filter), age-bounded, type-bounded, newest first;
* the session-id predicate against a two-column stand-in table, one row per
  interesting session shape. The rules that decide which agent a session
  belongs to are what would otherwise be discovered wrong later, on real data,
  as "Claude Desktop's questions are showing up under Claude Code" — so they are
  exercised against a table holding exactly the awkward cases and nothing else.
  ``test_session_predicate_is_wired_to_the_live_column`` covers the join back to
  the real ``Query.session_id``.
"""

import importlib
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import Column, MetaData, String, Table, create_engine, select

from cognee.infrastructure.databases.relational import Base
from cognee.infrastructure.databases.relational.create_relational_engine import (
    create_relational_engine,
)
from cognee.modules.recall_coverage.agent_scope import resolve_agent_scope
from cognee.modules.recall_coverage.config import RecallCoverageConfig
from cognee.modules.search.models.Query import Query

get_queries_mod = importlib.import_module("cognee.modules.search.operations.get_queries")

NOW = datetime(2026, 8, 12, 12, 0, 0, tzinfo=timezone.utc)

# The session-id cases, isolated from everything else ``queries`` carries. Its
# own MetaData so it never joins ``Base.metadata`` and cannot be created by
# anything else.
_stub_metadata = MetaData()
SESSIONED_QUERIES = Table(
    "sessioned_queries",
    _stub_metadata,
    Column("id", String, primary_key=True),
    Column("session_id", String, nullable=True),
)

# One session id per interesting case. The first two are the whole point: the
# prefix "claude_" must not claim "claude_desktop_...", and "cc_" must not claim
# "ccx...", because "_" is a single-character wildcard in LIKE.
SESSION_IDS = {
    "claude_code": "claude_a1",
    "claude_desktop": "claude_desktop_a1",
    "cc": "cc_a1",
    "cc_wildcard_trap": "ccx_a1",
    "codex": "codex_a1",
    "unknown_prefix": "weird_a1",
    "no_session": None,
}


def _config() -> RecallCoverageConfig:
    """A config that ignores the developer's ``.env``."""
    return RecallCoverageConfig(_env_file=None)


def _matching_session_keys(agent_label) -> set[str]:
    """Keys of ``SESSION_IDS`` selected by ``agent_label``, run against SQLite."""
    scope = resolve_agent_scope(agent_label, config=_config())
    predicate = get_queries_mod.build_session_predicate(
        SESSIONED_QUERIES.c.session_id, scope.mode, scope.prefixes, scope.excluded_prefixes
    )

    engine = create_engine("sqlite://")
    try:
        with engine.begin() as connection:
            _stub_metadata.create_all(connection)
            connection.execute(
                SESSIONED_QUERIES.insert(),
                [{"id": key, "session_id": value} for key, value in SESSION_IDS.items()],
            )
            statement = select(SESSIONED_QUERIES.c.id)
            if predicate is not None:
                statement = statement.where(predicate)
            return {row.id for row in connection.execute(statement)}
    finally:
        engine.dispose()


def test_claude_desktop_sessions_never_land_in_claude_code():
    """Longest prefix wins: ``claude_desktop_a1`` literally starts with ``claude_``."""
    assert _matching_session_keys("claude-code") == {"claude_code", "cc"}


def test_literal_underscore_in_a_prefix_is_escaped():
    """``cc_`` is a literal prefix, so ``ccx_a1`` is nobody's Claude Code session."""
    matched = _matching_session_keys("claude-code")

    assert "cc" in matched
    assert "cc_wildcard_trap" not in matched


def test_claude_desktop_label_matches_only_its_own_sessions():
    assert _matching_session_keys("claude-desktop") == {"claude_desktop"}


def test_api_negation_catches_null_sessions_and_unknown_prefixes():
    """``ccx_a1`` counts as api precisely because ``cc_`` is matched literally."""
    assert _matching_session_keys("api") == {
        "no_session",
        "unknown_prefix",
        "cc_wildcard_trap",
    }


def test_every_session_belongs_to_exactly_one_label():
    """No session may be double-counted or dropped across the labels plus api."""
    labels = list(_config().prefix_map()) + ["api"]
    owners: dict[str, list[str]] = {key: [] for key in SESSION_IDS}

    for label in labels:
        for key in _matching_session_keys(label):
            owners[key].append(label)

    assert all(len(matched) == 1 for matched in owners.values()), owners


def test_all_mode_applies_no_session_predicate():
    """The ``all`` scope is the absence of a filter, not a filter that passes."""
    scope = resolve_agent_scope("all", config=_config())

    assert (
        get_queries_mod.build_session_predicate(
            SESSIONED_QUERIES.c.session_id, scope.mode, scope.prefixes, scope.excluded_prefixes
        )
        is None
    )
    assert _matching_session_keys("all") == set(SESSION_IDS)


def test_session_predicate_is_wired_to_the_live_column():
    """The scoped predicates filter on the real ``Query.session_id`` column.

    Guards the wiring, not the matching rules — those are covered above against
    a purpose-built table. If ``session_predicate_for_scope`` were ever pointed
    somewhere other than the live column, every session-scoped label would go
    quietly wrong while the rule tests kept passing.
    """
    for label in ("claude-code", "api"):
        predicate = get_queries_mod.session_predicate_for_scope(
            resolve_agent_scope(label, config=_config())
        )
        compiled = str(predicate.compile(compile_kwargs={"literal_binds": True}))

        assert "queries.session_id" in compiled

    # "all" applies no session predicate at all.
    assert (
        get_queries_mod.session_predicate_for_scope(resolve_agent_scope("all", config=_config()))
        is None
    )


@pytest_asyncio.fixture
async def queries_engine(tmp_path, monkeypatch):
    """A SQLite engine holding only the ``queries`` table."""
    engine = create_relational_engine(
        db_path=str(tmp_path),
        db_name="queries_window_test.db",
        db_host="",
        db_port="",
        db_username="",
        db_password="",
        db_provider="sqlite",
    )

    async with engine.engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all, tables=[Query.__table__])

    monkeypatch.setattr(get_queries_mod, "get_relational_engine", lambda: engine)

    yield engine

    await engine.engine.dispose()


async def _insert_query(engine, **overrides):
    """Insert one ``queries`` row and return the values written."""
    values = {
        "id": uuid4(),
        "text": "what is the escalation path?",
        "query_type": "GRAPH_COMPLETION",
        "user_id": uuid4(),
        "dataset_id": None,
        "session_id": None,
        "created_at": NOW,
    }
    values.update(overrides)

    async with engine.engine.begin() as connection:
        await connection.execute(Query.__table__.insert().values(**values))

    return values


@pytest.mark.asyncio
async def test_window_applies_no_user_filter_by_default(queries_engine):
    """Unfiltered when nothing is passed; the boundary is the caller's ``user_ids``."""
    anna, ben = uuid4(), uuid4()
    await _insert_query(queries_engine, user_id=anna, text="anna asked")
    await _insert_query(queries_engine, user_id=ben, text="ben asked")

    rows = await get_queries_mod.get_queries()

    assert {row.user_id for row in rows} == {anna, ben}


@pytest.mark.asyncio
async def test_window_is_bounded_by_user_ids(queries_engine):
    """``user_ids`` is the tenant boundary: a stranger's rows never enter the window.

    The relational database is shared in OSS deployments, so without this filter
    a coverage run would read every unrelated user's question text.
    """
    anna, ben, stranger = uuid4(), uuid4(), uuid4()
    await _insert_query(queries_engine, user_id=anna, text="anna asked")
    await _insert_query(queries_engine, user_id=ben, text="ben asked")
    await _insert_query(queries_engine, user_id=stranger, text="stranger asked")

    rows = await get_queries_mod.get_queries(user_ids=(anna, ben))

    assert {row.user_id for row in rows} == {anna, ben}
    assert all("stranger" not in row.text for row in rows)


@pytest.mark.asyncio
async def test_count_queries_matches_the_window_it_counts(queries_engine):
    """Same filters as the read, so a capped fetch can report the true row count."""
    anna, stranger = uuid4(), uuid4()
    for index in range(3):
        await _insert_query(queries_engine, user_id=anna, created_at=NOW - timedelta(minutes=index))
    await _insert_query(queries_engine, user_id=anna, created_at=NOW - timedelta(days=45))
    await _insert_query(queries_engine, user_id=stranger, created_at=NOW)

    since = NOW - timedelta(days=30)
    rows = await get_queries_mod.get_queries(limit=2, since=since, user_ids=(anna,))
    count = await get_queries_mod.count_queries(since=since, user_ids=(anna,))

    assert len(rows) == 2
    assert count == 3


@pytest.mark.asyncio
async def test_window_still_supports_an_explicit_user_filter(queries_engine):
    anna, ben = uuid4(), uuid4()
    await _insert_query(queries_engine, user_id=anna)
    await _insert_query(queries_engine, user_id=ben)

    rows = await get_queries_mod.get_queries(anna)

    assert [row.user_id for row in rows] == [anna]


@pytest.mark.asyncio
async def test_window_filters_by_age(queries_engine):
    await _insert_query(queries_engine, text="recent", created_at=NOW)
    await _insert_query(queries_engine, text="stale", created_at=NOW - timedelta(days=45))

    rows = await get_queries_mod.get_queries(since=NOW - timedelta(days=30))

    assert [row.text for row in rows] == ["recent"]


@pytest.mark.asyncio
async def test_window_filters_by_query_type(queries_engine):
    await _insert_query(queries_engine, text="recall", query_type="GRAPH_COMPLETION")
    await _insert_query(queries_engine, text="lookup", query_type="CHUNKS")

    rows = await get_queries_mod.get_queries(query_types=_config().query_types())

    assert [row.text for row in rows] == ["recall"]


@pytest.mark.asyncio
async def test_window_is_newest_first_and_truncates_the_oldest(queries_engine):
    """A limit must drop the oldest rows -- the window is "the last N asks"."""
    for index in range(3):
        await _insert_query(
            queries_engine, text=f"ask {index}", created_at=NOW - timedelta(days=index)
        )

    rows = await get_queries_mod.get_queries(limit=2)

    assert [row.text for row in rows] == ["ask 0", "ask 1"]


@pytest.mark.asyncio
async def test_window_rows_carry_the_replay_projection(queries_engine):
    """Everything the pipeline needs, as a value object rather than a live ORM row."""
    dataset_id = uuid4()
    written = await _insert_query(queries_engine, dataset_id=dataset_id, session_id="codex_a1")

    (row,) = await get_queries_mod.get_queries()

    assert row.query_id == written["id"]
    assert row.text == written["text"]
    assert row.query_type == written["query_type"]
    assert row.user_id == written["user_id"]
    assert row.dataset_id == dataset_id
    assert row.created_at is not None
    # Selected, not merely filtered on: recall coverage attributes each question
    # row to an agent, and in an "all" run only the row itself knows its session.
    assert row.session_id == "codex_a1"


@pytest.mark.asyncio
async def test_a_row_with_no_session_carries_none(queries_engine):
    """NULL is a real case — rows predating the column, and plain API recalls."""
    await _insert_query(queries_engine, session_id=None)

    (row,) = await get_queries_mod.get_queries()

    assert row.session_id is None


@pytest.mark.asyncio
async def test_window_selects_only_the_labelled_agents_rows(queries_engine):
    """The end-to-end cut: three agents' traffic, one label's rows out."""
    await _insert_query(queries_engine, text="claude asked", session_id="claude_a1")
    await _insert_query(queries_engine, text="codex asked", session_id="codex_a1")
    await _insert_query(queries_engine, text="raw api asked", session_id=None)

    async def texts(agent_label):
        rows = await get_queries_mod.get_queries(
            session_scope=resolve_agent_scope(agent_label, config=_config())
        )
        return {row.text for row in rows}

    assert await texts("claude-code") == {"claude asked"}
    assert await texts("codex") == {"codex asked"}
    # A session id nobody owns, and no session at all, are both "api".
    assert await texts("api") == {"raw api asked"}
    # A configured label with no traffic is empty, not an error.
    assert await texts("cursor") == set()


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_label", [None, "all"])
async def test_window_applies_no_session_predicate_in_all_mode(queries_engine, agent_label):
    """The only scope that returns rows today, and the request default."""
    await _insert_query(queries_engine)

    rows = await get_queries_mod.get_queries(
        session_scope=resolve_agent_scope(agent_label, config=_config())
    )

    assert len(rows) == 1
