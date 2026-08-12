"""Guards on the recall-coverage replay: whose user, and no side effects.

Two things here are load-bearing beyond ordinary correctness.

**A replay must not write search history.** ``search()`` and ``recall()`` both
call ``log_search_history`` after searching, and every ``queries`` row they write
lands inside the *next* run's window and is replayed again — a feedback loop that
inflates every count for free. ``COGNEE_LOG_SEARCH_HISTORY`` cannot protect
against it, because ``_LOG_ENABLED`` is a module-level constant read at import.
So ``test_replay_writes_no_search_history`` runs the **real**
``authorized_search`` with ``log_query``, ``log_result`` and
``log_search_history`` all replaced by raisers, and with ``search()`` and
``recall()`` replaced by raisers too.

**A replay must run as the row's own user.** A run is tenant-wide, so most rows
belong to somebody else; replaying Ben's question as Anna answers it out of
Anna's brains and labels the result with Ben's ``user_id``.
"""

import asyncio
import importlib
from types import SimpleNamespace
from uuid import uuid4

import pytest

from cognee.modules.recall_coverage.config import RecallCoverageConfig
from cognee.modules.recall_coverage.dedup import DedupedQuestion
from cognee.modules.recall_coverage.replay import (
    REPLAY_QUERY_TYPE,
    ReplayedRow,
    ReplayUserCache,
    flatten_context,
    replay_question,
    replay_questions,
    without_session_user,
)
from cognee.modules.recall_coverage.types import CoverageParams, QuestionSource
from cognee.modules.search.models.SearchResultPayload import SearchResultPayload
from cognee.modules.search.types import SearchType

# ``importlib`` rather than ``from ... import``: the packages re-export functions
# that shadow the same-named submodule, and these tests monkeypatch attributes on
# the modules themselves.
context_module = importlib.import_module("cognee.context_global_variables")
replay_module = importlib.import_module("cognee.modules.recall_coverage.replay")
search_module = importlib.import_module("cognee.modules.search.methods.search")
log_query_module = importlib.import_module("cognee.modules.search.operations.log_query")
log_result_module = importlib.import_module("cognee.modules.search.operations.log_result")
log_history_module = importlib.import_module("cognee.modules.search.operations.log_search_history")


def _params(**overrides) -> CoverageParams:
    """Parameters from the packaged defaults, never a developer's ``.env``."""
    return CoverageParams.from_config(RecallCoverageConfig(_env_file=None), **overrides)


def _question(
    text: str = "Where are the runbooks?",
    *,
    user_id=None,
    dataset_id=None,
    source: str = QuestionSource.OBSERVED.value,
) -> DedupedQuestion:
    return DedupedQuestion(
        text=text,
        user_id=user_id or uuid4(),
        dataset_id=dataset_id,
        source=source,
        was_asked=source == QuestionSource.OBSERVED.value,
        occurrence_count=1,
        first_asked_at=None,
        last_asked_at=None,
        curated_question_id=None,
        canonical_index=0,
        ask_indices=[0],
        query_ids=[],
    )


def _payload(context, *, dataset_name="infra-docs", dataset_id=None) -> SearchResultPayload:
    return SearchResultPayload(
        result_object=None,
        context=context,
        completion=None,
        search_type=SearchType.GRAPH_COMPLETION,
        only_context=True,
        dataset_name=dataset_name,
        dataset_id=dataset_id,
    )


async def _resolved(value):
    """A ready-made awaitable, for stubbing an async loader with a lambda."""
    return value


class _RecordingSearch:
    """Stands in for ``authorized_search``, recording exactly how it was called."""

    def __init__(self, payloads=None):
        self.calls: list[dict] = []
        self.session_users: list[object] = []
        self._payloads = payloads if payloads is not None else [_payload("some context")]

    async def __call__(self, **kwargs):
        self.calls.append(kwargs)
        self.session_users.append(context_module.session_user.get())
        await asyncio.sleep(0)
        return list(self._payloads)


# --------------------------------------------------------------------------
# The row's own user
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_replay_searches_as_the_rows_own_user_not_the_caller():
    caller = SimpleNamespace(id=uuid4(), tenant_id=None)
    row_user_id = uuid4()
    row_user = SimpleNamespace(id=row_user_id, tenant_id=None)

    search = _RecordingSearch()
    cache = ReplayUserCache(loader=lambda user_id: _resolved(row_user))

    rows = await replay_questions(
        [_question(user_id=row_user_id)],
        params=_params(),
        user_cache=cache,
        search=search,
    )

    assert len(rows) == 1
    assert search.calls[0]["user"] is row_user
    assert search.calls[0]["user"] is not caller
    assert search.calls[0]["user"].id == row_user_id


@pytest.mark.asyncio
async def test_each_distinct_user_is_loaded_once_per_run():
    anna, ben = uuid4(), uuid4()
    loaded: list = []

    async def loader(user_id):
        loaded.append(user_id)
        return SimpleNamespace(id=user_id, tenant_id=None)

    cache = ReplayUserCache(loader=loader)
    questions = [
        _question("a", user_id=anna),
        _question("b", user_id=anna),
        _question("c", user_id=ben),
        _question("d", user_id=anna),
    ]

    await replay_questions(questions, params=_params(), user_cache=cache, search=_RecordingSearch())

    assert sorted(loaded, key=str) == sorted([anna, ben], key=str)
    assert cache.loaded_count == 2


@pytest.mark.asyncio
async def test_a_missing_user_is_recorded_as_that_rows_error_only_once():
    attempts: list = []

    async def loader(user_id):
        attempts.append(user_id)
        raise RuntimeError("Could not find user")

    missing = uuid4()
    cache = ReplayUserCache(loader=loader)
    search = _RecordingSearch()

    rows = await replay_questions(
        [_question("a", user_id=missing), _question("b", user_id=missing)],
        params=_params(),
        user_cache=cache,
        search=search,
    )

    assert len(attempts) == 1
    assert search.calls == []
    assert all(row.error == "RuntimeError: Could not find user" for row in rows)
    assert all(row.retrieval_context is None for row in rows)


# --------------------------------------------------------------------------
# Side-effect guards
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_replay_clears_session_user_and_restores_it_afterwards():
    caller = SimpleNamespace(id=uuid4(), tenant_id=None)
    token = context_module.session_user.set(caller)
    try:
        search = _RecordingSearch()
        await replay_questions(
            [_question()],
            params=_params(),
            user_cache=ReplayUserCache(loader=lambda user_id: _resolved(caller)),
            search=search,
        )

        # ``GraphCompletionRetriever._use_session_cache()`` is
        # ``bool(user_id and CacheConfig().caching)``, so a non-None session user
        # here would write a session-cache QA turn per replayed question.
        assert search.session_users == [None]
        assert context_module.session_user.get() is caller
    finally:
        context_module.session_user.reset(token)


def test_without_session_user_restores_the_previous_value_on_error():
    caller = SimpleNamespace(id=uuid4())
    token = context_module.session_user.set(caller)
    try:
        with pytest.raises(RuntimeError):
            with without_session_user():
                assert context_module.session_user.get() is None
                raise RuntimeError("boom")
        assert context_module.session_user.get() is caller
    finally:
        context_module.session_user.reset(token)


@pytest.mark.asyncio
async def test_replay_asks_for_context_only_so_no_completion_is_billed():
    search = _RecordingSearch()
    await replay_questions(
        [_question()],
        params=_params(replay_top_k=7),
        user_cache=ReplayUserCache(loader=lambda user_id: _resolved(SimpleNamespace(id=user_id))),
        search=search,
    )

    call = search.calls[0]
    # The completion is a judge-phase call made only for rows that scored above
    # zero; generating one here would bill every row including those.
    assert call["only_context"] is True
    assert call["session_id"] is None
    assert call["top_k"] == 7
    assert call["query_type"] is REPLAY_QUERY_TYPE is SearchType.GRAPH_COMPLETION


@pytest.mark.asyncio
async def test_replay_writes_no_search_history(monkeypatch):
    """The real ``authorized_search``, with every logging path armed to explode."""
    dataset = SimpleNamespace(id=uuid4(), name="infra-docs", owner_id=uuid4(), tenant_id=None)

    def _explode(*args, **kwargs):
        raise AssertionError("recall-coverage replay must not write search history")

    # Every writer, patched where it is defined and where it is imported.
    monkeypatch.setattr(log_query_module, "log_query", _explode)
    monkeypatch.setattr(log_result_module, "log_result", _explode)
    monkeypatch.setattr(log_history_module, "log_query", _explode)
    monkeypatch.setattr(log_history_module, "log_result", _explode)
    monkeypatch.setattr(log_history_module, "log_search_history", _explode)
    monkeypatch.setattr(search_module, "log_search_history", _explode)
    # Neither of the two logging entry points may be entered.
    monkeypatch.setattr(search_module, "search", _explode)

    recall_module = importlib.import_module("cognee.api.v1.recall.recall")
    monkeypatch.setattr(recall_module, "recall", _explode)

    # Stub out only the infrastructure below ``authorized_search``, so
    # ``authorized_search`` and ``search_in_datasets_context`` run for real.
    async def fake_authorized_datasets(datasets, permission_type, user):
        assert permission_type == "read"
        return [dataset]

    class _FakeContext:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc_info):
            return None

    class _FakeGraphEngine:
        async def is_empty(self):
            return False

    retriever_calls: list[dict] = []

    async def fake_get_retriever_output(**kwargs):
        retriever_calls.append(kwargs)
        return _payload("the runbooks live in infra-docs", dataset_id=dataset.id)

    monkeypatch.setattr(search_module, "get_authorized_existing_datasets", fake_authorized_datasets)
    monkeypatch.setattr(search_module, "backend_access_control_enabled", lambda: True)
    monkeypatch.setattr(
        search_module, "set_database_global_context_variables", lambda *a, **k: _FakeContext()
    )
    monkeypatch.setattr(search_module, "get_graph_engine", lambda: _resolved(_FakeGraphEngine()))
    monkeypatch.setattr(search_module, "get_retriever_output", fake_get_retriever_output)

    row_user = SimpleNamespace(id=uuid4(), tenant_id=None)
    rows = await replay_questions(
        [_question(dataset_id=dataset.id, user_id=row_user.id)],
        params=_params(),
        user_cache=ReplayUserCache(loader=lambda user_id: _resolved(row_user)),
    )

    assert len(retriever_calls) == 1
    assert retriever_calls[0]["only_context"] is True
    assert rows[0].error is None
    assert rows[0].retrieval_context == "the runbooks live in infra-docs"
    assert rows[0].dataset_name == "infra-docs"


@pytest.mark.asyncio
async def test_replay_stamps_no_last_accessed_on_the_data_it_retrieves(monkeypatch):
    """The real ``update_node_access_timestamps``, with ``ENABLE_LAST_ACCESSED`` on.

    The call is not behind ``only_context``, and ``last_accessed`` drives
    ``cleanup_unused_data``'s retention cutoff and the activity feed — so a run
    that stamped it would keep stale documents alive and report access no human
    made. The third write path in the search call, after the ``queries`` rows and
    the session cache.
    """
    monkeypatch.setenv("ENABLE_LAST_ACCESSED", "true")

    tracking = importlib.import_module("cognee.modules.retrieval.utils.access_tracking")

    def _explode(*args, **kwargs):
        raise AssertionError("recall-coverage replay must not stamp last_accessed")

    # Everything the real function would reach once it decided to proceed.
    monkeypatch.setattr(tracking, "_extract_access_node_ids", _explode)
    monkeypatch.setattr(tracking, "_find_origin_documents_via_projection", _explode)
    monkeypatch.setattr(tracking, "_update_sql_records", _explode)

    async def fake_search(**kwargs):
        # Stands in for get_retriever_output, which calls this unconditionally.
        await tracking.update_node_access_timestamps({"chunks": [{"id": str(uuid4())}]})
        return [_payload("the runbooks live in infra-docs")]

    rows = await replay_questions(
        [_question()],
        params=_params(),
        user_cache=ReplayUserCache(loader=lambda user_id: _resolved(SimpleNamespace(id=user_id))),
        search=fake_search,
    )

    assert rows[0].error is None

    # And the suppression is scoped: outside the replay, tracking still runs.
    with pytest.raises(AssertionError, match="must not stamp"):
        await tracking.update_node_access_timestamps({"chunks": [{"id": str(uuid4())}]})


# --------------------------------------------------------------------------
# Dataset scoping and context shaping
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dataset_scoping_branches_on_the_rows_dataset_not_its_source():
    dataset_id = uuid4()
    search = _RecordingSearch()
    cache = ReplayUserCache(loader=lambda user_id: _resolved(SimpleNamespace(id=user_id)))

    questions = [
        # A curated row that merged into a dataset partition replays against that
        # dataset — the same memory the observed rows there were answered from.
        _question("curated, merged", dataset_id=dataset_id, source=QuestionSource.CURATED.value),
        # A curated row that merged with nothing has no dataset, so it replays
        # against everything the user can read, like the original ask did.
        _question("curated, alone", dataset_id=None, source=QuestionSource.CURATED.value),
        _question("observed, scoped", dataset_id=dataset_id),
        _question("observed, unscoped", dataset_id=None),
    ]

    await replay_questions(questions, params=_params(), user_cache=cache, search=search)

    by_text = {call["query_text"]: call["dataset_ids"] for call in search.calls}
    assert by_text["curated, merged"] == [dataset_id]
    assert by_text["observed, scoped"] == [dataset_id]
    assert by_text["curated, alone"] is None
    assert by_text["observed, unscoped"] is None


def test_flatten_context_joins_a_list_instead_of_storing_a_python_repr():
    assert flatten_context(["first", "second"]) == "first\n\nsecond"
    assert "['" not in flatten_context(["first", "second"])
    assert flatten_context("  plain  ") == "plain"
    assert flatten_context(None) == ""
    assert flatten_context([]) == ""
    assert flatten_context(["", "  "]) == ""


@pytest.mark.asyncio
async def test_an_empty_context_is_stored_as_null_not_as_an_empty_string():
    for empty in (None, "", [], ["", "   "]):
        search = _RecordingSearch(payloads=[_payload(empty)])
        rows = await replay_questions(
            [_question()],
            params=_params(),
            user_cache=ReplayUserCache(
                loader=lambda user_id: _resolved(SimpleNamespace(id=user_id))
            ),
            search=search,
        )
        assert rows[0].retrieval_context is None
        assert rows[0].has_context is False
        assert rows[0].error is None


@pytest.mark.asyncio
async def test_zero_payloads_is_an_empty_context_not_an_error():
    """An unreadable dataset yields no payloads at all, and no exception."""
    search = _RecordingSearch(payloads=[])
    rows = await replay_questions(
        [_question(dataset_id=uuid4())],
        params=_params(),
        user_cache=ReplayUserCache(loader=lambda user_id: _resolved(SimpleNamespace(id=user_id))),
        search=search,
    )

    assert rows[0] == ReplayedRow(
        retrieval_context=None, dataset_name=None, payload_count=0, error=None
    )


@pytest.mark.asyncio
async def test_the_replay_carries_the_whole_context_whatever_the_storage_bound_is():
    """``store_context_max_chars`` bounds the *column*, never what the judge sees.

    Truncating here would make a storage knob decide the coverage score — and at
    ``0`` would score every row 0 with no LLM call, reporting "memory answered
    nothing" as a measurement. The bound is applied by ``build_rows``.
    """
    search = _RecordingSearch(payloads=[_payload("x" * 500)])
    rows = await replay_questions(
        [_question()],
        params=_params(store_context_max_chars=0),
        user_cache=ReplayUserCache(loader=lambda user_id: _resolved(SimpleNamespace(id=user_id))),
        search=search,
    )

    assert rows[0].retrieval_context == "x" * 500
    assert rows[0].has_context


@pytest.mark.asyncio
async def test_several_payloads_join_their_contexts_and_name_no_single_dataset():
    search = _RecordingSearch(
        payloads=[
            _payload("from infra", dataset_name="infra-docs"),
            _payload("from billing", dataset_name="billing"),
        ]
    )
    rows = await replay_questions(
        [_question(dataset_id=None)],
        params=_params(),
        user_cache=ReplayUserCache(loader=lambda user_id: _resolved(SimpleNamespace(id=user_id))),
        search=search,
    )

    assert rows[0].retrieval_context == "from infra\n\nfrom billing"
    assert rows[0].dataset_name is None
    assert rows[0].payload_count == 2


# --------------------------------------------------------------------------
# Failure isolation and concurrency
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_one_failing_row_does_not_fail_the_run():
    async def flaky(**kwargs):
        if kwargs["query_text"] == "boom":
            raise RuntimeError("retriever exploded")
        return [_payload("fine")]

    rows = await replay_questions(
        [_question("boom"), _question("ok")],
        params=_params(),
        user_cache=ReplayUserCache(loader=lambda user_id: _resolved(SimpleNamespace(id=user_id))),
        search=flaky,
    )

    # Class-prefixed and bounded: this string is persisted and returned by the API.
    assert rows[0].error == "RuntimeError: retriever exploded"
    assert rows[0].retrieval_context is None
    assert rows[1].error is None
    assert rows[1].retrieval_context == "fine"


@pytest.mark.asyncio
async def test_replay_is_bounded_by_replay_max_concurrent():
    in_flight = 0
    peak = 0

    async def slow(**kwargs):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0)
        in_flight -= 1
        return [_payload("context")]

    questions = [_question(f"q{index}") for index in range(12)]
    rows = await replay_questions(
        questions,
        params=_params(replay_max_concurrent=3),
        user_cache=ReplayUserCache(loader=lambda user_id: _resolved(SimpleNamespace(id=user_id))),
        search=slow,
    )

    assert len(rows) == 12
    assert peak <= 3


@pytest.mark.asyncio
async def test_replaying_nothing_calls_nothing():
    search = _RecordingSearch()
    assert await replay_questions([], params=_params(), search=search) == []
    assert search.calls == []


@pytest.mark.asyncio
async def test_replay_question_returns_rows_index_aligned_with_the_questions():
    async def by_text(**kwargs):
        return [_payload(f"context for {kwargs['query_text']}")]

    questions = [_question(f"q{index}") for index in range(5)]
    rows = await replay_questions(
        questions,
        params=_params(),
        user_cache=ReplayUserCache(loader=lambda user_id: _resolved(SimpleNamespace(id=user_id))),
        search=by_text,
    )

    assert [row.retrieval_context for row in rows] == [
        f"context for q{index}" for index in range(5)
    ]


@pytest.mark.asyncio
async def test_replay_question_is_usable_on_its_own():
    search = _RecordingSearch()
    user = SimpleNamespace(id=uuid4(), tenant_id=None)

    row = await replay_question(
        _question(user_id=user.id),
        user,
        replay_top_k=15,
        search=search,
    )

    assert row.retrieval_context == "some context"
    assert search.calls[0]["user"] is user
