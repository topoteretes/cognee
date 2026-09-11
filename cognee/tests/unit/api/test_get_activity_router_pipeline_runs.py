"""Tests for GET /activity/pipeline-runs (SDK-399 operation records).

Two things are checked, and both are needed:

1. **Serialization** — the SDK-399 columns reach the client with their
   NULL/0 distinction intact, and the derived ``kind`` field separates
   pipeline rows from operation rows.
2. **The statement the router built** — the fake session replays its canned
   rows for any statement, so row-level assertions alone cannot tell a correct
   query from a broken one. The visibility predicate, the ordering tiebreaker
   and the ``pipeline_name`` filter are therefore asserted on the generated
   SQLAlchemy constructs, which is what actually fails when the logic regresses.
"""

import importlib
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from sqlalchemy.sql import operators

from cognee.exceptions import CogneeApiError
from cognee.modules.pipelines.models import PipelineRun, PipelineRunStatus
from cognee.modules.users.exceptions import PermissionDeniedError

# The routers package re-exports the factory under the module's own name, so
# ``from ... import get_activity_router`` yields the function, not the module
# whose attributes these tests monkeypatch.
router_module = importlib.import_module("cognee.api.v1.activity.routers.get_activity_router")


def _run(**overrides) -> PipelineRun:
    """A ``pipeline_runs`` row shaped like an operation record.

    Column defaults are applied on INSERT, so an unpersisted instance gets
    every attribute the router reads set explicitly here — otherwise a missing
    key would surface as an ORM ``None`` and hide a real serialization bug.
    """
    fields = dict(
        id=uuid4(),
        created_at=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
        status=None,
        pipeline_run_id=None,
        pipeline_name=None,
        dataset_id=None,
        user_id=None,
        operation_name=None,
        started_at=None,
        ended_at=None,
        outcome=None,
        error_class=None,
        tokens_in=None,
        tokens_out=None,
        origin=None,
        session_id=None,
        parent_operation_id=None,
        background=None,
    )
    fields.update(overrides)
    return PipelineRun(**fields)


def _joined(run: PipelineRun, ds_name=None, owner_id=None, owner_email=None):
    """One result row of the pipeline_runs → datasets → users outer join."""
    return (run, ds_name, owner_id, owner_email)


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class _FakeSession:
    def __init__(self, rows, statements):
        self._rows = rows
        self._statements = statements

    async def execute(self, statement):
        self._statements.append(statement)
        return _FakeResult(self._rows)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False


class _FakeEngine:
    def __init__(self, rows, statements):
        self._rows = rows
        self._statements = statements

    def get_async_session(self):
        return _FakeSession(self._rows, self._statements)


def _stub_engine(monkeypatch, rows) -> list:
    """Swap the router's lazily imported engine for the fake one.

    Returns the list every executed statement is appended to.
    """
    from cognee.infrastructure.databases import relational as relational_module

    statements = []
    monkeypatch.setattr(
        relational_module, "get_relational_engine", lambda: _FakeEngine(rows, statements)
    )
    return statements


def _stub_visibility(monkeypatch, *, visible_user_ids, permitted_dataset_ids):
    async def fake_visible_user_ids(_user_id):
        return list(visible_user_ids)

    async def fake_permitted_dataset_ids(_user_id):
        return list(permitted_dataset_ids)

    monkeypatch.setattr(router_module, "get_visible_user_ids", fake_visible_user_ids)
    monkeypatch.setattr(router_module, "get_permitted_dataset_ids", fake_permitted_dataset_ids)


def _client(user_id) -> TestClient:
    app = FastAPI()
    app.include_router(router_module.get_activity_router(), prefix="/activity")
    app.dependency_overrides[router_module.get_authenticated_user] = lambda: SimpleNamespace(
        id=user_id, tenant_id=None, email="me@example.com"
    )

    # The real app registers this in cognee/api/client.py. Without it a
    # CogneeApiError escapes TestClient as a bare Python exception, and a test
    # asserting only that it propagates would pass even if callers never got
    # the documented status code.
    @app.exception_handler(CogneeApiError)
    async def _cognee_error_handler(_request, exc: CogneeApiError):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": f"{exc.message} [{exc.name}]"},
        )

    return TestClient(app)


def _where_terms(statement) -> dict:
    """Map column name -> operator for the top-level terms of a WHERE clause."""
    clauses = getattr(statement.whereclause, "clauses", [statement.whereclause])
    return {clause.left.name: clause.operator for clause in clauses}


# --------------------------------------------------------------------------- #
# Visibility
# --------------------------------------------------------------------------- #


def test_operation_row_without_dataset_is_visible_to_its_own_user(monkeypatch):
    """recall/prune records carry no dataset_id, so a dataset-only predicate
    would drop them. The response must contain the row *and* the query must
    actually be OR-ed over user_id — the fake session replays rows regardless,
    so only the statement assertion proves the predicate."""
    user_id = uuid4()
    run = _run(user_id=user_id, operation_name="recall", origin="sdk", outcome="succeeded")
    statements = _stub_engine(monkeypatch, [_joined(run)])
    _stub_visibility(monkeypatch, visible_user_ids=[user_id], permitted_dataset_ids=[uuid4()])

    response = _client(user_id).get("/activity/pipeline-runs")

    assert response.status_code == 200
    body = response.json()
    assert [row["id"] for row in body] == [str(run.id)]
    assert body[0]["dataset_id"] is None
    assert body[0]["user_id"] == str(user_id)

    where = statements[0].whereclause
    assert where.operator is operators.or_
    assert _where_terms(statements[0]) == {
        "user_id": operators.in_op,
        "dataset_id": operators.in_op,
    }


def test_own_row_on_an_unreadable_dataset_hides_the_dataset_join_fields(monkeypatch):
    """The user_id term can surface a row whose dataset_id the caller cannot
    read (write-only permission, or a since-revoked read grant) — the row
    itself must still show, since it is the caller's own operation, but the
    joined dataset_name/owner_id/owner_email are read-gated information the
    join alone must not leak just because the row passed the OR predicate."""
    user_id = uuid4()
    unreadable_dataset_id = uuid4()
    owner_id = uuid4()
    run = _run(
        user_id=user_id,
        operation_name="cognify",
        dataset_id=unreadable_dataset_id,
    )
    _stub_engine(monkeypatch, [_joined(run, "secret-dataset", owner_id, "owner@example.com")])
    _stub_visibility(monkeypatch, visible_user_ids=[user_id], permitted_dataset_ids=[uuid4()])

    body = _client(user_id).get("/activity/pipeline-runs").json()

    assert body[0]["dataset_id"] == str(unreadable_dataset_id)
    assert body[0]["dataset_name"] is None
    assert body[0]["owner_id"] is None
    assert body[0]["owner_email"] is None


def test_visibility_is_user_only_when_no_dataset_is_readable(monkeypatch):
    """With no readable datasets there is no dataset term to OR in, and the
    predicate must stay a plain user_id filter rather than an ``IN ()`` that
    matches nothing."""
    user_id = uuid4()
    statements = _stub_engine(monkeypatch, [])
    _stub_visibility(monkeypatch, visible_user_ids=[user_id], permitted_dataset_ids=[])

    response = _client(user_id).get("/activity/pipeline-runs")

    assert response.status_code == 200
    assert response.json() == []
    assert _where_terms(statements[0]) == {"user_id": operators.in_op}


def test_unreadable_dataset_id_returns_403(monkeypatch):
    """An explicit dataset the caller cannot read is a 403 through the global
    CogneeApiError handler, not an empty feed and not an unhandled exception."""
    user_id = uuid4()
    _stub_engine(monkeypatch, [])

    async def denied(_user_id, _permission, _dataset_ids):
        raise PermissionDeniedError("User does not have read permission on dataset")

    monkeypatch.setattr(router_module, "get_specific_user_permission_datasets", denied)

    response = _client(user_id).get("/activity/pipeline-runs", params={"dataset_id": str(uuid4())})

    assert response.status_code == 403
    assert "read permission" in response.json()["detail"]


# --------------------------------------------------------------------------- #
# Serialization of the SDK-399 columns
# --------------------------------------------------------------------------- #


def test_zero_tokens_serialize_as_zero_not_null(monkeypatch):
    """NULL means "not measured" and 0 means "measured zero" (see
    PipelineRun.py). A truthiness check would collapse the two."""
    user_id = uuid4()
    run = _run(user_id=user_id, operation_name="search", tokens_in=0, tokens_out=0)
    _stub_engine(monkeypatch, [_joined(run)])
    _stub_visibility(monkeypatch, visible_user_ids=[user_id], permitted_dataset_ids=[])

    body = _client(user_id).get("/activity/pipeline-runs").json()

    assert body[0]["tokens_in"] == 0
    assert body[0]["tokens_out"] == 0


def test_legacy_row_returns_every_new_column_as_null(monkeypatch):
    """Pre-SDK-399 rows were not backfilled, so every operation column is
    NULL. They must still be present as keys instead of erroring or vanishing."""
    user_id = uuid4()
    dataset_id = uuid4()
    owner_id = uuid4()
    run = _run(
        created_at=datetime(2025, 6, 1, tzinfo=timezone.utc),
        status=PipelineRunStatus.DATASET_PROCESSING_COMPLETED,
        pipeline_run_id=uuid4(),
        pipeline_name="cognify_pipeline",
        dataset_id=dataset_id,
    )
    _stub_engine(monkeypatch, [_joined(run, "docs", owner_id, "owner@example.com")])
    _stub_visibility(monkeypatch, visible_user_ids=[user_id], permitted_dataset_ids=[dataset_id])

    body = _client(user_id).get("/activity/pipeline-runs").json()

    assert body[0]["status"] == "DATASET_PROCESSING_COMPLETED"
    assert body[0]["dataset_name"] == "docs"
    assert body[0]["owner_email"] == "owner@example.com"
    for column in (
        "operation_name",
        "origin",
        "outcome",
        "background",
        "error_class",
        "tokens_in",
        "tokens_out",
        "started_at",
        "ended_at",
        "user_id",
        "session_id",
        "parent_operation_id",
    ):
        assert body[0][column] is None, column


def test_kind_distinguishes_pipeline_rows_from_operation_rows(monkeypatch):
    """pipeline_name is the only reliable marker: operation_name is set on both
    kinds, so clients need the derived field."""
    user_id = uuid4()
    pipeline_run = _run(
        user_id=user_id, pipeline_name="cognify_pipeline", operation_name="cognify_pipeline"
    )
    operation_run = _run(user_id=user_id, operation_name="recall")
    _stub_engine(monkeypatch, [_joined(pipeline_run), _joined(operation_run)])
    _stub_visibility(monkeypatch, visible_user_ids=[user_id], permitted_dataset_ids=[])

    body = _client(user_id).get("/activity/pipeline-runs").json()

    assert [row["kind"] for row in body] == ["pipeline", "operation"]
    assert body[1]["pipeline_name"] is None
    # Same operation_name on both kinds — that is why kind is derived.
    assert body[0]["operation_name"] == "cognify_pipeline"


def test_background_row_keeps_its_outcome(monkeypatch):
    """background=True makes outcome="succeeded" mean accepted-and-started.
    The router documents that ambiguity rather than dropping the field, so
    callers can still deduplicate and audit those rows."""
    user_id = uuid4()
    run = _run(
        user_id=user_id,
        operation_name="cognify",
        background=True,
        outcome="succeeded",
        origin="api",
    )
    _stub_engine(monkeypatch, [_joined(run)])
    _stub_visibility(monkeypatch, visible_user_ids=[user_id], permitted_dataset_ids=[])

    body = _client(user_id).get("/activity/pipeline-runs").json()

    assert body[0]["background"] is True
    assert body[0]["outcome"] == "succeeded"


# --------------------------------------------------------------------------- #
# Paging and filtering
# --------------------------------------------------------------------------- #


def test_limit_and_offset_are_forwarded_with_a_stable_tiebreaker(monkeypatch):
    """created_at alone orders equal timestamps arbitrarily, so OFFSET paging
    over a table that gains a row per operation re-serves rows the client
    already saw. id must be part of the ORDER BY."""
    user_id = uuid4()
    statements = _stub_engine(monkeypatch, [])
    _stub_visibility(monkeypatch, visible_user_ids=[user_id], permitted_dataset_ids=[])

    response = _client(user_id).get("/activity/pipeline-runs", params={"limit": 10, "offset": 20})

    assert response.status_code == 200
    # _limit/_offset are SQLAlchemy internals, but they are the only way to read
    # back the values without a live database.
    assert (statements[0]._limit, statements[0]._offset) == (10, 20)
    assert "ORDER BY pipeline_runs.created_at DESC, pipeline_runs.id DESC" in str(statements[0])


@pytest.mark.parametrize(
    "params",
    [{"limit": 0}, {"limit": 501}, {"offset": -1}],
    ids=["limit-0", "limit-501", "offset-negative"],
)
def test_out_of_range_pagination_is_rejected(monkeypatch, params):
    """ge/le bounds keep a caller from asking for a whole-table scan (or an
    empty page) instead of the endpoint silently clamping."""
    user_id = uuid4()
    statements = _stub_engine(monkeypatch, [])
    _stub_visibility(monkeypatch, visible_user_ids=[user_id], permitted_dataset_ids=[])

    response = _client(user_id).get("/activity/pipeline-runs", params=params)

    assert response.status_code == 422
    assert statements == []


def test_pipeline_name_filters_by_exact_match(monkeypatch):
    user_id = uuid4()
    statements = _stub_engine(monkeypatch, [])
    _stub_visibility(monkeypatch, visible_user_ids=[user_id], permitted_dataset_ids=[])

    response = _client(user_id).get(
        "/activity/pipeline-runs", params={"pipeline_name": "cognify_pipeline"}
    )

    assert response.status_code == 200
    # The filter is AND-ed on top of the visibility predicate, so the
    # top-level clause list carries both.
    name_terms = [
        clause
        for clause in statements[0].whereclause.clauses
        if getattr(clause, "left", None) is not None and clause.left.name == "pipeline_name"
    ]
    assert len(name_terms) == 1
    assert name_terms[0].operator is operators.eq
    assert name_terms[0].right.value == "cognify_pipeline"


def test_no_pipeline_name_leaves_the_filter_out(monkeypatch):
    """Operation records have no pipeline_name, so an always-applied filter
    would hide them entirely."""
    user_id = uuid4()
    statements = _stub_engine(monkeypatch, [])
    _stub_visibility(monkeypatch, visible_user_ids=[user_id], permitted_dataset_ids=[])

    _client(user_id).get("/activity/pipeline-runs")

    assert "pipeline_name" not in _where_terms(statements[0])
