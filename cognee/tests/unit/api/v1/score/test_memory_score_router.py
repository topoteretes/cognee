"""Router tests for /api/v1/score, focused on authorisation and spend limits.

Three things are asserted here because getting any of them wrong leaks another
dataset's contents or spends unbounded LLM budget:

* starting a run requires READ permission on the dataset, not just shared tenancy;
* reading a run requires the same, and answers 404 rather than confirming the id;
* the request parameters that drive LLM spend are bounded by the API.

Asserted against the REAL app from cognee.api.client, so the status codes here are
the ones callers actually receive.
"""

from importlib import import_module
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from cognee.modules.users.exceptions import PermissionDeniedError

router_module = import_module("cognee.api.v1.score.routers.get_score_router")
# The handlers import their helpers from the methods PACKAGE at call time, so the
# package namespace is what has to be patched, not the defining submodule.
methods_module = import_module("cognee.modules.memory_score.methods")
run_module = import_module("cognee.modules.memory_score.methods.run_memory_score")

# The REAL app, not a bare FastAPI() with the router bolted on. A hand-built app
# has none of the app-wide exception handlers, which silently changes the status
# codes under test: client.py maps every RequestValidationError to 400, so a
# bare app reports an out-of-range cap as 422 while production reports 400. Tests
# that assert against a hand-built app therefore certify statuses no caller ever
# sees. Using the real app also covers the mount prefix and route ordering
# (/latest must be matched before /{run_id}).
client_module = import_module("cognee.api.client")

TENANT_ID = uuid4()
USER_ID = uuid4()

# What the app-wide RequestValidationError handler turns a rejected body into.
VALIDATION_ERROR_STATUS = 400


def _client(monkeypatch, tenant_id=TENANT_ID) -> TestClient:
    app = client_module.app
    # setitem, not assignment: dependency_overrides lives on the shared app object,
    # and monkeypatch reverts this entry after the test instead of leaking it.
    monkeypatch.setitem(
        app.dependency_overrides,
        router_module.get_authenticated_user,
        lambda: SimpleNamespace(id=USER_ID, tenant_id=tenant_id),
    )
    monkeypatch.setattr(router_module, "send_telemetry", lambda *args, **kwargs: None)
    return TestClient(app, raise_server_exceptions=False)


def _run_row(dataset_id, tenant_id=TENANT_ID, run_id=None):
    return SimpleNamespace(
        id=run_id or uuid4(),
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        status="COMPLETED",
        below_data_floor=False,
        floor_reason=None,
        schema_defined=False,
        overall_accuracy=0.9,
        synthetic_question_count=1,
        real_question_count=0,
        topics=[],
        created_at=None,
        completed_at=None,
    )


# --------------------------------------------------------------------------
# POST /score — dataset authorisation
# --------------------------------------------------------------------------


def test_start_run_is_refused_without_read_permission_on_the_dataset(monkeypatch):
    """Sharing a tenant is not permission to score a dataset.

    A run returns expected answers lifted verbatim from the dataset's chunk text
    plus full recall output over it, and it enters the dataset context as the
    dataset OWNER, so database-level isolation never sees the caller. This check
    is the only thing standing between a tenant member and a colleague's data.
    """
    started = []

    async def deny(tenant_id, dataset_id, requesting_user_id=None):
        assert requesting_user_id == USER_ID, "the CALLER is who must be authorised"
        raise PermissionDeniedError("Request owner does not have necessary permission: [read]")

    async def fail_if_run(**kwargs):
        started.append(kwargs)
        raise AssertionError("the run must not start")

    monkeypatch.setattr(methods_module, "resolve_memory_score_dataset", deny)
    monkeypatch.setattr(methods_module, "run_memory_score", fail_if_run)
    monkeypatch.setattr(methods_module, "create_memory_score_run", fail_if_run)

    response = _client(monkeypatch).post("/api/v1/score", json={"dataset_id": str(uuid4())})

    assert response.status_code == 403
    assert started == []


def test_start_run_passes_the_caller_identity_to_the_resolver(monkeypatch):
    """The permission check is worthless if the handler forgets to pass user.id."""
    resolved = {}
    dataset_id = uuid4()
    run_id = uuid4()

    async def resolve(tenant_id, requested_dataset_id, requesting_user_id=None):
        resolved.update(
            tenant_id=tenant_id,
            dataset_id=requested_dataset_id,
            requesting_user_id=requesting_user_id,
        )
        return SimpleNamespace(id=requested_dataset_id, owner_id=uuid4())

    async def no_active_run(_tenant_id):
        return None

    async def create_run(_tenant_id, _dataset_id, _user_id):
        return run_id

    async def never_run(**_kwargs):
        return run_id

    monkeypatch.setattr(methods_module, "resolve_memory_score_dataset", resolve)
    monkeypatch.setattr(methods_module, "find_active_memory_score_run", no_active_run)
    monkeypatch.setattr(methods_module, "create_memory_score_run", create_run)
    monkeypatch.setattr(methods_module, "run_memory_score", never_run)

    response = _client(monkeypatch).post("/api/v1/score", json={"dataset_id": str(dataset_id)})

    assert response.status_code == 200
    assert response.json() == {"run_id": str(run_id)}
    assert resolved == {
        "tenant_id": TENANT_ID,
        "dataset_id": dataset_id,
        "requesting_user_id": USER_ID,
    }


# --------------------------------------------------------------------------
# POST /score — spend ceilings
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field, value",
    [
        ("synthetic_target", run_module.MAX_SYNTHETIC_TARGET + 1),
        ("synthetic_target", 1_000_000),
        ("synthetic_target", -1),
        ("real_question_limit", run_module.MAX_REAL_QUESTION_LIMIT + 1),
    ],
)
def test_out_of_range_spend_parameters_are_refused(monkeypatch, field, value):
    """An uncapped question count is an uncapped LLM bill."""

    async def fail(*_args, **_kwargs):
        raise AssertionError("validation must reject the request before any work")

    monkeypatch.setattr(methods_module, "resolve_memory_score_dataset", fail)
    monkeypatch.setattr(methods_module, "run_memory_score", fail)

    response = _client(monkeypatch).post(
        "/api/v1/score", json={"dataset_id": str(uuid4()), field: value}
    )

    assert response.status_code == VALIDATION_ERROR_STATUS


def test_api_advertises_the_same_ceilings_it_enforces(monkeypatch):
    """The documented limit and the enforced one are one constant, not two."""
    schema = _client(monkeypatch).app.openapi()
    properties = schema["components"]["schemas"]["StartScoreRunPayload"]["properties"]

    assert properties["synthetic_target"]["maximum"] == run_module.MAX_SYNTHETIC_TARGET
    assert properties["real_question_limit"]["maximum"] == run_module.MAX_REAL_QUESTION_LIMIT


# --------------------------------------------------------------------------
# GET /score/{run_id} and /score/latest — read authorisation
# --------------------------------------------------------------------------


def test_reading_a_run_over_an_unreadable_dataset_is_404(monkeypatch):
    """404, not 403: the endpoint never confirms the run id exists."""
    run = _run_row(dataset_id=uuid4())

    async def get_run(_run_id):
        return run

    async def no_readable_datasets(_user_id):
        return set()

    async def unexpected_questions(_run_id):
        raise AssertionError("questions must not be read for an unreadable run")

    monkeypatch.setattr(methods_module, "get_memory_score_run", get_run)
    monkeypatch.setattr(methods_module, "readable_dataset_ids", no_readable_datasets)
    monkeypatch.setattr(methods_module, "get_memory_score_questions", unexpected_questions)

    response = _client(monkeypatch).get(f"/api/v1/score/{run.id}")

    assert response.status_code == 404
    assert response.json() == {"error": "Memory score run not found"}


def test_another_tenants_run_is_404(monkeypatch):
    run = _run_row(dataset_id=uuid4(), tenant_id=uuid4())

    async def get_run(_run_id):
        return run

    async def fail(_user_id):
        raise AssertionError("tenancy is checked before permission")

    monkeypatch.setattr(methods_module, "get_memory_score_run", get_run)
    monkeypatch.setattr(methods_module, "readable_dataset_ids", fail)

    response = _client(monkeypatch).get(f"/api/v1/score/{run.id}")

    assert response.status_code == 404


def test_reading_a_run_over_a_readable_dataset_succeeds(monkeypatch):
    run = _run_row(dataset_id=uuid4())

    async def get_run(_run_id):
        return run

    async def readable(_user_id):
        return {str(run.dataset_id)}

    async def no_questions(_run_id):
        return []

    monkeypatch.setattr(methods_module, "get_memory_score_run", get_run)
    monkeypatch.setattr(methods_module, "readable_dataset_ids", readable)
    monkeypatch.setattr(methods_module, "get_memory_score_questions", no_questions)

    response = _client(monkeypatch).get(f"/api/v1/score/{run.id}")

    assert response.status_code == 200
    assert response.json()["run_id"] == str(run.id)
    assert response.json()["judged_synthetic_question_count"] == 0


def test_latest_is_scoped_to_the_datasets_the_caller_can_read(monkeypatch):
    """Otherwise a colleague's newer run puts their dataset's contents at /latest."""
    readable_ids = {str(uuid4())}
    passed = {}

    async def readable(_user_id):
        return readable_ids

    async def latest(tenant_id, dataset_ids=None):
        passed.update(tenant_id=tenant_id, dataset_ids=dataset_ids)
        return None

    monkeypatch.setattr(methods_module, "readable_dataset_ids", readable)
    monkeypatch.setattr(methods_module, "get_latest_memory_score_run", latest)

    response = _client(monkeypatch).get("/api/v1/score/latest")

    assert response.status_code == 404
    assert passed == {"tenant_id": TENANT_ID, "dataset_ids": readable_ids}


def test_no_tenant_is_rejected_before_any_work(monkeypatch):
    async def fail(*_args, **_kwargs):
        raise AssertionError("nothing should run without a tenant")

    monkeypatch.setattr(methods_module, "get_latest_memory_score_run", fail)
    monkeypatch.setattr(methods_module, "readable_dataset_ids", fail)

    response = _client(monkeypatch, tenant_id=None).get("/api/v1/score/latest")

    assert response.status_code == 400
