"""
Consolidated tests for API endpoint error responses and basic endpoint functionality.

Tests cover:
- Error handling (PipelineRunErrored → 422, PermissionDenied → 403, etc.)
- Basic endpoint functionality (happy path with mocked backends)
"""

import importlib
from types import SimpleNamespace
from uuid import uuid4
from unittest.mock import AsyncMock

import pytest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from cognee.modules.users.methods import get_authenticated_user
from cognee.modules.pipelines.models import PipelineRunErrored, PipelineRunCompleted
from cognee.modules.users.exceptions.exceptions import PermissionDeniedError
from cognee.infrastructure.llm.exceptions import LLMPaymentRequiredError
from cognee.api.v1.add.routers.get_add_router import get_add_router
from cognee.api.v1.cognify.routers.get_cognify_router import get_cognify_router
from cognee.api.v1.memify.routers.get_memify_router import get_memify_router
from cognee.api.v1.improve.routers.get_improve_router import get_improve_router
from cognee.api.v1.recall.routers.get_recall_router import get_recall_router
from cognee.api.v1.remember.routers.get_remember_router import get_remember_router
from cognee.api.v1.search.routers.get_search_router import get_search_router
from cognee.api.v1.update.routers.get_update_router import get_update_router
from cognee.exceptions import CogneeApiError, CogneeValidationError


MOCK_USER = SimpleNamespace(id=uuid4(), email="test@example.com", is_active=True, tenant_id=uuid4())
MOCK_DATASET_ID = uuid4()
MOCK_PIPELINE_RUN_ID = uuid4()


def _make_completed(**kwargs):
    defaults = dict(
        pipeline_run_id=MOCK_PIPELINE_RUN_ID,
        dataset_id=MOCK_DATASET_ID,
        dataset_name="test_dataset",
        status="completed",
    )
    defaults.update(kwargs)
    return PipelineRunCompleted(**defaults)


def _make_errored(error="pipeline failed", **kwargs):
    defaults = dict(
        pipeline_run_id=MOCK_PIPELINE_RUN_ID,
        dataset_id=MOCK_DATASET_ID,
        dataset_name="test_dataset",
        status="errored",
        error=error,
    )
    defaults.update(kwargs)
    return PipelineRunErrored(**defaults)


@pytest.fixture
def app():
    app = FastAPI()
    app.include_router(get_add_router(), prefix="/add")
    app.include_router(get_cognify_router(), prefix="/cognify")
    app.include_router(get_improve_router(), prefix="/improve")
    app.include_router(get_memify_router(), prefix="/memify")
    app.include_router(get_recall_router(), prefix="/recall")
    app.include_router(get_remember_router(), prefix="/remember")
    app.include_router(get_search_router(), prefix="/search")
    app.include_router(get_update_router(), prefix="/update")

    async def override_user():
        return MOCK_USER

    app.dependency_overrides[get_authenticated_user] = override_user

    # The real app registers this in cognee/api/client.py. Session-dataset 409s
    # rely on it: the routers re-raise them so this handler can return the
    # exception's own actionable message at its own status code.
    @app.exception_handler(CogneeApiError)
    async def cognee_error_handler(_, exc: CogneeApiError):
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": f"{exc.message} [{exc.name}]"},
        )

    return app


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture(autouse=True)
def _restore_api_package_functions():
    """Undo the bare ``pkg.fn = AsyncMock(...)`` assignments the tests below make.

    Without this, the last-installed mock leaks into every later test module
    that resolves these package attributes lazily — e.g. remember()-based tests
    failed on dev CI with whichever side_effect ran last here
    (LLMPaymentRequiredError from the 402 test).
    """
    import cognee.api.v1.add as add_pkg
    import cognee.api.v1.cognify as cognify_pkg
    import cognee.api.v1.memify as memify_pkg
    import cognee.api.v1.search as search_pkg
    import cognee.api.v1.update as update_pkg

    targets = [
        (add_pkg, "add"),
        (cognify_pkg, "cognify"),
        (memify_pkg, "memify"),
        (search_pkg, "search"),
        (update_pkg, "update"),
    ]
    missing = object()
    originals = [(pkg, name, getattr(pkg, name, missing)) for pkg, name in targets]
    yield
    for pkg, name, original in originals:
        if original is missing:
            # The attribute only exists because a test assigned it — remove it.
            if hasattr(pkg, name):
                delattr(pkg, name)
        else:
            setattr(pkg, name, original)


# ---------------------------------------------------------------------------
# Add endpoint
# ---------------------------------------------------------------------------


class TestAddEndpoint:
    def test_add_missing_dataset_returns_400(self, client):
        resp = client.post(
            "/add",
            data={},
            files={"data": ("x.txt", b"hello", "text/plain")},
        )
        assert resp.status_code == 400

    def test_add_pipeline_errored_returns_500(self, client):
        import cognee.api.v1.add as add_pkg

        add_pkg.add = AsyncMock(return_value=_make_errored())

        resp = client.post(
            "/add",
            data={"datasetName": "test_dataset"},
            files={"data": ("x.txt", b"hello", "text/plain")},
        )
        assert resp.status_code == 500
        body = resp.json()
        assert body["error"] == "Pipeline run errored"
        assert "detail" in body

    def test_add_success_returns_200(self, client):
        import cognee.api.v1.add as add_pkg

        completed = _make_completed()
        add_pkg.add = AsyncMock(return_value=completed)

        resp = client.post(
            "/add",
            data={"datasetName": "test_dataset"},
            files={"data": ("test.txt", b"Cognee is an AI memory platform.", "text/plain")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "completed"
        assert body["dataset_name"] == "test_dataset"

    def test_add_internal_error_returns_500(self, client):
        import cognee.api.v1.add as add_pkg

        add_pkg.add = AsyncMock(side_effect=RuntimeError("unexpected"))

        resp = client.post(
            "/add",
            data={"datasetName": "test_dataset"},
            files={"data": ("x.txt", b"hello", "text/plain")},
        )
        assert resp.status_code == 500
        assert resp.json()["error"] == "Internal server error"


# ---------------------------------------------------------------------------
# Remember endpoint
# ---------------------------------------------------------------------------


class TestRememberEndpoint:
    def test_remember_passes_chunk_size(self, client, monkeypatch):
        remember_pkg = importlib.import_module("cognee.api.v1.remember")

        class RememberResultStub:
            status = "completed"

            def to_dict(self):
                return {"status": "completed", "dataset_name": "test_dataset"}

        remember = AsyncMock(return_value=RememberResultStub())
        monkeypatch.setattr(remember_pkg, "remember", remember)

        resp = client.post(
            "/remember",
            data={"datasetName": "test_dataset", "chunk_size": "42"},
            files={"data": ("test.txt", b"Cognee is an AI memory platform.", "text/plain")},
        )

        assert resp.status_code == 200
        remember.assert_awaited_once()
        assert remember.await_args.kwargs["chunk_size"] == 42

    def test_remember_passes_self_improvement(self, client, monkeypatch):
        remember_pkg = importlib.import_module("cognee.api.v1.remember")

        class RememberResultStub:
            status = "completed"

            def to_dict(self):
                return {"status": "completed", "dataset_name": "test_dataset"}

        remember = AsyncMock(return_value=RememberResultStub())
        monkeypatch.setattr(remember_pkg, "remember", remember)

        resp = client.post(
            "/remember",
            data={"datasetName": "test_dataset", "self_improvement": "false"},
            files={"data": ("test.txt", b"Cognee is an AI memory platform.", "text/plain")},
        )

        assert resp.status_code == 200
        remember.assert_awaited_once()
        assert remember.await_args.kwargs["self_improvement"] is False


# ---------------------------------------------------------------------------
# Cognify endpoint
# ---------------------------------------------------------------------------


class TestCognifyEndpoint:
    def test_cognify_missing_datasets_returns_400(self, client):
        resp = client.post("/cognify", json={})
        assert resp.status_code == 400

    def test_cognify_pipeline_errored_returns_500(self, client):
        import cognee.api.v1.cognify as cognify_pkg

        cognify_pkg.cognify = AsyncMock(return_value={"run": _make_errored()})

        resp = client.post(
            "/cognify",
            json={"datasets": ["test_dataset"], "run_in_background": False},
        )
        assert resp.status_code == 500
        body = resp.json()
        assert body["error"] == "Pipeline run errored"

    def test_cognify_success_returns_200(self, client):
        import cognee.api.v1.cognify as cognify_pkg

        completed = _make_completed()
        cognify_pkg.cognify = AsyncMock(return_value={str(MOCK_DATASET_ID): completed})

        resp = client.post(
            "/cognify",
            json={"datasets": ["test_dataset"], "run_in_background": False},
        )
        assert resp.status_code == 200

    def test_cognify_passes_chunk_size(self, client, monkeypatch):
        import cognee.api.v1.cognify as cognify_pkg

        completed = _make_completed()
        cognify = AsyncMock(return_value={str(MOCK_DATASET_ID): completed})
        monkeypatch.setattr(cognify_pkg, "cognify", cognify)

        resp = client.post(
            "/cognify",
            json={"datasets": ["test_dataset"], "chunk_size": 42},
        )

        assert resp.status_code == 200
        cognify.assert_awaited_once()
        assert cognify.await_args.kwargs["chunk_size"] == 42

    def test_cognify_internal_error_returns_500(self, client):
        import cognee.api.v1.cognify as cognify_pkg

        cognify_pkg.cognify = AsyncMock(side_effect=RuntimeError("unexpected"))

        resp = client.post(
            "/cognify",
            json={"datasets": ["test_dataset"]},
        )
        assert resp.status_code == 500
        assert resp.json()["error"] == "Internal server error"

    def test_cognify_llm_payment_required_returns_402(self, client):
        import cognee.api.v1.cognify as cognify_pkg

        cognify_pkg.cognify = AsyncMock(side_effect=LLMPaymentRequiredError())

        resp = client.post(
            "/cognify",
            json={"datasets": ["test_dataset"]},
        )
        assert resp.status_code == 402
        assert resp.json()["error"] == "Token budget exhausted"


# ---------------------------------------------------------------------------
# Search endpoint
# ---------------------------------------------------------------------------


class TestSearchEndpoint:
    def test_search_permission_denied_returns_403(self, client):
        import cognee.api.v1.search as search_pkg

        search_pkg.search = AsyncMock(side_effect=PermissionDeniedError("no access to dataset"))

        resp = client.post(
            "/search",
            json={
                "search_type": "GRAPH_COMPLETION",
                "query": "What is Cognee?",
            },
        )
        # The router re-raises CogneeApiError; the global handler answers with
        # the error's own status code and message.
        assert resp.status_code == 403
        assert "no access to dataset" in resp.json()["detail"]

    def test_search_success_returns_200(self, client):
        import cognee.api.v1.search as search_pkg

        search_pkg.search = AsyncMock(
            return_value=[
                {
                    "search_result": "Cognee is an AI memory platform.",
                    "dataset_id": str(MOCK_DATASET_ID),
                    "dataset_name": "test_dataset",
                }
            ]
        )

        resp = client.post(
            "/search",
            json={
                "search_type": "GRAPH_COMPLETION",
                "query": "What is Cognee?",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert len(body) == 1
        assert "Cognee" in body[0]["search_result"]

    def test_search_internal_error_returns_500(self, client):
        import cognee.api.v1.search as search_pkg

        search_pkg.search = AsyncMock(side_effect=RuntimeError("unexpected"))

        resp = client.post(
            "/search",
            json={"query": "test"},
        )
        assert resp.status_code == 500
        assert resp.json()["error"] == "Internal server error"

    def test_search_llm_payment_required_returns_402(self, client):
        import cognee.api.v1.search as search_pkg

        search_pkg.search = AsyncMock(side_effect=LLMPaymentRequiredError())

        resp = client.post(
            "/search",
            json={"search_type": "GRAPH_COMPLETION", "query": "test"},
        )
        assert resp.status_code == 402
        assert "budget" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Memify endpoint
# ---------------------------------------------------------------------------


class TestMemifyEndpoint:
    def test_memify_missing_dataset_returns_400(self, client):
        resp = client.post("/memify", json={})
        assert resp.status_code == 400

    def test_memify_pipeline_errored_returns_500(self, client):
        import cognee.modules.memify as memify_pkg

        memify_pkg.memify = AsyncMock(return_value=_make_errored(error="memify failed"))

        resp = client.post(
            "/memify",
            json={"dataset_name": "test_dataset", "run_in_background": False},
        )
        assert resp.status_code == 500
        body = resp.json()
        assert body["error"] == "Pipeline run errored"

    def test_memify_success_returns_200(self, client):
        import cognee.modules.memify as memify_pkg

        completed = _make_completed()
        memify_pkg.memify = AsyncMock(return_value=completed.model_dump())

        resp = client.post(
            "/memify",
            json={"dataset_name": "test_dataset", "run_in_background": False},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "completed"

    def test_memify_internal_error_returns_500(self, client):
        import cognee.modules.memify as memify_pkg

        memify_pkg.memify = AsyncMock(side_effect=RuntimeError("unexpected"))

        resp = client.post(
            "/memify",
            json={"dataset_name": "test_dataset"},
        )
        assert resp.status_code == 500
        assert resp.json()["error"] == "Internal server error"


# ---------------------------------------------------------------------------
# Update endpoint
# ---------------------------------------------------------------------------


class TestUpdateEndpoint:
    def test_update_pipeline_errored_returns_500(self, client):
        import cognee.api.v1.update as update_pkg

        update_pkg.update = AsyncMock(return_value={"run": _make_errored(error="update failed")})

        resp = client.patch(
            "/update",
            params={"data_id": str(uuid4()), "dataset_id": str(uuid4())},
            files={"data": ("x.txt", b"hello", "text/plain")},
            data={"node_set": ""},
        )
        assert resp.status_code == 500
        body = resp.json()
        assert body["error"] == "Pipeline run errored"

    def test_update_success_returns_200(self, client):
        import cognee.api.v1.update as update_pkg

        completed = _make_completed()
        update_pkg.update = AsyncMock(return_value={str(MOCK_DATASET_ID): completed})

        resp = client.patch(
            "/update",
            params={"data_id": str(uuid4()), "dataset_id": str(uuid4())},
            files={"data": ("updated.txt", b"updated content", "text/plain")},
            data={"node_set": ""},
        )
        assert resp.status_code == 200

    def test_update_internal_error_returns_500(self, client):
        import cognee.api.v1.update as update_pkg

        update_pkg.update = AsyncMock(side_effect=RuntimeError("unexpected"))

        resp = client.patch(
            "/update",
            params={"data_id": str(uuid4()), "dataset_id": str(uuid4())},
            files={"data": ("x.txt", b"hello", "text/plain")},
            data={"node_set": ""},
        )
        assert resp.status_code == 500
        assert resp.json()["error"] == "Internal server error"


# ---------------------------------------------------------------------------
# Session-dataset conflicts must reach HTTP callers with their message
# ---------------------------------------------------------------------------


class TestCogneeErrorPassthrough:
    """CogneeApiError subclasses carry their own status code and actionable
    message. The routers re-raise them so the global CogneeApiError handler
    answers — instead of replacing the message with generic advice that cannot
    fix the underlying problem (e.g. "run cognify" for a 409 conflict)."""

    def _conflict(self, message):
        return CogneeValidationError(message=message, name="ConflictError", status_code=409)

    def test_search_surfaces_cognee_error_message_and_status(self, client, monkeypatch):
        search_pkg = importlib.import_module("cognee.api.v1.search")
        error = self._conflict("session chat-1 is bound to another dataset")
        monkeypatch.setattr(search_pkg, "search", AsyncMock(side_effect=error))

        resp = client.post("/search", json={"query": "q"})

        assert resp.status_code == 409
        assert "session chat-1 is bound to another dataset" in resp.text

    def test_recall_surfaces_cognee_error_message_and_status(self, client, monkeypatch):
        recall_pkg = importlib.import_module("cognee.api.v1.recall")
        error = self._conflict("recall conflict detail the caller must see")
        monkeypatch.setattr(recall_pkg, "recall", AsyncMock(side_effect=error))

        resp = client.post("/recall", json={"query": "q"})

        assert resp.status_code == 409
        assert "recall conflict detail the caller must see" in resp.text

    def test_improve_surfaces_cognee_error_message_and_status(self, client, monkeypatch):
        improve_pkg = importlib.import_module("cognee.api.v1.improve")
        error = self._conflict("improve conflict detail the caller must see")
        monkeypatch.setattr(improve_pkg, "improve", AsyncMock(side_effect=error))

        resp = client.post("/improve", json={"datasetName": "docs"})

        assert resp.status_code == 409
        assert "improve conflict detail the caller must see" in resp.text

    def test_remember_surfaces_cognee_error_message_and_status(self, client, monkeypatch):
        remember_pkg = importlib.import_module("cognee.api.v1.remember")
        error = self._conflict("remember conflict detail the caller must see")
        monkeypatch.setattr(remember_pkg, "remember", AsyncMock(side_effect=error))

        resp = client.post(
            "/remember",
            data={"datasetName": "docs"},
            files={"data": ("x.txt", b"hello", "text/plain")},
        )

        assert resp.status_code == 409
        assert "remember conflict detail the caller must see" in resp.text


class TestRecallPermissionDenied:
    def test_recall_permission_denied_returns_403_with_message(self, client, monkeypatch):
        """Permission denials surface like every other CogneeApiError: the
        exception's own 403 and message via the global handler — not the old
        misleading "prerequisites not met, run cognify" body."""
        from cognee.modules.users.exceptions.exceptions import PermissionDeniedError

        recall_pkg = importlib.import_module("cognee.api.v1.recall")
        monkeypatch.setattr(
            recall_pkg, "recall", AsyncMock(side_effect=PermissionDeniedError("no access"))
        )

        resp = client.post("/recall", json={"query": "q"})

        assert resp.status_code == 403
        assert "no access" in resp.text
        assert "cognify" not in resp.text.lower()
