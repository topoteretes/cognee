"""Every error the API returns must emit exactly one telemetry event (SDK-635).

Before this, the HTTP layer reported nothing when it failed: only the pipeline
runner emitted a failure event, so self-hosted errors were invisible to us.
These tests pin the three exits an error can take, that each fires once, that
the response is unchanged, and — most importantly — that no user content rides
along in the event.
"""

import os
from unittest.mock import patch

import pytest

with patch("dotenv.load_dotenv"):
    os.environ["ENABLE_BACKEND_ACCESS_CONTROL"] = "false"
    os.environ["REQUIRE_AUTHENTICATION"] = "false"

    from fastapi import FastAPI, Request, status
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import JSONResponse
    from fastapi.testclient import TestClient

from cognee.api.exception_telemetry import (
    API_EXCEPTION_EVENT,
    send_api_exception_telemetry,
)
from cognee.exceptions import CogneeApiError

TELEMETRY_TARGET = "cognee.api.exception_telemetry.send_telemetry"


class DatasetMissingError(CogneeApiError):
    def __init__(self):
        super().__init__(
            # Deliberately carries user content, as real messages do.
            message="Dataset 'q3-revenue-forecast' not found.",
            name="DatasetMissingError",
            status_code=status.HTTP_404_NOT_FOUND,
            log=False,
        )


def _app() -> FastAPI:
    """A miniature app wired exactly like cognee/api/client.py."""
    app = FastAPI()

    @app.middleware("http")
    async def _report_unhandled_exceptions(request, call_next):
        try:
            return await call_next(request)
        except Exception as error:
            send_api_exception_telemetry(request, error, status.HTTP_500_INTERNAL_SERVER_ERROR)
            raise

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError):
        send_api_exception_telemetry(request, exc, status.HTTP_400_BAD_REQUEST)
        return JSONResponse(status_code=400, content={"detail": "invalid"})

    @app.exception_handler(CogneeApiError)
    async def _cognee_error(request: Request, exc: CogneeApiError):
        send_api_exception_telemetry(request, exc, exc.status_code, error_name=exc.name)
        return JSONResponse(
            status_code=exc.status_code, content={"detail": f"{exc.message} [{exc.name}]"}
        )

    @app.get("/api/v1/datasets/{dataset_id}/graph")
    async def _raises_cognee_error(dataset_id: str):
        raise DatasetMissingError()

    @app.get("/api/v1/boom")
    async def _raises_unhandled():
        raise RuntimeError("upstream exploded")

    @app.get("/api/v1/typed")
    async def _needs_int(count: int):
        return {"count": count}

    return app


def _events(mock) -> list:
    return [c for c in mock.call_args_list if c.args and c.args[0] == API_EXCEPTION_EVENT]


def test_cognee_api_error_emits_one_event_and_keeps_the_response():
    with patch(TELEMETRY_TARGET) as telemetry:
        client = TestClient(_app())
        response = client.get("/api/v1/datasets/abc-123/graph")

    assert response.status_code == 404
    assert "Dataset 'q3-revenue-forecast' not found." in response.json()["detail"]

    events = _events(telemetry)
    assert len(events) == 1, "exactly one event per error"
    props = events[0].kwargs["additional_properties"]
    assert props["status_code"] == 404
    assert props["error_name"] == "DatasetMissingError"
    assert props["exception_type"] == "DatasetMissingError"


def test_event_carries_the_templated_route_not_the_resolved_one():
    """The resolved path embeds ids, which identify user data."""
    with patch(TELEMETRY_TARGET) as telemetry:
        client = TestClient(_app())
        client.get("/api/v1/datasets/abc-123/graph")

    endpoint = _events(telemetry)[0].kwargs["additional_properties"]["endpoint"]
    assert endpoint == "GET /api/v1/datasets/{dataset_id}/graph"
    assert "abc-123" not in endpoint


def test_no_user_content_reaches_the_event():
    """exc.message interpolates dataset names; none of it may be sent."""
    with patch(TELEMETRY_TARGET) as telemetry:
        client = TestClient(_app())
        client.get("/api/v1/datasets/abc-123/graph")

    payload = repr(_events(telemetry)[0].kwargs["additional_properties"])
    assert "q3-revenue-forecast" not in payload
    assert "Dataset '" not in payload


def test_unhandled_exception_emits_one_event_and_still_raises():
    with patch(TELEMETRY_TARGET) as telemetry:
        client = TestClient(_app(), raise_server_exceptions=False)
        response = client.get("/api/v1/boom")

    assert response.status_code == 500
    events = _events(telemetry)
    assert len(events) == 1
    props = events[0].kwargs["additional_properties"]
    assert props["exception_type"] == "RuntimeError"
    assert props["status_code"] == 500


def test_validation_error_emits_one_event():
    with patch(TELEMETRY_TARGET) as telemetry:
        client = TestClient(_app())
        response = client.get("/api/v1/typed", params={"count": "not-a-number"})

    assert response.status_code == 400
    events = _events(telemetry)
    assert len(events) == 1
    assert events[0].kwargs["additional_properties"]["status_code"] == 400


def test_handled_errors_are_not_double_counted_by_the_middleware():
    """ExceptionMiddleware sits inside the middleware, so handled errors arrive as responses."""
    with patch(TELEMETRY_TARGET) as telemetry:
        client = TestClient(_app())
        client.get("/api/v1/datasets/abc-123/graph")
        client.get("/api/v1/typed", params={"count": "nope"})

    assert len(_events(telemetry)) == 2, "one per error, not two"


def test_telemetry_failure_never_breaks_the_error_response():
    with patch(TELEMETRY_TARGET, side_effect=RuntimeError("telemetry down")):
        client = TestClient(_app())
        response = client.get("/api/v1/datasets/abc-123/graph")

    assert response.status_code == 404


@pytest.mark.parametrize("request_object", [None])
def test_missing_request_does_not_raise(request_object):
    with patch(TELEMETRY_TARGET) as telemetry:
        send_api_exception_telemetry(request_object, RuntimeError("x"), 500)

    assert _events(telemetry)[0].kwargs["additional_properties"]["endpoint"] == (
        "UNKNOWN unmatched"
    )


class TestRealApplicationWiring:
    """The tests above drive a replica of the wiring; these drive the real app.

    A replica can pass while ``cognee/api/client.py`` is wired wrongly, so these
    assert against the actual handler and the actual middleware stack.
    """

    @staticmethod
    def _request(method: str, template: str):
        from unittest.mock import MagicMock

        request = MagicMock()
        request.method = method
        route = MagicMock()
        route.path = template
        request.scope = {"route": route}
        return request

    def test_unhandled_exception_middleware_is_registered(self):
        from cognee.api.client import app

        registered = [
            middleware.kwargs["dispatch"].__name__
            for middleware in app.user_middleware
            if middleware.kwargs.get("dispatch")
        ]
        assert "_report_unhandled_exceptions" in registered

    @pytest.mark.asyncio
    async def test_real_cognee_api_error_handler_emits_one_clean_event(self):
        from cognee.api.client import exception_handler

        error = CogneeApiError(
            message="Dataset 'q3-revenue-forecast' not found.",
            name="DatasetMissing",
            status_code=status.HTTP_404_NOT_FOUND,
            log=False,
        )
        request = self._request("POST", "/api/v1/datasets/{dataset_id}/graph")

        with patch(TELEMETRY_TARGET) as telemetry:
            response = await exception_handler(request, error)

        assert response.status_code == 404
        events = _events(telemetry)
        assert len(events) == 1

        props = events[0].kwargs["additional_properties"]
        assert props["endpoint"] == "POST /api/v1/datasets/{dataset_id}/graph"
        assert props["status_code"] == 404
        assert props["error_name"] == "DatasetMissing"
        assert "q3-revenue-forecast" not in repr(props)

    @pytest.mark.asyncio
    async def test_real_handler_flags_an_improperly_defined_exception(self):
        """A malformed exception class is our bug, and must be distinguishable."""
        from cognee.api.client import exception_handler

        error = CogneeApiError(message="", name="", status_code=None, log=False)
        request = self._request("GET", "/api/v1/search")

        with patch(TELEMETRY_TARGET) as telemetry:
            response = await exception_handler(request, error)

        assert response.status_code == 500
        props = _events(telemetry)[0].kwargs["additional_properties"]
        assert props["improperly_defined_exception"] is True
        assert "error_name" not in props

    @pytest.mark.asyncio
    async def test_real_validation_handler_emits_one_event(self):
        from cognee.api.client import request_validation_exception_handler

        request = self._request("POST", "/api/v1/search")
        request.url.path = "/api/v1/search"
        error = RequestValidationError([])

        with patch(TELEMETRY_TARGET) as telemetry:
            response = await request_validation_exception_handler(request, error)

        assert response.status_code == 400
        events = _events(telemetry)
        assert len(events) == 1
        assert events[0].kwargs["additional_properties"]["status_code"] == 400

    @pytest.mark.asyncio
    async def test_real_middleware_reports_then_reraises(self):
        """The middleware must report the crash and change nothing about it."""
        from cognee.api.client import app

        dispatch = next(
            middleware.kwargs["dispatch"]
            for middleware in app.user_middleware
            if middleware.kwargs.get("dispatch")
            and middleware.kwargs["dispatch"].__name__ == "_report_unhandled_exceptions"
        )

        async def _explode(_request):
            raise RuntimeError("upstream exploded")

        request = self._request("GET", "/api/v1/boom")

        with (
            patch(TELEMETRY_TARGET) as telemetry,
            pytest.raises(RuntimeError, match="upstream exploded"),
        ):
            await dispatch(request, _explode)

        events = _events(telemetry)
        assert len(events) == 1
        props = events[0].kwargs["additional_properties"]
        assert props["exception_type"] == "RuntimeError"
        assert props["status_code"] == 500
