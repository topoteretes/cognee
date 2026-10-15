"""Operation failures, surfaces and model paths in telemetry (SDK-775).

``telemetry_on_error`` gives search and recall the terminal event their
Started/Completed pair lacked; the operation context tells each
event which surface initiated it; ``telemetry_model_label`` keeps a model
setting that is a filesystem path (and so an account name) out of the payload.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from cognee.modules.operations.origin import operation_origin_scope
from cognee.shared import utils


@pytest.fixture
def sent(monkeypatch):
    calls = []
    monkeypatch.setattr(utils, "send_telemetry", lambda *a, **k: calls.append((a, k)))
    return calls


@pytest.mark.asyncio
async def test_failure_emits_the_event_with_the_class_and_re_raises(sent):
    user = SimpleNamespace(id=uuid4(), tenant_id=None)

    @utils.telemetry_on_error("cognee.search EXECUTION ERRORED")
    async def search(query_text, user):
        raise PermissionError("dataset X is private")

    with pytest.raises(PermissionError):
        await search("q", user=user)

    ((event, sent_user), kwargs) = sent[0]
    assert event == "cognee.search EXECUTION ERRORED"
    assert sent_user is user
    assert kwargs["additional_properties"]["exception_type"] == "PermissionError"
    assert "dataset X" not in str(kwargs)  # the message never travels


@pytest.mark.asyncio
async def test_failure_carries_the_cause_and_status_under_the_wrapper(sent):
    class ProviderError(Exception):
        status_code = 503

    @utils.telemetry_on_error("cognee.recall ERRORED")
    async def recall(query_text, user=None):
        try:
            raise ProviderError("upstream down")
        except ProviderError as error:
            raise RuntimeError("recall failed") from error

    with pytest.raises(RuntimeError):
        await recall("q")

    properties = sent[0][1]["additional_properties"]
    assert properties["exception_type"] == "RuntimeError"
    assert properties["exception_cause"] == "ProviderError"
    assert properties["status_code"] == 503
    assert "upstream down" not in str(sent)


@pytest.mark.asyncio
async def test_positional_user_and_missing_user_both_resolve(sent):
    user = SimpleNamespace(id=uuid4(), tenant_id=None)

    @utils.telemetry_on_error("op ERRORED")
    async def op(query_text, user=None):
        raise ValueError()

    with pytest.raises(ValueError):
        await op("q", user)
    with pytest.raises(ValueError):
        await op("q")

    assert sent[0][0][1] is user
    assert sent[1][0][1] == "sdk"


@pytest.mark.asyncio
async def test_cancellation_is_a_terminal_event_too(sent):
    @utils.telemetry_on_error("op ERRORED")
    async def op():
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await op()

    assert sent[0][1]["additional_properties"]["exception_type"] == "CancelledError"


@pytest.mark.asyncio
async def test_success_emits_nothing_and_returns_the_value(sent):
    @utils.telemetry_on_error("op ERRORED")
    async def op():
        return 42

    assert await op() == 42
    assert sent == []


def test_wrapped_function_keeps_its_name_and_signature():
    async def recall(query_text: str, *, top_k: int = 5):
        return query_text

    wrapped = utils.telemetry_on_error("x")(recall)
    assert wrapped.__name__ == "recall"
    assert wrapped.__wrapped__ is recall


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("openai/gpt-5-mini", "openai/gpt-5-mini"),
        ("ollama/phi4:latest", "ollama/phi4:latest"),
        ("gpt-4o", "gpt-4o"),
        (None, None),
        ("/Users/alice/models/x.gguf", utils.TELEMETRY_LOCAL_PATH_LABEL),
        ("/home/alice/x.gguf", utils.TELEMETRY_LOCAL_PATH_LABEL),
        ("C:\\models\\x.gguf", utils.TELEMETRY_LOCAL_PATH_LABEL),
        ("~/models/x.gguf", utils.TELEMETRY_LOCAL_PATH_LABEL),
        ("./models/x.gguf", utils.TELEMETRY_LOCAL_PATH_LABEL),
        ("/models/x.gguf", utils.TELEMETRY_LOCAL_PATH_LABEL),
        ("\\\\server\\share\\x", utils.TELEMETRY_LOCAL_PATH_LABEL),
    ],
)
def test_model_label_keeps_provider_names_and_hides_paths(model, expected):
    assert utils.telemetry_model_label(model) == expected


def test_origin_uses_the_operation_context_and_the_environment_wins(monkeypatch):
    monkeypatch.delenv(utils.TELEMETRY_ORIGIN_ENV, raising=False)
    assert utils.telemetry_origin() == "sdk"
    with operation_origin_scope("cli"):
        assert utils.telemetry_origin() == "cli"
        assert utils.TELEMETRY_ORIGIN_ENV not in utils.os.environ
        with operation_origin_scope("background"):
            assert utils.telemetry_origin() == "background"
        monkeypatch.setenv(utils.TELEMETRY_ORIGIN_ENV, "cloud")
        assert utils.telemetry_origin() == "cloud"


@pytest.mark.asyncio
async def test_concurrent_origins_do_not_label_each_other(monkeypatch):
    monkeypatch.delenv(utils.TELEMETRY_ORIGIN_ENV, raising=False)

    async def operation(origin):
        with operation_origin_scope(origin):
            await asyncio.sleep(0)
            return utils.telemetry_origin()

    assert await asyncio.gather(operation("api"), operation("sdk")) == ["api", "sdk"]
    assert utils.telemetry_origin() == "sdk"


@pytest.mark.parametrize("override", [None, "cloud"])
def test_threaded_origins_are_isolated_and_restored_after_failure(monkeypatch, override):
    if override is None:
        monkeypatch.delenv(utils.TELEMETRY_ORIGIN_ENV, raising=False)
    else:
        monkeypatch.setenv(utils.TELEMETRY_ORIGIN_ENV, override)
    origins = ["api", "cli", "mcp"]
    barrier = Barrier(len(origins), timeout=5)

    def operation(origin):
        previous = utils.telemetry_origin()
        with pytest.raises(ValueError, match="operation failed"), operation_origin_scope(origin):
            barrier.wait()
            observed = utils.telemetry_origin()
            # Keep all scopes active until every thread has read its origin.
            barrier.wait()
            raise ValueError("operation failed")
        assert utils.telemetry_origin() == previous
        return observed

    with operation_origin_scope("background"):
        with ThreadPoolExecutor(max_workers=len(origins)) as executor:
            assert list(executor.map(operation, origins)) == [override or o for o in origins]
        assert utils.telemetry_origin() == (override or "background")
    assert utils.os.getenv(utils.TELEMETRY_ORIGIN_ENV) == override


@pytest.mark.asyncio
async def test_payload_carries_the_process_origin(monkeypatch):
    monkeypatch.delenv("TELEMETRY_DISABLED", raising=False)
    monkeypatch.setenv("ENV", "local")
    monkeypatch.setenv(utils.TELEMETRY_ORIGIN_ENV, "mcp")
    request = AsyncMock()
    monkeypatch.setattr(utils, "_send_telemetry_request", request)

    # Only this call's task: another test's unmocked send, left pending on a
    # loop that is now closed, must not fail this one.
    pending_before = set(utils._TELEMETRY_TASKS)
    utils.send_telemetry("cognee.recall", "sdk", additional_properties={"top_k": 3})
    await asyncio.gather(*(utils._TELEMETRY_TASKS - pending_before))

    payload = request.await_args.args[0]
    assert payload["event_name"] == "cognee.recall"
    assert payload["properties"]["telemetry_origin"] == "mcp"
    assert payload["properties"]["top_k"] == 3


def test_api_request_origin_does_not_label_sdk_calls_alongside_the_server(monkeypatch):
    monkeypatch.delenv(utils.TELEMETRY_ORIGIN_ENV, raising=False)
    monkeypatch.setenv("ENABLE_BACKEND_ACCESS_CONTROL", "false")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from cognee.api.client import _stamp_operation_origin

    app = FastAPI()
    app.middleware("http")(_stamp_operation_origin)

    @app.get("/origin")
    async def origin():
        return {"origin": utils.telemetry_origin()}

    with TestClient(app) as client:
        assert client.get("/origin").json() == {"origin": "api"}
        assert utils.telemetry_origin() == "sdk"
    assert utils.telemetry_origin() == "sdk"
