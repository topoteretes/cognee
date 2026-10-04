"""Operation failures, surfaces and model paths in telemetry (SDK-775).

``telemetry_on_error`` gives search and recall the terminal event their
Started/Completed pair lacked; ``set_default_telemetry_origin`` lets each
entrypoint say which surface it is; ``telemetry_model_label`` keeps a model
setting that is a filesystem path (and so an account name) out of the payload.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

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


def test_default_origin_is_process_local_and_the_environment_wins(monkeypatch):
    monkeypatch.delenv(utils.TELEMETRY_ORIGIN_ENV, raising=False)
    monkeypatch.setattr(utils, "_default_telemetry_origin", utils.TELEMETRY_ORIGIN_SDK)
    assert utils.telemetry_origin() == "sdk"

    utils.set_default_telemetry_origin(utils.TELEMETRY_ORIGIN_CLI)
    assert utils.telemetry_origin() == "cli"
    # never written to the environment: a child process must label itself
    assert utils.TELEMETRY_ORIGIN_ENV not in utils.os.environ

    monkeypatch.setenv(utils.TELEMETRY_ORIGIN_ENV, "cloud")
    utils.set_default_telemetry_origin(utils.TELEMETRY_ORIGIN_API)
    assert utils.telemetry_origin() == "cloud"


@pytest.mark.asyncio
async def test_payload_carries_the_process_origin(monkeypatch):
    monkeypatch.delenv("TELEMETRY_DISABLED", raising=False)
    monkeypatch.setenv("ENV", "local")
    monkeypatch.setenv(utils.TELEMETRY_ORIGIN_ENV, "mcp")
    request = AsyncMock()
    monkeypatch.setattr(utils, "_send_telemetry_request", request)

    utils.send_telemetry("cognee.recall", "sdk", additional_properties={"top_k": 3})
    await asyncio.gather(*list(utils._TELEMETRY_TASKS))

    payload = request.await_args.args[0]
    assert payload["event_name"] == "cognee.recall"
    assert payload["properties"]["telemetry_origin"] == "mcp"
    assert payload["properties"]["top_k"] == 3
