import asyncio
import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from cognee.modules.search.types import SearchType

search_module = importlib.import_module("cognee.modules.search.methods.search")


@pytest.fixture
def search_boundaries(monkeypatch):
    telemetry = Mock()
    retrieval = AsyncMock(return_value=[])
    history = AsyncMock()
    monkeypatch.setattr(search_module, "send_telemetry", telemetry)
    monkeypatch.setattr(search_module, "authorized_search", retrieval)
    monkeypatch.setattr(search_module, "log_search_history", history)
    return telemetry, retrieval, history


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [RuntimeError, ValueError, TimeoutError])
async def test_failed_search_emits_error_without_private_content(search_boundaries, error_type):
    telemetry, retrieval, history = search_boundaries
    user = SimpleNamespace(id=uuid4(), tenant_id=uuid4())
    failure = error_type("private exception detail")
    retrieval.side_effect = failure

    with pytest.raises(error_type) as raised:
        await search_module.search("private query", SearchType.CHUNKS, [uuid4()], user)

    assert raised.value is failure
    assert [call.args[0] for call in telemetry.call_args_list] == [
        "cognee.search EXECUTION STARTED",
        "cognee.search EXECUTION ERRORED",
    ]
    event = telemetry.call_args_list[-1]
    assert event.args[1] is user
    assert event.kwargs["additional_properties"] == {
        "cognee_version": search_module.cognee_version,
        "tenant_id": str(user.tenant_id),
        "error_type": error_type.__name__,
    }
    history.assert_not_awaited()


@pytest.mark.asyncio
async def test_successful_search_emits_completion(search_boundaries):
    telemetry, _, history = search_boundaries
    user = SimpleNamespace(id=uuid4(), tenant_id=None)

    assert await search_module.search("query", SearchType.CHUNKS, None, user) == []

    assert [call.args[0] for call in telemetry.call_args_list] == [
        "cognee.search EXECUTION STARTED",
        "cognee.search EXECUTION COMPLETED",
    ]
    history.assert_awaited_once()


@pytest.mark.asyncio
async def test_cancelled_search_is_not_reported_as_an_error(search_boundaries):
    telemetry, retrieval, history = search_boundaries
    retrieval.side_effect = asyncio.CancelledError
    user = SimpleNamespace(id=uuid4(), tenant_id=None)

    with pytest.raises(asyncio.CancelledError):
        await search_module.search("query", SearchType.CHUNKS, None, user)

    assert [call.args[0] for call in telemetry.call_args_list] == [
        "cognee.search EXECUTION STARTED"
    ]
    history.assert_not_awaited()
