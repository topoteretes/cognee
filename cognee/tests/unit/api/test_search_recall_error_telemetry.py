"""search() and recall() report their failures to telemetry (SDK-775).

Before, a failed search or recall was a Started event with no end: in the
warehouse it looked like a run that never finished. Both now carry the
``telemetry_on_error`` decorator, so a failure ends with an ERRORED event
naming the error class and the operation still raises.
"""

import importlib
from types import SimpleNamespace
from uuid import uuid4

import pytest

from cognee.exceptions import CogneeValidationError
from cognee.modules.search.types import SearchType
from cognee.shared import utils

search_module = importlib.import_module("cognee.modules.search.methods.search")
recall_module = importlib.import_module("cognee.api.v1.recall.recall")


@pytest.fixture
def events(monkeypatch):
    calls = []

    def record(event, user=None, additional_properties=None, **_):
        calls.append((event, user, additional_properties or {}))

    monkeypatch.setattr(utils, "send_telemetry", record)
    monkeypatch.setattr(search_module, "send_telemetry", record)
    return calls


@pytest.mark.asyncio
async def test_search_failure_ends_with_an_errored_event(events, monkeypatch):
    user = SimpleNamespace(id=uuid4(), tenant_id=None)

    async def failing_search(**_):
        raise PermissionError("no read grant on dataset X")

    monkeypatch.setattr(search_module, "authorized_search", failing_search)

    with pytest.raises(PermissionError):
        await search_module.search(
            query_text="who owns it", query_type=SearchType.CHUNKS, dataset_ids=None, user=user
        )

    names = [event for event, _, _ in events]
    assert names == ["cognee.search EXECUTION STARTED", "cognee.search EXECUTION ERRORED"]
    _, sent_user, properties = events[-1]
    assert sent_user is user
    assert properties["exception_type"] == "PermissionError"
    assert "dataset X" not in str(properties)


@pytest.mark.asyncio
async def test_recall_failure_ends_with_an_errored_event(events):
    with pytest.raises(CogneeValidationError):
        await recall_module.recall("who owns it", tools_trigger="bogus")

    assert [event for event, _, _ in events] == ["cognee.recall ERRORED"]
    _, sent_user, properties = events[0]
    assert sent_user == "sdk"
    assert properties["exception_type"] == "CogneeValidationError"
