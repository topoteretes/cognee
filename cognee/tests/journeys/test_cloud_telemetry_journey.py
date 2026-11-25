"""Journey: the cloud telemetry contract removes only what it means to (SDK-1167).

With ``TELEMETRY_ORIGIN=cloud`` a property no event model declares is removed, and
an event with no model is not sent. Both happen quietly, by design, so a property a
call site starts sending, or a new event, would never reach the warehouse and no
other test would fail. This journey runs the memory API end to end and pins what
reaches the proxy and every removal the contract made on the way.

The non-cloud twin runs the same flow and checks that the SDK payload still carries
everything the cloud profile removes.
"""

from __future__ import annotations

import pytest

import cognee
from cognee.shared import telemetry_schema, utils
from cognee.tests.journeys.test_telemetry_journey import _route_telemetry_to_fake_proxy

DATASET = "journey_cloud_telemetry"
DOC = "Marie Curie discovered polonium in 1898 in Paris."
QUESTION = "Who discovered polonium?"

# What the cloud proxy receives for the flow in _run_memory_flow.
CLOUD_EVENTS = {
    "cognee.remember",
    "cognee.recall",
    "cognee.search EXECUTION STARTED",
    "cognee.search EXECUTION COMPLETED",
    "cognee.improve",
    "cognee.forget",
    "Pipeline Run Started",
    "Pipeline Run Completed",
}

# Events with no cloud model: sent by the SDK, never by the cloud.
EVENTS_WITHOUT_CLOUD_MODEL = {
    "Pipeline Item Started",
    "Pipeline Item Completed",
    "Coroutine Task Started",
    "Coroutine Task Completed",
    "Async Generator Task Started",
    "Async Generator Task Completed",
    "cognee.session.add_qa",
}

# Every property the contract removes on this flow. Each one is listed under
# "Removed from cloud events" in docs/cloud-telemetry-contract.md.
REMOVED_FIELDS = {
    ("cognee.remember", "dataset_name"),
    ("cognee.remember", "session_id"),
    ("cognee.remember", "session_ids"),
    ("cognee.recall", "datasets"),
    ("cognee.recall", "session_id"),
    ("cognee.improve", "dataset"),
    ("cognee.improve", "session_ids"),
    ("cognee.forget", "dataset_name"),
    ("Pipeline Run Started", "graph.url"),
    ("Pipeline Run Started", "vector.url"),
    ("Pipeline Run Started", "relational.url"),
    ("Pipeline Run Completed", "graph.url"),
    ("Pipeline Run Completed", "vector.url"),
    ("Pipeline Run Completed", "relational.url"),
}


@pytest.mark.journey
@pytest.mark.asyncio
async def test_cloud_telemetry_removes_only_documented_fields(clean_env, default_user, monkeypatch):
    monkeypatch.setenv("TELEMETRY_ORIGIN", "cloud")
    events = await _run_memory_flow(monkeypatch)

    assert {event["event_name"] for event in events} == CLOUD_EVENTS
    for event in events:
        assert event["properties"]["telemetry_origin"] == "cloud"
    # Counted is not enough: the removed fields must be absent from what was sent.
    for name, field in REMOVED_FIELDS:
        sent = [event["properties"] for event in events if event["event_name"] == name]
        assert not any(_has_path(properties, field) for properties in sent), (name, field)

    # Nothing else was removed: a new property or event fails here until it is
    # given a model field or added to the lists above (and to the contract doc).
    expected = {(name, f"unknown_field:{field}") for name, field in REMOVED_FIELDS} | {
        (name, "unknown_event") for name in EVENTS_WITHOUT_CLOUD_MODEL
    }
    assert set(telemetry_schema.CLOUD_REJECTIONS) == expected


@pytest.mark.journey
@pytest.mark.asyncio
async def test_sdk_telemetry_keeps_what_the_cloud_removes(clean_env, default_user, monkeypatch):
    monkeypatch.delenv("TELEMETRY_ORIGIN", raising=False)
    events = await _run_memory_flow(monkeypatch)

    assert {event["event_name"] for event in events} == CLOUD_EVENTS | EVENTS_WITHOUT_CLOUD_MODEL
    assert not telemetry_schema.CLOUD_REJECTIONS
    for name, field in REMOVED_FIELDS:
        sent = [event["properties"] for event in events if event["event_name"] == name]
        assert any(_has_path(properties, field) for properties in sent), (name, field)


async def _run_memory_flow(monkeypatch) -> list[dict]:
    """remember → recall → search → improve → forget; return what the proxy received."""
    bodies, _ = _route_telemetry_to_fake_proxy(monkeypatch)
    telemetry_schema.CLOUD_REJECTIONS.clear()

    remembered = await cognee.remember(DOC, dataset_name=DATASET, self_improvement=False)
    assert remembered.status == "completed"
    await cognee.recall(QUESTION, datasets=[DATASET])
    await cognee.search(QUESTION, datasets=[DATASET])
    await cognee.improve(dataset=DATASET)
    await cognee.forget(dataset=DATASET)
    await utils.flush_telemetry()

    return [event for body in bodies for event in body]


def _has_path(properties: dict, dotted: str) -> bool:
    """Whether ``properties`` holds a key at ``dotted`` (``graph.url`` is nested)."""
    node = properties
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            return False
        node = node[key]
    return True
