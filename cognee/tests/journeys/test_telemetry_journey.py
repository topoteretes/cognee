"""Journey: telemetry from one remember() reaches the proxy intact, in few requests.

Batching (SDK-1082) must not change what the analytics proxy receives: every
event a remember() emits arrives in the order it was emitted, with the same
payload, and the whole run costs a handful of requests instead of one per event.
"""

from __future__ import annotations

import json

import httpx
import pytest

import cognee
from cognee.shared import telemetry_sender, utils
from cognee.shared.telemetry_limits import MILESTONE_EVENTS, TelemetryLimits
from cognee.shared.telemetry_sender import TelemetrySender

DATASET = "journey_telemetry"

DOC = (
    "Title: Fenwick Tidal Mill\n\nThe Fenwick Tidal Mill was restored by millwright Beatriz Anand "
    "and grinds spelt on spring tides."
)


@pytest.mark.journey
@pytest.mark.asyncio
async def test_remember_telemetry_arrives_intact_in_few_requests(
    clean_env, default_user, monkeypatch
):
    bodies, queued = _route_telemetry_to_fake_proxy(monkeypatch)

    result = await cognee.remember(DOC, dataset_name=DATASET)
    assert result.status == "completed"
    await utils.flush_telemetry()

    assert len(queued) > 10, f"expected dozens of events, got {len(queued)}"
    posted = [event for body in bodies for event in body]
    assert posted == queued
    # Milestones and detail events use separate batches to protect the reserve.
    assert len(bodies) < len(queued)
    for body in bodies:
        assert len(body) <= telemetry_sender.BATCH_SIZE
        assert (
            len(json.dumps(body, separators=(",", ":")).encode()) <= TelemetryLimits().batch_bytes
        )
        assert len({event["event_name"] in MILESTONE_EVENTS for event in body}) == 1


def _route_telemetry_to_fake_proxy(monkeypatch) -> tuple[list, list[dict]]:
    """Turn telemetry on and swap the sender for one posting to an in-memory proxy."""
    monkeypatch.setenv("ENV", "prod")
    monkeypatch.delenv("TELEMETRY_DISABLED", raising=False)
    monkeypatch.setattr(utils, "get_anonymous_id", lambda: "anonymous-test-id")
    monkeypatch.setattr(utils, "get_persistent_id", lambda: "persistent-test-id")
    # Buffer this small journey until close; admission limits still apply.
    monkeypatch.setattr(telemetry_sender, "FLUSH_INTERVAL_SECONDS", 3600)

    bodies: list = []

    def proxy(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200)

    sender = TelemetrySender("http://proxy.test", 5.0, httpx.MockTransport(proxy))
    queued: list[dict] = []
    enqueue = sender.enqueue

    def record_and_enqueue(payload: dict) -> None:
        queued.append(json.loads(json.dumps(payload)))
        enqueue(payload)

    monkeypatch.setattr(sender, "enqueue", record_and_enqueue)
    monkeypatch.setattr(utils, "_telemetry_sender", sender)
    return bodies, queued
