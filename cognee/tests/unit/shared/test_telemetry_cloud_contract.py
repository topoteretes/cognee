"""The cloud telemetry contract, checked on the payload send_telemetry queues (SDK-1167)."""

from typing import Any

import pytest

from cognee.modules.pipelines.operations.run_tasks_with_telemetry import (
    pipeline_run_telemetry_properties,
)
from cognee.shared import utils
from cognee.shared.telemetry_schema import CLOUD_REJECTIONS
from cognee.shared.utils import send_telemetry, telemetry_exception_properties


@pytest.fixture
def cloud_payloads(monkeypatch) -> list[dict[str, Any]]:
    """What send_telemetry queues under TELEMETRY_ORIGIN=cloud, starting from no rejections."""
    payloads: list[dict[str, Any]] = []
    monkeypatch.setenv("ENV", "prod")
    monkeypatch.setenv("TELEMETRY_ORIGIN", "cloud")
    monkeypatch.delenv("TELEMETRY_DISABLED", raising=False)
    monkeypatch.setattr(utils._telemetry_sender, "enqueue", payloads.append)
    CLOUD_REJECTIONS.clear()
    return payloads


def test_envelope_carries_only_trusted_identity(cloud_payloads, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "provider-prefix-secret")
    send_telemetry("cognee.search EXECUTION COMPLETED", "sdk")

    [payload] = cloud_payloads
    assert payload["anonymous_id"] == ""  # the proxy requires the key
    assert payload["user_properties"] == {"user_id": "sdk", "tenant_id": "Single User Tenant"}
    assert set(payload["properties"]) == set(utils._cloud_trusted_fields("sdk", None))


def test_event_properties_cannot_override_trusted_fields(cloud_payloads):
    trusted = utils._cloud_trusted_fields("sdk", None)
    forged = {name: "forged" for name in trusted}
    send_telemetry("cognee.search EXECUTION COMPLETED", "sdk", forged)

    [payload] = cloud_payloads
    assert payload["properties"] == trusted


def test_unknown_events_and_fields_are_removed_and_counted_without_values(cloud_payloads):
    send_telemetry("Task Started", "sdk", {"task": "secret-task"})
    send_telemetry(
        "Pipeline Run Started",
        "sdk",
        {"graph": {"provider": "ladybug", "url": "secret-url"}, "dataset_name": "secret-name"},
    )

    [payload] = cloud_payloads
    assert payload["event_name"] == "Pipeline Run Started"
    assert payload["properties"]["graph"] == {"provider": "ladybug"}
    assert "dataset_name" not in payload["properties"]
    assert set(CLOUD_REJECTIONS) == {
        ("Task Started", "unknown_event"),
        ("Pipeline Run Started", "unknown_field:graph.url"),
        ("Pipeline Run Started", "unknown_field:dataset_name"),
    }
    assert "secret" not in str(CLOUD_REJECTIONS)


_UUID = "00000000-0000-4000-8000-000000000000"


@pytest.mark.parametrize(
    ("event_name", "path", "bad_value"),
    [
        ("cognee.recall", "top_k", "15"),
        ("cognee.recall", "top_k", True),
        ("cognee.recall", "search_type", "what did we decide?"),
        ("cognee.recall", "dataset_ids", ",".join([_UUID] * 101)),
        ("Pipeline Run Errored", "exception_chain", ["RateLimitError"] * 5),
        ("Pipeline Run Errored", "exception_type", "Error: key sk-123 rejected"),
        ("Pipeline Run Errored", "status_code", 42),
        ("cognee.export", "format", "/home/someone/export"),
        ("Pipeline Run Started", "llm.model", "m" * 129),
    ],
)
def test_a_bad_value_removes_only_its_field(cloud_payloads, event_name, path, bad_value):
    *parents, name = path.split(".")
    properties = sent = {}
    for parent in parents:
        sent[parent] = sent = {}
    sent[name] = bad_value
    send_telemetry(event_name, "sdk", properties)

    [payload] = cloud_payloads
    received = payload["properties"]
    for parent in parents:
        received = received.get(parent, {})
    assert name not in received
    assert set(CLOUD_REJECTIONS) == {(event_name, f"invalid_value:{path}")}


def test_real_pipeline_error_properties_lose_only_database_urls(cloud_payloads):
    """Built by the emitters' own helpers, so a field they add without the contract fails here."""
    properties = {
        **pipeline_run_telemetry_properties(_UUID, _UUID, None, graph_extractor="llm"),
        "pipeline_event_scope": "run",
        **telemetry_exception_properties(RuntimeError("secret message")),
    }
    send_telemetry("Pipeline Run Errored", "sdk", properties)

    [payload] = cloud_payloads
    assert payload["properties"]["exception_type"] == "RuntimeError"
    assert "secret message" not in str(payload)
    assert set(CLOUD_REJECTIONS) == {
        ("Pipeline Run Errored", f"unknown_field:{database}.url")
        for database in ("graph", "vector", "relational")
    }


def test_a_recovered_run_keeps_its_unknown_version(cloud_payloads):
    """Recovery cannot know which version the dead run was on; the current one is not it."""
    properties = pipeline_run_telemetry_properties(_UUID, _UUID, None, recovered=True)
    send_telemetry("Pipeline Run Errored", "sdk", {**properties, "recovered_at_startup": True})

    [payload] = cloud_payloads
    assert payload["properties"]["cognee_version"] == "unknown"


def test_a_bad_trusted_field_drops_the_event(cloud_payloads):
    send_telemetry("cognee.recall", "someone@example.com")

    assert cloud_payloads == []
    assert set(CLOUD_REJECTIONS) == {("cognee.recall", "invalid_trusted_fields")}


def test_a_failing_contract_never_reaches_the_caller(cloud_payloads, monkeypatch):
    def broken(*_args):
        raise RuntimeError("contract bug")

    monkeypatch.setattr("cognee.shared.telemetry_schema.apply_cloud_contract", broken)

    assert send_telemetry("cognee.recall", "sdk", {"top_k": 5}) is None
    assert cloud_payloads == []


def test_disabled_telemetry_sends_nothing_in_the_cloud(cloud_payloads, monkeypatch):
    monkeypatch.setenv("TELEMETRY_DISABLED", "1")
    send_telemetry("cognee.recall", "sdk", {"top_k": 5})

    assert cloud_payloads == []


def test_other_origins_keep_the_full_payload(cloud_payloads, monkeypatch):
    monkeypatch.setenv("TELEMETRY_ORIGIN", "sdk")
    monkeypatch.setenv("LLM_API_KEY", "provider-prefix-secret")
    send_telemetry("cognee.recall", "sdk", {"datasets": ["my_dataset"], "top_k": 5})

    [payload] = cloud_payloads
    assert payload["anonymous_id"] != ""
    assert {"persistent_id", "api_key_hash"} <= set(payload["properties"])
    assert payload["properties"]["datasets"][0].startswith("fp:")
    assert CLOUD_REJECTIONS == {}
