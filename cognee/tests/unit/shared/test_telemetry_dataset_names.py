"""Dataset names never leave the process in telemetry (SDK-739).

`Search API Endpoint Invoked` and `cognee.recall` carry the request's dataset
names. Names are user-chosen and descriptive, so `send_telemetry` replaces each
one with a uuid5 fingerprint, the same treatment `session_id` already gets.
"""

import asyncio
from typing import Any
from uuid import NAMESPACE_OID, uuid5

import pytest

from cognee.shared import utils

NAMES = ["acme-confidential", "patient_service_15f2f3f1"]
FINGERPRINTS = [str(uuid5(NAMESPACE_OID, name)) for name in NAMES]


def test_sanitizer_fingerprints_each_name_in_a_list():
    out = utils._sanitize_nested_properties(
        {"datasets": NAMES, "top_k": 5}, utils.TELEMETRY_SANITIZED_PROPERTIES
    )

    assert out["datasets"] == FINGERPRINTS
    assert out["top_k"] == 5


def test_sanitizer_fingerprints_a_single_name_and_leaves_other_lists_alone():
    out = utils._sanitize_nested_properties(
        {
            "datasets": NAMES[0],
            "dataset_ids": ["not-a-sanitized-key"],
            "nested": {"datasets": NAMES},
        },
        utils.TELEMETRY_SANITIZED_PROPERTIES,
    )

    assert out["datasets"] == FINGERPRINTS[0]
    assert out["dataset_ids"] == ["not-a-sanitized-key"]
    assert out["nested"]["datasets"] == FINGERPRINTS


def test_sanitizer_keeps_non_string_list_elements():
    out = utils._sanitize_nested_properties({"datasets": [None, 3]}, ["datasets"])

    assert out["datasets"] == [None, 3]


@pytest.mark.asyncio
@pytest.mark.parametrize("event_name", ["Search API Endpoint Invoked", "cognee.recall"])
async def test_send_telemetry_ships_fingerprints_not_names(monkeypatch, event_name):
    payloads: list[dict[str, Any]] = []

    async def capture(payload: dict[str, Any]) -> None:
        payloads.append(payload)

    monkeypatch.setenv("ENV", "prod")
    monkeypatch.delenv("TELEMETRY_DISABLED", raising=False)
    monkeypatch.setattr(utils, "_send_telemetry_request", capture)
    monkeypatch.setattr(utils, "get_anonymous_id", lambda: "anon")
    monkeypatch.setattr(utils, "get_persistent_id", lambda: "persistent")

    utils.send_telemetry(event_name, None, additional_properties={"datasets": NAMES})
    await asyncio.sleep(0)

    assert len(payloads) == 1
    assert payloads[0]["properties"]["datasets"] == FINGERPRINTS
    for name in NAMES:
        assert name not in repr(payloads[0])
