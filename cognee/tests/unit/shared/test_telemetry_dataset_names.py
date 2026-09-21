"""Dataset names never leave the process in telemetry (SDK-739, SDK-775).

`Search API Endpoint Invoked` and `cognee.recall` carry the request's dataset
names under ``datasets``; `cognee.remember` / `cognee.forget` carry one under
``dataset_name`` and `cognee.push` / `cognee.export` under ``dataset``. Names
are user-chosen and descriptive, so `send_telemetry` replaces each one with a
marked uuid5 fingerprint; dataset ids pass through.
"""

import asyncio
from typing import Any
from uuid import NAMESPACE_OID, uuid5

import pytest

from cognee.shared import utils

NAMES = ["acme-confidential", "patient_service_15f2f3f1"]
FINGERPRINTS = [
    utils.TELEMETRY_FINGERPRINT_PREFIX + str(uuid5(NAMESPACE_OID, name)) for name in NAMES
]


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

    # A string value keeps the pre-existing form: bare fingerprint, no prefix.
    assert out["datasets"] == str(uuid5(NAMESPACE_OID, NAMES[0]))
    assert out["dataset_ids"] == ["not-a-sanitized-key"]
    assert out["nested"]["datasets"] == FINGERPRINTS


def test_sanitizer_keeps_non_string_list_elements():
    out = utils._sanitize_nested_properties({"datasets": [None, 3]}, ["datasets"])

    assert out["datasets"] == [None, 3]


def test_sanitizer_passes_dataset_ids_through_and_fingerprints_names():
    """The datasets status events put dataset UUIDs under the same key; ids are not content."""
    dataset_id = "3f2c9a10-6b7d-4c1e-9a2b-8d5e4f6a7b8c"

    out = utils._sanitize_nested_properties(
        {"datasets": [dataset_id, NAMES[0]]}, utils.TELEMETRY_SANITIZED_PROPERTIES
    )

    assert out["datasets"] == [dataset_id, FINGERPRINTS[0]]
    assert FINGERPRINTS[0].startswith("fp:") and not dataset_id.startswith("fp:")


def test_sanitizer_still_fingerprints_a_uuid_shaped_string_value():
    """String values keep the existing behaviour: hashed regardless of shape."""
    session_id = "3f2c9a10-6b7d-4c1e-9a2b-8d5e4f6a7b8c"

    out = utils._sanitize_nested_properties(
        {"session_id": session_id}, utils.TELEMETRY_SANITIZED_PROPERTIES
    )

    assert out["session_id"] == str(uuid5(NAMESPACE_OID, session_id))


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


def test_sanitizer_masks_a_single_dataset_name_and_passes_an_id_through():
    """remember/forget send ``dataset_name``, push/export send ``dataset`` — the
    live 1.6.0 payloads carried both in clear. Same rule as the ``datasets`` list."""
    dataset_id = "3f2c9a10-6b7d-4c1e-9a2b-8d5e4f6a7b8c"

    out = utils._sanitize_nested_properties(
        {"dataset_name": NAMES[0], "dataset": dataset_id, "dataset_id": dataset_id},
        utils.TELEMETRY_SANITIZED_PROPERTIES,
    )

    assert out["dataset_name"] == FINGERPRINTS[0]
    assert out["dataset"] == dataset_id
    assert out["dataset_id"] == dataset_id


def test_sanitizer_keeps_an_empty_dataset_name_empty():
    """forget() sends ``""`` when no dataset was given; that is not a name."""
    out = utils._sanitize_nested_properties({"dataset_name": ""}, [])

    assert out["dataset_name"] == ""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("event_name", "key"),
    [
        ("cognee.remember", "dataset_name"),
        ("cognee.forget", "dataset_name"),
        ("cognee.push", "dataset"),
    ],
)
async def test_send_telemetry_ships_a_fingerprint_for_a_single_dataset_name(
    monkeypatch, event_name, key
):
    payloads: list[dict[str, Any]] = []

    async def capture(payload: dict[str, Any]) -> None:
        payloads.append(payload)

    monkeypatch.setenv("ENV", "prod")
    monkeypatch.delenv("TELEMETRY_DISABLED", raising=False)
    monkeypatch.setattr(utils, "_send_telemetry_request", capture)
    monkeypatch.setattr(utils, "get_anonymous_id", lambda: "anon")
    monkeypatch.setattr(utils, "get_persistent_id", lambda: "persistent")

    utils.send_telemetry(event_name, None, additional_properties={key: NAMES[0]})
    await asyncio.sleep(0)

    assert payloads[0]["properties"][key] == FINGERPRINTS[0]
    assert NAMES[0] not in repr(payloads[0])
