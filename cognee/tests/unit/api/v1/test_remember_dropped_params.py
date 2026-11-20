"""remember() sends node_set, labels and external_metadata to the remote instance (cognee #6706).

Before this, ``remember(data, node_set=[...], labels=[...], external_metadata=[...])``
under ``cognee.serve()`` returned success while the server never saw those
parameters: CloudClient.remember built its form from a fixed field list that
did not include them. Same bug class as the graph_model drop fixed under
SDK-1387 (test_remote_graph_model.py) — a param the caller provides is
silently dropped by the client's fixed field list.

The wire contract is the server's own: node_set travels as repeated form
fields (one per tag, as update() already sends it); labels and
external_metadata travel as one JSON-encoded string each, which the
server's parse_labels / parse_external_metadata read back.
"""

import importlib
import json

import pytest

from cognee.api.v1.serve import state as state_mod
from cognee.api.v1.serve.cloud_client import CloudClient
from cognee.tasks.ingestion.data_item import (
    parse_external_metadata,
    parse_labels,
)

remember_mod = importlib.import_module("cognee.api.v1.remember.remember")


class _FakeResponse:
    status = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def json(self):
        return {"status": "completed", "dataset_name": "main_dataset"}

    async def text(self):
        return ""


class _FakeSession:
    """Captures what the client POSTs: the form of a remember."""

    def __init__(self):
        self.calls = []

    def post(self, url, data=None, json=None, **kwargs):
        self.calls.append({"url": url, "form": data, "json": json})
        return _FakeResponse()

    async def close(self):
        pass


@pytest.fixture
def remote(monkeypatch):
    client = CloudClient("http://remote.invalid", "key")
    session = _FakeSession()

    async def get_session():
        return session

    monkeypatch.setattr(client, "_get_session", get_session)
    monkeypatch.setattr(state_mod, "_remote_client", client)
    return session


def _form_values(form, name):
    """All values for a form field name, in order (repeated fields included)."""
    return [value for options, _headers, value in form._fields if options["name"] == name]


def _form_field(form, name):
    """The single value for a non-repeated form field, or None if absent."""
    values = _form_values(form, name)
    return values[0] if values else None


# ----- node_set -----


@pytest.mark.asyncio
async def test_remember_sends_node_set_as_repeated_fields(remote):
    await remember_mod.remember("Python was created by Guido.", node_set=["alpha", "beta"])

    (call,) = remote.calls
    assert call["url"] == "http://remote.invalid/api/v1/remember"
    assert _form_values(call["form"], "node_set") == ["alpha", "beta"]


# ----- labels -----


@pytest.mark.asyncio
async def test_remember_sends_labels_the_route_reads_back(remote):
    await remember_mod.remember("Python was created by Guido.", labels=["finance", "people"])

    (call,) = remote.calls
    assert call["url"] == "http://remote.invalid/api/v1/remember"
    # The route's own parser reads the field back: the client must send
    # something parse_labels accepts (a JSON array string).
    assert parse_labels(_form_field(call["form"], "labels")) == ["finance", "people"]


# ----- external_metadata -----


@pytest.mark.asyncio
async def test_remember_sends_external_metadata_the_route_reads_back(remote):
    await remember_mod.remember(
        "Python was created by Guido.",
        external_metadata=[{"source": "crm", "ticket": 42}, None],
    )

    (call,) = remote.calls
    assert call["url"] == "http://remote.invalid/api/v1/remember"
    # The route requires valid JSON (an array of objects/nulls); anything
    # else raises InvalidExternalMetadataError server-side.
    assert parse_external_metadata(_form_field(call["form"], "external_metadata")) == [
        {"source": "crm", "ticket": 42},
        None,
    ]


# ----- defaults: nothing sent when not given -----


@pytest.mark.asyncio
async def test_remember_sends_no_dropped_fields_by_default(remote):
    await remember_mod.remember("Python was created by Guido.")

    (call,) = remote.calls
    names = {options["name"] for options, _headers, _value in call["form"]._fields}
    assert "node_set" not in names
    assert "labels" not in names
    assert "external_metadata" not in names
