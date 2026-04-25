from importlib import import_module
from uuid import uuid4
from unittest.mock import AsyncMock, patch

import pytest

from cognee.infrastructure.engine import DataPoint
from cognee.modules.graph.methods.sanitize_relational_payload import sanitize_relational_payload
from cognee.modules.graph.methods.upsert_edges import upsert_edges
from cognee.modules.graph.methods.upsert_nodes import upsert_nodes

upsert_nodes_module = import_module("cognee.modules.graph.methods.upsert_nodes")
upsert_edges_module = import_module("cognee.modules.graph.methods.upsert_edges")


class RelationalPoint(DataPoint):
    name: str
    text: str
    details: dict
    metadata: dict = {"index_fields": ["text"]}


class FakeInsertStatement:
    def __init__(self):
        self.values_arg = None
        self.index_elements = None

    def values(self, value):
        self.values_arg = value
        return self

    def on_conflict_do_nothing(self, index_elements):
        self.index_elements = index_elements
        return self


def test_sanitize_relational_payload_strips_null_bytes_recursively():
    payload = {
        "ti\x00tle": "hel\x00lo",
        "nested": ["wo\x00rld", {"ke\x00y": "va\x00lue"}],
        "tupled": ("a\x00", 1),
    }

    assert sanitize_relational_payload(payload) == {
        "title": "hello",
        "nested": ["world", {"key": "value"}],
        "tupled": ("a", 1),
    }


def test_sanitize_relational_payload_decodes_bytes_and_bytearray():
    payload = {
        "bytes": b"hel\x00lo",
        "bytearray": bytearray(b"wo\x00rld"),
        "invalid": b"\xff\x00",
    }

    assert sanitize_relational_payload(payload) == {
        "bytes": "hello",
        "bytearray": "world",
        "invalid": "\ufffd",
    }


@pytest.mark.asyncio
async def test_upsert_nodes_sanitizes_strings_before_insert():
    point = RelationalPoint(
        name="Doc\x00ument",
        text="Bad\x00 text",
        details={"summa\x00ry": "Nested\x00 value", "items": ["A\x00", "B"]},
    )
    session = AsyncMock()
    statement = FakeInsertStatement()

    with patch.object(upsert_nodes_module, "insert", return_value=statement):
        await upsert_nodes(
            [point],
            tenant_id=uuid4(),
            user_id=uuid4(),
            dataset_id=uuid4(),
            data_id=uuid4(),
            session=session,
        )

    payload = statement.values_arg[0]
    assert payload["type"] == "RelationalPoint"
    assert payload["indexed_fields"] == ["text"]
    assert payload["label"] == "Document"
    assert payload["attributes"]["name"] == "Document"
    assert payload["attributes"]["text"] == "Bad text"
    assert payload["attributes"]["details"] == {"summary": "Nested value", "items": ["A", "B"]}
    session.execute.assert_awaited_once_with(statement)
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_upsert_edges_sanitizes_strings_before_insert():
    source_id = uuid4()
    target_id = uuid4()
    session = AsyncMock()
    statement = FakeInsertStatement()
    edge = (
        source_id,
        target_id,
        "rel\x00ates",
        {
            "edge_text": "des\x00cribes",
            "no\x00te": "nul\x00 byte",
            "nested": {"va\x00lue": "still\x00 here"},
        },
    )

    with patch.object(upsert_edges_module, "insert", return_value=statement):
        await upsert_edges(
            [edge],
            tenant_id=uuid4(),
            user_id=uuid4(),
            dataset_id=uuid4(),
            data_id=uuid4(),
            session=session,
        )

    payload = statement.values_arg[0]
    assert payload["relationship_name"] == "relates"
    assert payload["label"] == "relates"
    assert payload["attributes"] == {
        "edge_text": "describes",
        "note": "nul byte",
        "nested": {"value": "still here"},
    }
    session.execute.assert_awaited_once_with(statement)
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_upsert_edges_sanitizes_contains_edge_text_before_insert():
    source_id = uuid4()
    target_id = uuid4()
    session = AsyncMock()
    statement = FakeInsertStatement()
    edge = (
        source_id,
        target_id,
        "contains",
        {
            "edge_text": "relationship_name: con\x00tains; entity_description: bad\x00 text",
            "no\x00te": "nul\x00 byte",
        },
    )

    with patch.object(upsert_edges_module, "insert", return_value=statement):
        await upsert_edges(
            [edge],
            tenant_id=uuid4(),
            user_id=uuid4(),
            dataset_id=uuid4(),
            data_id=uuid4(),
            session=session,
        )

    payload = statement.values_arg[0]
    expected_relationship_name = "relationship_name: contains; entity_description: bad text"
    assert payload["relationship_name"] == expected_relationship_name
    assert payload["label"] == "contains"
    assert payload["attributes"] == {
        "edge_text": expected_relationship_name,
        "note": "nul byte",
    }
    session.execute.assert_awaited_once_with(statement)
    session.commit.assert_awaited_once()
