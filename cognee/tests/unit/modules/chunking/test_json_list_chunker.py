"""Unit tests for the enhanced JsonListChunker.

Coverage matrix
---------------
Scenario                                        | Test(s)
------------------------------------------------|-------------------------------
Flat root list – original behaviour preserved   | test_flat_list_*
Nested object with auto-detected array          | test_nested_auto_*
Explicit json_path selector                     | test_json_path_*
Parent-context injection                        | test_parent_context_*
Nested field flattening (_flatten_nested)       | test_flatten_*
flatten_nested=False (backward-compat repr)     | test_no_flatten_*
Chunk metadata fields                           | test_metadata_*
Error handling (bad JSON, bad path, bad type)   | test_error_*
Edge cases (empty list, single item, scalars)   | test_edge_*
"""

import json
from uuid import uuid4

import pytest

from cognee.modules.chunking.JsonListChunker import (
    JsonListChunker,
    _flatten_nested,
    _find_nested_array,
    _item_to_text,
)
from cognee.modules.data.processing.document_types import Document


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _doc(name="test.json") -> Document:
    return Document(
        id=uuid4(),
        name=name,
        raw_data_location=f"/tmp/{name}",
        external_metadata=None,
        mime_type="application/json",
    )


def _gen(*texts):
    """Return an async generator that yields the given strings one at a time."""

    async def _inner():
        for t in texts:
            yield t

    return _inner


async def _chunks(chunker: JsonListChunker) -> list:
    return [chunk async for chunk in chunker.read()]


# ---------------------------------------------------------------------------
# _flatten_nested unit tests
# ---------------------------------------------------------------------------


def test_flatten_flat_dict():
    assert _flatten_nested({"a": 1, "b": "hello"}) == {"a": 1, "b": "hello"}


def test_flatten_nested_dict():
    assert _flatten_nested({"a": {"b": {"c": 42}}}) == {"a.b.c": 42}


def test_flatten_scalar_list():
    assert _flatten_nested({"tags": ["x", "y", "z"]}) == {"tags": "x, y, z"}


def test_flatten_object_list():
    result = _flatten_nested({"users": [{"name": "Alice"}, {"name": "Bob"}]})
    assert result == {"users.0.name": "Alice", "users.1.name": "Bob"}


def test_flatten_mixed_depth():
    data = {
        "id": 1,
        "meta": {"created": "2024-01"},
        "scores": [10, 20],
    }
    result = _flatten_nested(data)
    assert result["id"] == 1
    assert result["meta.created"] == "2024-01"
    assert result["scores"] == "10, 20"


def test_flatten_boolean_and_none():
    result = _flatten_nested({"active": True, "deleted": None})
    assert result["active"] is True
    assert result["deleted"] is None


def test_flatten_custom_separator():
    result = _flatten_nested({"a": {"b": 1}}, separator="/")
    assert result["a/b"] == 1


# ---------------------------------------------------------------------------
# _item_to_text unit tests
# ---------------------------------------------------------------------------


def test_item_to_text_dict_flattened():
    item = {"name": "Alice", "address": {"city": "Berlin"}}
    text = _item_to_text(item, flatten_nested=True)
    assert "name: Alice" in text
    assert "address.city: Berlin" in text


def test_item_to_text_dict_no_flatten():
    item = {"name": "Alice"}
    text = _item_to_text(item, flatten_nested=False)
    assert text == str(item)


def test_item_to_text_scalar():
    assert _item_to_text(42, flatten_nested=True) == "42"
    assert _item_to_text("hello", flatten_nested=True) == "hello"


# ---------------------------------------------------------------------------
# _find_nested_array unit tests
# ---------------------------------------------------------------------------


def test_find_array_root_list():
    items, ctx = _find_nested_array([1, 2, 3], None)
    assert items == [1, 2, 3]
    assert ctx == {}


def test_find_array_auto_detect_object():
    data = {"version": "1.0", "records": [{"id": 1}, {"id": 2}]}
    items, ctx = _find_nested_array(data, None)
    assert items == [{"id": 1}, {"id": 2}]
    assert ctx == {"version": "1.0"}


def test_find_array_explicit_json_path():
    data = {"meta": {"created": "2024"}, "data": {"items": [{"x": 1}]}}
    items, ctx = _find_nested_array(data, "data.items")
    assert items == [{"x": 1}]
    assert "meta" not in ctx  # only direct siblings collected
    assert "data.created" not in ctx


def test_find_array_json_path_collects_siblings():
    data = {"name": "api-response", "payload": {"count": 3, "rows": [{"a": 1}]}}
    items, ctx = _find_nested_array(data, "payload.rows")
    assert items == [{"a": 1}]
    assert ctx.get("payload.count") == 3


def test_find_array_error_no_array_in_object():
    with pytest.raises(ValueError, match="auto-detect"):
        _find_nested_array({"key": "value"}, None)


def test_find_array_error_bad_json_path():
    with pytest.raises(ValueError, match="missing_key"):
        _find_nested_array({"a": [1, 2]}, "missing_key.sub")


def test_find_array_error_path_not_array():
    with pytest.raises(ValueError, match="not a list"):
        _find_nested_array({"a": {"b": "not-a-list"}}, "a.b")


def test_find_array_error_unsupported_root_type():
    with pytest.raises(ValueError, match="unsupported root type"):
        _find_nested_array(42, None)


# ---------------------------------------------------------------------------
# JsonListChunker – flat list (original behaviour)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_flat_list_yields_one_chunk_per_item():
    doc = _doc()
    chunker = JsonListChunker(doc, _gen('[{"a": 1}, {"b": 2}]'), max_chunk_size=512)
    chunks = await _chunks(chunker)
    assert len(chunks) == 2
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1


@pytest.mark.asyncio
async def test_flat_list_text_contains_key_values():
    doc = _doc()
    chunker = JsonListChunker(doc, _gen('[{"name": "Alice", "age": 30}]'), max_chunk_size=512)
    chunks = await _chunks(chunker)
    assert "name: Alice" in chunks[0].text
    assert "age: 30" in chunks[0].text


@pytest.mark.asyncio
async def test_flat_list_cut_type_is_json_list_item():
    doc = _doc()
    chunker = JsonListChunker(doc, _gen("[1, 2]"), max_chunk_size=512)
    chunks = await _chunks(chunker)
    assert all(c.cut_type == "json_list_item" for c in chunks)


@pytest.mark.asyncio
async def test_flat_list_chunker_id_stamped():
    doc = _doc()
    chunker = JsonListChunker(doc, _gen('[{"x": 1}]'), max_chunk_size=512)
    chunks = await _chunks(chunker)
    assert all(c.chunker_id == "json_list_chunker_v1" for c in chunks)


@pytest.mark.asyncio
async def test_flat_list_index_in_metadata():
    doc = _doc()
    chunker = JsonListChunker(doc, _gen('[{"a": 1}, {"b": 2}]'), max_chunk_size=512)
    chunks = await _chunks(chunker)
    assert chunks[0].metadata["json_list_index"] == 0
    assert chunks[1].metadata["json_list_index"] == 1


# ---------------------------------------------------------------------------
# JsonListChunker – nested object with auto-detection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_nested_auto_detects_array():
    payload = json.dumps({"version": "2.0", "users": [{"name": "Alice"}, {"name": "Bob"}]})
    doc = _doc()
    chunker = JsonListChunker(doc, _gen(payload), max_chunk_size=512)
    chunks = await _chunks(chunker)
    assert len(chunks) == 2


@pytest.mark.asyncio
async def test_nested_auto_parent_context_in_text():
    payload = json.dumps({"source": "erp", "records": [{"id": 1}]})
    doc = _doc()
    chunker = JsonListChunker(doc, _gen(payload), max_chunk_size=512)
    chunks = await _chunks(chunker)
    assert "source=erp" in chunks[0].text


@pytest.mark.asyncio
async def test_nested_auto_parent_context_in_metadata():
    payload = json.dumps({"source": "erp", "records": [{"id": 1}]})
    doc = _doc()
    chunker = JsonListChunker(doc, _gen(payload), max_chunk_size=512)
    chunks = await _chunks(chunker)
    assert chunks[0].metadata["json_parent_context"]["source"] == "erp"


# ---------------------------------------------------------------------------
# JsonListChunker – explicit json_path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_json_path_selects_nested_array():
    payload = json.dumps({"data": {"rows": [{"val": "A"}, {"val": "B"}, {"val": "C"}]}})
    doc = _doc()
    chunker = JsonListChunker(doc, _gen(payload), max_chunk_size=512, json_path="data.rows")
    chunks = await _chunks(chunker)
    assert len(chunks) == 3


@pytest.mark.asyncio
async def test_json_path_recorded_in_metadata():
    payload = json.dumps({"items": [{"x": 1}]})
    doc = _doc()
    chunker = JsonListChunker(doc, _gen(payload), max_chunk_size=512, json_path="items")
    chunks = await _chunks(chunker)
    assert chunks[0].metadata["json_path"] == "items"


@pytest.mark.asyncio
async def test_json_path_error_on_bad_path():
    payload = json.dumps({"items": [{"x": 1}]})
    doc = _doc()
    chunker = JsonListChunker(
        doc, _gen(payload), max_chunk_size=512, json_path="missing.path"
    )
    with pytest.raises(ValueError):
        await _chunks(chunker)


# ---------------------------------------------------------------------------
# JsonListChunker – parent context control
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_include_parent_context_false_suppresses_prefix():
    payload = json.dumps({"source": "erp", "records": [{"id": 1}]})
    doc = _doc()
    chunker = JsonListChunker(
        doc, _gen(payload), max_chunk_size=512, include_parent_context=False
    )
    chunks = await _chunks(chunker)
    assert "source" not in chunks[0].text


@pytest.mark.asyncio
async def test_include_parent_context_metadata_still_stored():
    """Parent context is stored in metadata even when suppressed from text."""
    payload = json.dumps({"source": "erp", "records": [{"id": 1}]})
    doc = _doc()
    chunker = JsonListChunker(
        doc, _gen(payload), max_chunk_size=512, include_parent_context=False
    )
    chunks = await _chunks(chunker)
    # metadata["json_parent_context"] is set by _find_nested_array regardless
    assert "json_parent_context" in chunks[0].metadata


# ---------------------------------------------------------------------------
# JsonListChunker – flatten_nested control
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_flatten_nested_true_renders_key_value_lines():
    payload = json.dumps([{"user": {"name": "Alice", "age": 30}}])
    doc = _doc()
    chunker = JsonListChunker(doc, _gen(payload), max_chunk_size=512, flatten_nested=True)
    chunks = await _chunks(chunker)
    assert "user.name: Alice" in chunks[0].text
    assert "user.age: 30" in chunks[0].text


@pytest.mark.asyncio
async def test_flatten_nested_false_keeps_repr():
    payload = json.dumps([{"user": {"name": "Alice"}}])
    doc = _doc()
    chunker = JsonListChunker(doc, _gen(payload), max_chunk_size=512, flatten_nested=False)
    chunks = await _chunks(chunker)
    # Repr/str output, not key: value format
    assert "user.name:" not in chunks[0].text
    assert "Alice" in chunks[0].text


@pytest.mark.asyncio
async def test_flatten_array_values_joined():
    payload = json.dumps([{"tags": ["python", "ai", "memory"]}])
    doc = _doc()
    chunker = JsonListChunker(doc, _gen(payload), max_chunk_size=512, flatten_nested=True)
    chunks = await _chunks(chunker)
    assert "tags: python, ai, memory" in chunks[0].text


# ---------------------------------------------------------------------------
# JsonListChunker – error handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_error_on_invalid_json():
    doc = _doc()
    chunker = JsonListChunker(doc, _gen("{not valid json"), max_chunk_size=512)
    with pytest.raises(ValueError, match="could not parse"):
        await _chunks(chunker)


@pytest.mark.asyncio
async def test_error_on_root_not_list_or_object():
    doc = _doc()
    chunker = JsonListChunker(doc, _gen("42"), max_chunk_size=512)
    with pytest.raises(ValueError):
        await _chunks(chunker)


@pytest.mark.asyncio
async def test_error_on_object_no_array():
    doc = _doc()
    chunker = JsonListChunker(doc, _gen('{"key": "value"}'), max_chunk_size=512)
    with pytest.raises(ValueError, match="auto-detect"):
        await _chunks(chunker)


# ---------------------------------------------------------------------------
# JsonListChunker – edge cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_edge_empty_list():
    doc = _doc()
    chunker = JsonListChunker(doc, _gen("[]"), max_chunk_size=512)
    chunks = await _chunks(chunker)
    assert chunks == []


@pytest.mark.asyncio
async def test_edge_single_item():
    doc = _doc()
    chunker = JsonListChunker(doc, _gen('[{"only": "one"}]'), max_chunk_size=512)
    chunks = await _chunks(chunker)
    assert len(chunks) == 1


@pytest.mark.asyncio
async def test_edge_scalar_items_still_work():
    """Plain scalar arrays remain supported (original behaviour)."""
    doc = _doc()
    chunker = JsonListChunker(doc, _gen('["hello", "world"]'), max_chunk_size=512)
    chunks = await _chunks(chunker)
    assert len(chunks) == 2
    assert chunks[0].text == "hello"
    assert chunks[1].text == "world"


@pytest.mark.asyncio
async def test_edge_multi_part_text_stream():
    """Content arriving in multiple generator yields is assembled correctly."""
    doc = _doc()
    # Split mid-JSON to simulate streaming
    chunker = JsonListChunker(doc, _gen('[{"a": 1}', ", ", '{"b": 2}]'), max_chunk_size=512)
    chunks = await _chunks(chunker)
    assert len(chunks) == 2


@pytest.mark.asyncio
async def test_edge_deeply_nested_json_path():
    payload = json.dumps(
        {"level1": {"level2": {"level3": {"items": [{"v": "deep"}]}}}}
    )
    doc = _doc()
    chunker = JsonListChunker(
        doc, _gen(payload), max_chunk_size=512, json_path="level1.level2.level3.items"
    )
    chunks = await _chunks(chunker)
    assert len(chunks) == 1
    assert "v: deep" in chunks[0].text


@pytest.mark.asyncio
async def test_edge_importance_weight_propagated():
    doc = _doc()
    doc.importance_weight = 0.9
    chunker = JsonListChunker(doc, _gen('[{"x": 1}]'), max_chunk_size=512)
    chunks = await _chunks(chunker)
    assert chunks[0].importance_weight == 0.9


@pytest.mark.asyncio
async def test_edge_text_is_indexed():
    doc = _doc()
    chunker = JsonListChunker(doc, _gen('[{"x": 1}]'), max_chunk_size=512)
    chunks = await _chunks(chunker)
    assert "text" in chunks[0].metadata["index_fields"]
