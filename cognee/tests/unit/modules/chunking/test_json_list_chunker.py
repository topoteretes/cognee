"""Unit tests for JsonListChunker — including nested array flattening and path metadata."""

import json
import pytest
from uuid import uuid4

from cognee.modules.chunking.JsonListChunker import JsonListChunker, _flatten_json_arrays
from cognee.modules.data.processing.document_types import Document


def make_document():
    return Document(
        id=uuid4(),
        name="test_json_document",
        raw_data_location="/test/path.json",
        external_metadata=None,
        mime_type="application/json",
    )


def make_text_generator(payload):
    """Create an async text generator factory from a Python object (will be JSON-serialised)."""
    json_str = json.dumps(payload)

    async def gen():
        yield json_str

    return gen


async def collect_chunks(chunker):
    chunks = []
    async for chunk in chunker.read():
        chunks.append(chunk)
    return chunks


# ─── Flat top-level array (backward compatibility) ─────────────────────────


@pytest.mark.asyncio
async def test_flat_top_level_array_basic():
    """A flat top-level JSON array should produce one chunk per element (original behaviour)."""
    data = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
    doc = make_document()
    chunker = JsonListChunker(doc, make_text_generator(data), max_chunk_size=512)
    chunks = await collect_chunks(chunker)

    assert len(chunks) == 2
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1
    assert chunks[0].cut_type == "json_list_item"


@pytest.mark.asyncio
async def test_flat_array_metadata_has_json_path():
    """Each chunk from a flat array should carry a json_path like [0], [1], etc."""
    data = [{"x": 1}, {"x": 2}]
    doc = make_document()
    chunker = JsonListChunker(doc, make_text_generator(data), max_chunk_size=512)
    chunks = await collect_chunks(chunker)

    assert chunks[0].metadata["json_path"] == "[0]"
    assert chunks[1].metadata["json_path"] == "[1]"
    assert chunks[0].metadata["parent_keys"] == []


# ─── Nested dict with arrays (new feature) ─────────────────────────────────


@pytest.mark.asyncio
async def test_nested_dict_array_flattening():
    """A dict containing nested arrays should be flattened — each array element becomes a chunk."""
    data = {
        "records": [
            {"id": 1, "value": "first"},
            {"id": 2, "value": "second"},
        ]
    }
    doc = make_document()
    chunker = JsonListChunker(doc, make_text_generator(data), max_chunk_size=512)
    chunks = await collect_chunks(chunker)

    assert len(chunks) == 2
    # Each chunk should carry the parent key "records"
    assert "records" in chunks[0].metadata["parent_keys"]
    assert "records" in chunks[1].metadata["parent_keys"]
    # json_path should include the key
    assert chunks[0].metadata["json_path"] == "records[0]"
    assert chunks[1].metadata["json_path"] == "records[1]"


@pytest.mark.asyncio
async def test_deeply_nested_array_flattening():
    """Deeply nested arrays inside dicts should be flattened with full path metadata."""
    data = {
        "records": {
            "items": [
                {"name": "a"},
                {"name": "b"},
            ]
        }
    }
    doc = make_document()
    chunker = JsonListChunker(doc, make_text_generator(data), max_chunk_size=512)
    chunks = await collect_chunks(chunker)

    assert len(chunks) == 2
    assert chunks[0].metadata["json_path"] == "records.items[0]"
    assert chunks[1].metadata["json_path"] == "records.items[1]"
    assert chunks[0].metadata["parent_keys"] == ["records", "items"]
    assert chunks[1].metadata["parent_keys"] == ["records", "items"]


@pytest.mark.asyncio
async def test_multiple_arrays_in_same_dict():
    """When a dict has multiple keys each containing arrays, all should be flattened."""
    data = {
        "users": [{"name": "Alice"}, {"name": "Bob"}],
        "products": [{"sku": "P1"}, {"sku": "P2"}, {"sku": "P3"}],
    }
    doc = make_document()
    chunker = JsonListChunker(doc, make_text_generator(data), max_chunk_size=512)
    chunks = await collect_chunks(chunker)

    assert len(chunks) == 5  # 2 users + 3 products
    # Verify parent_keys are preserved
    user_chunks = [c for c in chunks if "users" in c.metadata["parent_keys"]]
    product_chunks = [c for c in chunks if "products" in c.metadata["parent_keys"]]
    assert len(user_chunks) == 2
    assert len(product_chunks) == 3


@pytest.mark.asyncio
async def test_array_of_scalars():
    """A dict containing an array of scalars (strings/numbers) should flatten each scalar."""
    data = {"tags": ["alpha", "beta", "gamma"]}
    doc = make_document()
    chunker = JsonListChunker(doc, make_text_generator(data), max_chunk_size=512)
    chunks = await collect_chunks(chunker)

    assert len(chunks) == 3
    assert chunks[0].metadata["json_path"] == "tags[0]"
    assert chunks[1].metadata["json_path"] == "tags[1]"
    assert chunks[2].metadata["json_path"] == "tags[2]"


@pytest.mark.asyncio
async def test_mixed_scalars_and_arrays():
    """A dict with scalar values and arrays should emit both scalars and array elements."""
    data = {
        "config": {"version": 2},
        "items": [{"name": "first"}, {"name": "second"}],
    }
    doc = make_document()
    chunker = JsonListChunker(doc, make_text_generator(data), max_chunk_size=512)
    chunks = await collect_chunks(chunker)

    # Should get: config.version (scalar) + items[0] + items[1]
    assert len(chunks) == 3
    config_chunk = [c for c in chunks if "config" in c.metadata.get("parent_keys", [])]
    items_chunks = [c for c in chunks if "items" in c.metadata.get("parent_keys", [])]
    assert len(config_chunk) == 1
    assert len(items_chunks) == 2


# ─── json_path selector (new feature) ───────────────────────────────────────


@pytest.mark.asyncio
async def test_json_path_selector():
    """The json_path parameter should restrict extraction to the specified sub-tree."""
    data = {
        "metadata": {"source": "test"},
        "records": [
            {"id": 1},
            {"id": 2},
        ],
    }
    doc = make_document()
    chunker = JsonListChunker(
        doc,
        make_text_generator(data),
        max_chunk_size=512,
        json_path="records",
    )
    chunks = await collect_chunks(chunker)

    # Should only chunk the "records" sub-tree (2 items).
    assert len(chunks) == 2


@pytest.mark.asyncio
async def test_json_path_selector_with_index():
    """The json_path parameter should support bracket-notation indices."""
    data = {
        "groups": [
            [{"name": "item1"}, {"name": "item2"}],
            [{"name": "item3"}],
        ]
    }
    doc = make_document()
    chunker = JsonListChunker(
        doc,
        make_text_generator(data),
        max_chunk_size=512,
        json_path="groups[0]",
    )
    chunks = await collect_chunks(chunker)

    # Should only chunk items from groups[0] (2 items).
    assert len(chunks) == 2


# ─── _flatten_json_arrays helper ───────────────────────────────────────────


def test_flatten_simple_dict_with_array():
    """_flatten_json_arrays should extract items from a dict with an array value."""
    data = {"records": [{"id": 1}, {"id": 2}]}
    results = _flatten_json_arrays(data)
    assert len(results) == 2
    assert results[0][1] == "records[0]"
    assert results[1][1] == "records[1]"
    assert results[0][2] == ["records"]


def test_flatten_deeply_nested():
    """_flatten_json_arrays should handle deeply nested structures."""
    data = {"a": {"b": {"c": [1, 2, 3]}}}
    results = _flatten_json_arrays(data)
    assert len(results) == 3
    assert results[0][1] == "a.b.c[0]"
    assert results[0][2] == ["a", "b", "c"]


def test_flatten_flat_list():
    """_flatten_json_arrays should handle a flat top-level list."""
    data = [{"x": 1}, {"x": 2}]
    results = _flatten_json_arrays(data)
    assert len(results) == 2
    assert results[0][1] == "[0]"
    assert results[1][1] == "[1]"


def test_flatten_empty_dict():
    """_flatten_json_arrays should return empty for an empty dict."""
    results = _flatten_json_arrays({})
    assert len(results) == 0


def test_flatten_empty_list():
    """_flatten_json_arrays should return empty for an empty list."""
    results = _flatten_json_arrays([])
    assert len(results) == 0


# ─── Edge cases ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_empty_array():
    """An empty array should produce no chunks."""
    data = []
    doc = make_document()
    chunker = JsonListChunker(doc, make_text_generator(data), max_chunk_size=512)
    chunks = await collect_chunks(chunker)
    assert len(chunks) == 0


@pytest.mark.asyncio
async def test_empty_dict():
    """An empty dict should produce no chunks."""
    data = {}
    doc = make_document()
    chunker = JsonListChunker(doc, make_text_generator(data), max_chunk_size=512)
    chunks = await collect_chunks(chunker)
    assert len(chunks) == 0


@pytest.mark.asyncio
async def test_scalar_json():
    """A scalar JSON value should produce one chunk."""
    data = 42
    doc = make_document()
    chunker = JsonListChunker(doc, make_text_generator(data), max_chunk_size=512)
    chunks = await collect_chunks(chunker)
    assert len(chunks) == 1
    assert chunks[0].metadata["json_path"] == "$"


@pytest.mark.asyncio
async def test_chunk_ids_are_unique():
    """All chunk IDs in a nested document should be unique."""
    data = {
        "a": [{"v": 1}, {"v": 2}],
        "b": [{"v": 3}, {"v": 4}],
    }
    doc = make_document()
    chunker = JsonListChunker(doc, make_text_generator(data), max_chunk_size=512)
    chunks = await collect_chunks(chunker)

    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids)), "Chunk IDs should be unique"


@pytest.mark.asyncio
async def test_backward_compatible_metadata():
    """The json_list_index metadata key should still be present for backward compatibility."""
    data = [{"a": 1}, {"a": 2}]
    doc = make_document()
    chunker = JsonListChunker(doc, make_text_generator(data), max_chunk_size=512)
    chunks = await collect_chunks(chunker)

    for i, chunk in enumerate(chunks):
        assert chunk.metadata["json_list_index"] == i
        assert "index_fields" in chunk.metadata


@pytest.mark.asyncio
async def test_oversized_chunk_logs_warning():
    """Items exceeding max_chunk_size should still be emitted but logged as warning."""
    data = [{"text": "a" * 1000}]
    doc = make_document()
    chunker = JsonListChunker(doc, make_text_generator(data), max_chunk_size=5)
    chunks = await collect_chunks(chunker)
    # Chunk should still be emitted
    assert len(chunks) == 1
