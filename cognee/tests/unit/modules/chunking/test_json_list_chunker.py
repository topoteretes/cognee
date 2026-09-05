"""Unit tests for JsonListChunker with nested JSON support."""

import pytest
from uuid import uuid4

from cognee.modules.chunking.JsonListChunker import (
    JsonListChunker,
    find_json_arrays,
    get_parent_context,
)
from cognee.modules.data.processing.document_types import Document


@pytest.fixture
def make_json_generator():
    """Factory for async JSON text generators."""

    def _factory(json_str: str):
        async def gen():
            yield json_str

        return gen

    return _factory


async def collect_chunks(chunker):
    """Consume async generator and return list of chunks."""
    chunks = []
    async for chunk in chunker.read():
        chunks.append(chunk)
    return chunks


@pytest.fixture
def sample_flat_list():
    return [{"id": 1, "name": "item1"}, {"id": 2, "name": "item2"}]


@pytest.fixture
def sample_nested_dict():
    return {
        "batch_id": "batch-123",
        "records": {
            "items": [
                {"id": 1, "value": "first"},
                {"id": 2, "value": "second"},
                {"id": 3, "value": "third"},
            ]
        },
        "metadata": {"source": "api"},
    }


@pytest.fixture
def sample_multiple_arrays():
    return {"users": [{"id": 1}, {"id": 2}], "products": [{"sku": "A"}, {"sku": "B"}]}


@pytest.fixture
def sample_deeply_nested():
    return {"level1": {"level2": {"items": [{"deep": 1}, {"deep": 2}]}}}


# ── find_json_arrays tests ──


@pytest.mark.asyncio
async def test_find_json_arrays_flat_list():
    data = [1, 2, 3]
    result = find_json_arrays(data)
    assert result == [("$", [1, 2, 3])]


@pytest.mark.asyncio
async def test_find_json_arrays_nested_single():
    data = {"records": {"items": [1, 2, 3]}}
    result = find_json_arrays(data)
    assert len(result) == 1
    assert result[0] == ("$.records.items", [1, 2, 3])


@pytest.mark.asyncio
async def test_find_json_arrays_multiple():
    data = {"users": [1, 2], "products": ["a", "b"]}
    result = find_json_arrays(data)
    assert len(result) == 2
    paths = [p for p, _ in result]
    assert "$.users" in paths
    assert "$.products" in paths


@pytest.mark.asyncio
async def test_find_json_arrays_deeply_nested():
    data = {"a": {"b": {"c": [1, 2]}}}
    result = find_json_arrays(data)
    assert len(result) == 1
    assert result[0][0] == "$.a.b.c"


@pytest.mark.asyncio
async def test_find_json_arrays_empty():
    data = {"key": "value", "nested": {"empty": {}}}
    result = find_json_arrays(data)
    assert result == []


# ── get_parent_context tests ──


@pytest.mark.asyncio
async def test_get_parent_context_sibling_values():
    data = {"batch_id": "batch-123", "records": {"items": [1, 2, 3]}}
    context = get_parent_context(data, "$.records.items")
    assert context == {"batch_id": "batch-123"}


@pytest.mark.asyncio
async def test_get_parent_context_no_siblings():
    data = {"records": {"items": [1, 2, 3]}}
    context = get_parent_context(data, "$.records.items")
    assert context == {}


@pytest.mark.asyncio
async def test_get_parent_context_root_level():
    data = {"batch_id": "batch-123", "items": [1, 2, 3]}
    context = get_parent_context(data, "$.items")
    assert context == {"batch_id": "batch-123"}


@pytest.mark.asyncio
async def test_get_parent_context_non_dict_parent():
    data = [1, 2, 3]
    context = get_parent_context(data, "$")
    assert context == {}


# ── JsonListChunker tests ──


@pytest.fixture
def document(sample_flat_list):
    return Document(
        id=uuid4(),
        name="test_doc",
        raw_data_location="/test/path.json",
        external_metadata=None,
        mime_type="application/json",
    )


@pytest.fixture
def make_json_gen():
    def _factory(json_data):
        async def gen():
            import json

            yield json.dumps(json_data)

        return gen

    return _factory


# Test 1: Original flat list behavior (backward compatibility)
@pytest.mark.asyncio
async def test_flat_list_backward_compatibility(document, make_json_gen, sample_flat_list):
    """Flat JSON list at root should work as before."""
    document.metadata = {}
    get_text = make_json_gen(sample_flat_list)
    chunker = JsonListChunker(document, get_text, max_chunk_size=1000)
    chunks = await collect_chunks(chunker)

    assert len(chunks) == 2
    assert chunks[0].metadata["json_list_index"] == 0
    assert chunks[1].metadata["json_list_index"] == 1
    assert chunks[0].metadata["json_path"] == "$[0]"
    assert chunks[1].metadata["json_path"] == "$[1]"
    assert chunks[0].metadata["parent_context"] == {}
    assert "id" in chunks[0].text


# Test 2: Nested array with auto-detection
@pytest.mark.asyncio
async def test_nested_array_auto_detect(document, make_json_gen, sample_nested_dict):
    """Nested array should be auto-detected when auto_detect=True."""
    document.metadata = {"auto_detect": True}
    get_text = make_json_gen(sample_nested_dict)
    chunker = JsonListChunker(document, get_text, max_chunk_size=1000)
    chunks = await collect_chunks(chunker)

    assert len(chunks) == 3
    assert chunks[0].metadata["json_list_index"] == 0
    assert chunks[0].metadata["json_path"] == "$.records.items[0]"
    assert chunks[1].metadata["json_path"] == "$.records.items[1]"
    assert chunks[2].metadata["json_path"] == "$.records.items[2]"
    # Parent context should include sibling values
    assert chunks[0].metadata["parent_context"] == {
        "batch_id": "batch-123",
        "metadata": {"source": "api"},
    }


# Test 3: Explicit json_path selector
@pytest.mark.asyncio
async def test_explicit_json_path(document, make_json_gen, sample_nested_dict):
    """Explicit json_path should select the specified array."""
    document.metadata = {"json_path": "$.records.items"}
    get_text = make_json_gen(sample_nested_dict)
    chunker = JsonListChunker(document, get_text, max_chunk_size=1000)
    chunks = await collect_chunks(chunker)

    assert len(chunks) == 3
    assert chunks[0].metadata["json_path"] == "$.records.items[0]"


# Test 4: json_path with dict root (no $ prefix)
@pytest.mark.asyncio
async def test_json_path_without_dollar_prefix(document, make_json_gen, sample_nested_dict):
    """json_path should work with or without $ prefix."""
    document.metadata = {"json_path": "records.items"}
    get_text = make_json_gen(sample_nested_dict)
    chunker = JsonListChunker(document, get_text, max_chunk_size=1000)
    chunks = await collect_chunks(chunker)

    assert len(chunks) == 3
    assert chunks[0].metadata["json_path"] == "$.records.items[0]"


# Test 5: Multiple arrays without json_path should error
@pytest.mark.asyncio
async def test_multiple_arrays_requires_json_path(document, make_json_gen, sample_multiple_arrays):
    """Multiple arrays without json_path should raise ValueError."""
    document.metadata = {"auto_detect": True}
    get_text = make_json_gen(sample_multiple_arrays)
    chunker = JsonListChunker(document, get_text, max_chunk_size=1000)

    with pytest.raises(ValueError, match="Multiple JSON arrays found"):
        await collect_chunks(chunker)


# Test 6: Multiple arrays with json_path should work
@pytest.mark.asyncio
async def test_multiple_arrays_with_json_path(document, make_json_gen, sample_multiple_arrays):
    """Multiple arrays with explicit json_path should work."""
    document.metadata = {"json_path": "$.users"}
    get_text = make_json_gen(sample_multiple_arrays)
    chunker = JsonListChunker(document, get_text, max_chunk_size=1000)
    chunks = await collect_chunks(chunker)

    assert len(chunks) == 2
    assert chunks[0].metadata["json_path"] == "$.users[0]"


# Test 7: Deeply nested array
@pytest.mark.asyncio
async def test_deeply_nested_array(document, make_json_gen, sample_deeply_nested):
    """Deeply nested arrays should be found via auto-detection."""
    document.metadata = {"auto_detect": True}
    get_text = make_json_gen(sample_deeply_nested)
    chunker = JsonListChunker(document, get_text, max_chunk_size=1000)
    chunks = await collect_chunks(chunker)

    assert len(chunks) == 2
    assert chunks[0].metadata["json_path"] == "$.level1.level2.items[0]"


# Test 8: Parent context includes sibling simple values
@pytest.mark.asyncio
async def test_parent_context_includes_siblings(document, make_json_gen, sample_nested_dict):
    """Parent context should include non-array, non-dict siblings."""
    document.metadata = {"auto_detect": True}
    get_text = make_json_gen(sample_nested_dict)
    chunker = JsonListChunker(document, get_text, max_chunk_size=1000)
    chunks = await collect_chunks(chunker)

    ctx = chunks[0].metadata["parent_context"]
    assert "batch_id" in ctx
    assert ctx["batch_id"] == "batch-123"
    assert "metadata" in ctx
    assert ctx["metadata"] == {"source": "api"}


# Test 9: Explicit json_path overrides auto-detect
@pytest.mark.asyncio
async def test_explicit_path_overrides_auto_detect(document, make_json_gen, sample_multiple_arrays):
    """Explicit json_path should take precedence over auto-detection."""
    document.metadata = {"json_path": "$.products", "auto_detect": True}
    get_text = make_json_gen(sample_multiple_arrays)
    chunker = JsonListChunker(document, get_text, max_chunk_size=1000)
    chunks = await collect_chunks(chunker)

    assert len(chunks) == 2
    assert all(c.metadata["json_path"].startswith("$.products") for c in chunks)


# Test 10: Invalid json_path raises error
@pytest.mark.asyncio
async def test_invalid_json_path_raises_error(document, make_json_gen, sample_nested_dict):
    """Invalid json_path should raise ValueError with clear message."""
    document.metadata = {"json_path": "$.nonexistent.path"}
    get_text = make_json_gen(sample_nested_dict)
    chunker = JsonListChunker(document, get_text, max_chunk_size=1000)

    with pytest.raises(ValueError, match="Invalid json_path"):
        await collect_chunks(chunker)


# Test 11: json_path pointing to non-array raises error
@pytest.mark.asyncio
async def test_json_path_to_non_array_raises_error(document, make_json_gen, sample_nested_dict):
    """json_path pointing to a non-array should raise ValueError."""
    document.metadata = {"json_path": "$.metadata"}
    get_text = make_json_gen(sample_nested_dict)
    chunker = JsonListChunker(document, get_text, max_chunk_size=1000)

    with pytest.raises(ValueError, match="does not resolve to a list"):
        await collect_chunks(chunker)


# Test 12: Empty document raises error
@pytest.mark.asyncio
async def test_empty_object_raises_error(document, make_json_gen):
    """Empty JSON object with no arrays should raise error."""
    document.metadata = {"auto_detect": True}
    get_text = make_json_gen({"key": "value"})
    chunker = JsonListChunker(document, get_text, max_chunk_size=1000)

    with pytest.raises(ValueError, match="No suitable JSON array found"):
        await collect_chunks(chunker)


# Test 13: json_path in document.metadata takes precedence over class attribute
@pytest.mark.asyncio
async def test_metadata_precedence_over_class_attr(document, make_json_gen, sample_multiple_arrays):
    """document.metadata json_path should override class-level json_path."""
    # Set class-level default
    JsonListChunker.json_path = "$.users"
    try:
        document.metadata = {"json_path": "$.products"}
        get_text = make_json_gen(sample_multiple_arrays)
        chunker = JsonListChunker(
            document, make_json_gen(sample_multiple_arrays), max_chunk_size=1000
        )
        chunks = await collect_chunks(chunker)

        assert len(chunks) == 2
        assert all(c.metadata["json_path"].startswith("$.products") for c in chunks)
    finally:
        # Reset class attribute
        JsonListChunker.json_path = None


# Test 14: Invalid JSON raises error
@pytest.mark.asyncio
async def test_invalid_json_raises_error(document):
    """Invalid JSON should raise ValueError."""
    document.metadata = {}

    async def bad_gen():
        yield "not valid json {"

    chunker = JsonListChunker(document, bad_gen, max_chunk_size=1000)

    with pytest.raises(ValueError, match="Invalid JSON"):
        await collect_chunks(chunker)


# Test 15: Chunk metadata includes all required fields
@pytest.mark.asyncio
async def test_chunk_metadata_completeness(document, make_json_gen, sample_nested_dict):
    """Each chunk should have complete metadata."""
    document.metadata = {"auto_detect": True}
    get_text = make_json_gen(sample_nested_dict)
    chunker = JsonListChunker(document, get_text, max_chunk_size=1000)
    chunks = await collect_chunks(chunker)

    for i, chunk in enumerate(chunks):
        meta = chunk.metadata
        assert "index_fields" in meta
        assert meta["index_fields"] == ["text"]
        assert "json_list_index" in meta
        assert meta["json_list_index"] == i
        assert "json_path" in meta
        assert meta["json_path"] == f"$.records.items[{i}]"
        assert "parent_context" in meta
        assert "batch_id" in meta["parent_context"]


# Test 16: Chunk IDs are deterministic for same input
@pytest.mark.asyncio
async def test_deterministic_chunk_ids(document, make_json_gen, sample_flat_list):
    """Same input should produce deterministic chunk IDs."""
    doc_id = uuid4()
    doc1 = Document(
        id=doc_id,
        name="test_doc",
        raw_data_location="/test/path.json",
        external_metadata=None,
        mime_type="application/json",
    )
    doc2 = Document(
        id=doc_id,
        name="test_doc",
        raw_data_location="/test/path.json",
        external_metadata=None,
        mime_type="application/json",
    )

    chunker1 = JsonListChunker(doc1, make_json_gen(sample_flat_list), max_chunk_size=1000)
    chunker2 = JsonListChunker(doc2, make_json_gen(sample_flat_list), max_chunk_size=1000)

    chunks1 = await collect_chunks(chunker1)
    chunks2 = await collect_chunks(chunker2)

    assert chunks1[0].id == chunks2[0].id
    assert chunks1[1].id == chunks2[1].id


# Test 17: Class-level defaults work
@pytest.mark.asyncio
async def test_class_level_defaults(document, make_json_gen, sample_nested_dict):
    """Class-level json_path and auto_detect defaults should work."""
    JsonListChunker.json_path = "$.records.items"
    JsonListChunker.auto_detect = False
    try:
        document.metadata = {}
        get_text = make_json_gen(sample_nested_dict)
        chunker = JsonListChunker(document, get_text, max_chunk_size=1000)
        chunks = await collect_chunks(chunker)

        assert len(chunks) == 3
        assert all(c.metadata["json_path"].startswith("$.records.items") for c in chunks)
    finally:
        JsonListChunker.json_path = None
        JsonListChunker.auto_detect = True


# Test 18: Document metadata overrides class-level defaults
@pytest.mark.asyncio
async def test_metadata_overrides_class_defaults(document, make_json_gen, sample_multiple_arrays):
    """document.metadata should override class-level json_path."""
    JsonListChunker.json_path = "$.users"
    try:
        document.metadata = {"json_path": "$.products"}
        get_text = make_json_gen(sample_multiple_arrays)
        chunker = JsonListChunker(document, get_text, max_chunk_size=1000)
        chunks = await collect_chunks(chunker)

        assert len(chunks) == 2
        assert all(c.metadata["json_path"].startswith("$.products") for c in chunks)
    finally:
        JsonListChunker.json_path = None
