"""Unit tests for JsonListChunker, including nested-array support."""

import json
import pytest
from uuid import uuid4

from cognee.modules.chunking.JsonListChunker import JsonListChunker
from cognee.modules.data.processing.document_types import Document


@pytest.fixture
def make_text_generator():
    """Factory for async text generators."""

    def _factory(*texts):
        async def gen():
            for text in texts:
                yield text

        return gen

    return _factory


def make_document():
    return Document(
        id=uuid4(),
        name="test_document",
        raw_data_location="/test/path",
        external_metadata=None,
        mime_type="application/json",
    )


async def collect_chunks(chunker):
    chunks = []
    async for chunk in chunker.read():
        chunks.append(chunk)
    return chunks


@pytest.mark.asyncio
async def test_flat_list_behavior_is_unchanged(make_text_generator):
    """Original flat-list behavior must still work exactly as before."""
    items = [{"name": "Alice"}, {"name": "Bob"}]
    get_text = make_text_generator(json.dumps(items))
    chunker = JsonListChunker(make_document(), get_text, max_chunk_size=512)
    chunks = await collect_chunks(chunker)

    assert len(chunks) == 2, "Should produce one chunk per list item"
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1
    assert chunks[0].metadata["json_path"] == "[0]"
    assert chunks[1].metadata["json_path"] == "[1]"
    assert "json_context" not in chunks[0].metadata, "Flat list has no sibling context to attach"


@pytest.mark.asyncio
async def test_single_nested_array_is_auto_detected(make_text_generator):
    """A dict with exactly one nested array should be found automatically."""
    data = {"records": {"items": [{"name": "Alice"}, {"name": "Bob"}]}}
    get_text = make_text_generator(json.dumps(data))
    chunker = JsonListChunker(make_document(), get_text, max_chunk_size=512)
    chunks = await collect_chunks(chunker)

    assert len(chunks) == 2, "Should find and chunk the nested array automatically"
    assert chunks[0].metadata["json_path"] == "records.items[0]"
    assert chunks[1].metadata["json_path"] == "records.items[1]"


@pytest.mark.asyncio
async def test_sibling_context_is_preserved(make_text_generator):
    """Scalar values next to the array should ride along in each chunk's metadata."""
    data = {"records": {"batch_id": "B1", "items": [{"name": "Alice"}]}}
    get_text = make_text_generator(json.dumps(data))
    chunker = JsonListChunker(make_document(), get_text, max_chunk_size=512)
    chunks = await collect_chunks(chunker)

    assert len(chunks) == 1
    assert chunks[0].metadata["json_context"] == {"batch_id": "B1"}


@pytest.mark.asyncio
async def test_multiple_nested_arrays_without_json_path_raises(make_text_generator):
    """Ambiguous documents (more than one array) must raise, not silently guess."""
    data = {
        "records": {"items": [{"name": "Alice"}]},
        "other": {"items": [{"name": "Bob"}]},
    }
    get_text = make_text_generator(json.dumps(data))
    chunker = JsonListChunker(make_document(), get_text, max_chunk_size=512)

    with pytest.raises(ValueError, match="multiple JSON lists"):
        await collect_chunks(chunker)


@pytest.mark.asyncio
async def test_json_path_resolves_correct_array_when_ambiguous(make_text_generator):
    """Passing json_path should pick the intended array even when multiple exist."""
    data = {
        "records": {"items": [{"name": "Alice"}]},
        "other": {"items": [{"name": "Bob"}]},
    }
    get_text = make_text_generator(json.dumps(data))
    chunker = JsonListChunker(
        make_document(), get_text, max_chunk_size=512, json_path="other.items"
    )
    chunks = await collect_chunks(chunker)

    assert len(chunks) == 1
    assert "Bob" in chunks[0].text
    assert chunks[0].metadata["json_path"] == "other.items[0]"


@pytest.mark.asyncio
async def test_no_array_found_raises(make_text_generator):
    """A dict with no nested array anywhere should raise a clear error."""
    data = {"name": "Alice", "age": 30}
    get_text = make_text_generator(json.dumps(data))
    chunker = JsonListChunker(make_document(), get_text, max_chunk_size=512)

    with pytest.raises(ValueError, match="could not find any JSON list"):
        await collect_chunks(chunker)


@pytest.mark.asyncio
async def test_invalid_json_path_raises(make_text_generator):
    """A json_path that doesn't exist in the document should raise a clear error."""
    data = {"records": {"items": [{"name": "Alice"}]}}
    get_text = make_text_generator(json.dumps(data))
    chunker = JsonListChunker(
        make_document(), get_text, max_chunk_size=512, json_path="records.missing"
    )

    with pytest.raises(ValueError, match="does not exist"):
        await collect_chunks(chunker)


@pytest.mark.asyncio
async def test_non_list_non_dict_top_level_raises(make_text_generator):
    """A JSON document that is neither a list nor a dict should raise a clear error."""
    get_text = make_text_generator(json.dumps("just a string"))
    chunker = JsonListChunker(make_document(), get_text, max_chunk_size=512)

    with pytest.raises(ValueError, match="JSON list or a JSON object"):
        await collect_chunks(chunker)
