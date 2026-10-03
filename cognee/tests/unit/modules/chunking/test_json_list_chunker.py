"""Unit tests for JsonListChunker nested JSON support."""

import json
from uuid import uuid4

import pytest

from cognee.modules.chunking.JsonListChunker import JsonListChunker
from cognee.modules.data.processing.document_types import Document


def _make_text_generator(*texts):
    async def gen():
        for text in texts:
            yield text

    return gen


async def _collect(chunker):
    chunks = []
    async for chunk in chunker.read():
        chunks.append(chunk)
    return chunks


def _make_document(name="test.json"):
    return Document(
        id=uuid4(),
        name=name,
        raw_data_location=f"/test/path/{name}",
        external_metadata=None,
        mime_type="application/json",
    )


@pytest.mark.asyncio
async def test_flat_list_still_works():
    """Original flat-list behavior is preserved."""
    data = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
    document = _make_document()
    get_text = _make_text_generator(json.dumps(data))
    chunker = JsonListChunker(document, get_text, max_chunk_size=512)

    chunks = await _collect(chunker)

    assert len(chunks) == 2
    assert chunks[0].text == str(data[0])
    assert chunks[1].text == str(data[1])
    assert chunks[0].metadata["json_list_index"] == 0
    assert chunks[1].metadata["json_list_index"] == 1
    assert chunks[0].metadata["json_path"] == "[0]"
    assert chunks[1].metadata["json_path"] == "[1]"


@pytest.mark.asyncio
async def test_nested_dict_with_auto_detection():
    """Dict with nested list is auto-detected (longest list wins)."""
    data = {
        "metadata": {"version": "1.0"},
        "records": [
            {"id": 1, "value": "a"},
            {"id": 2, "value": "b"},
            {"id": 3, "value": "c"},
        ],
        "errors": [],
    }
    document = _make_document()
    get_text = _make_text_generator(json.dumps(data))
    chunker = JsonListChunker(document, get_text, max_chunk_size=512)

    chunks = await _collect(chunker)

    assert len(chunks) == 3
    assert chunks[0].text == str(data["records"][0])
    assert chunks[0].metadata["json_path"] == "records[0]"
    assert chunks[1].metadata["json_path"] == "records[1]"
    assert chunks[2].metadata["json_path"] == "records[2]"


@pytest.mark.asyncio
async def test_nested_dict_with_json_path():
    """Explicit json_path selects the correct nested list."""
    data = {
        "data": {
            "users": [
                {"name": "Alice"},
                {"name": "Bob"},
            ],
            "admins": [
                {"name": "Charlie"},
            ],
        }
    }
    document = _make_document()
    get_text = _make_text_generator(json.dumps(data))
    chunker = JsonListChunker(document, get_text, max_chunk_size=512, json_path="data.users")

    chunks = await _collect(chunker)

    assert len(chunks) == 2
    assert chunks[0].text == str(data["data"]["users"][0])
    assert chunks[1].text == str(data["data"]["users"][1])
    assert chunks[0].metadata["json_path"] == "data.users[0]"
    assert chunks[1].metadata["json_path"] == "data.users[1]"


@pytest.mark.asyncio
async def test_parent_context_preserved():
    """Primitive parent key-values are preserved in chunk metadata."""
    data = {
        "company": "Acme",
        "department": "Engineering",
        "employees": [
            {"name": "Alice", "role": "Lead"},
            {"name": "Bob", "role": "Dev"},
        ],
    }
    document = _make_document()
    get_text = _make_text_generator(json.dumps(data))
    chunker = JsonListChunker(document, get_text, max_chunk_size=512)

    chunks = await _collect(chunker)

    assert len(chunks) == 2
    for chunk in chunks:
        assert chunk.metadata["company"] == "Acme"
        assert chunk.metadata["department"] == "Engineering"


@pytest.mark.asyncio
async def test_dict_without_list_raises():
    """A dict with no nested lists raises ValueError."""
    data = {"a": 1, "b": "hello", "c": {"d": 2}}
    document = _make_document()
    get_text = _make_text_generator(json.dumps(data))
    chunker = JsonListChunker(document, get_text, max_chunk_size=512)

    with pytest.raises(ValueError, match="at least one nested list"):
        await _collect(chunker)


@pytest.mark.asyncio
async def test_non_list_json_path_raises():
    """json_path pointing to a non-list raises ValueError."""
    data = {"records": {"key": "value"}}
    document = _make_document()
    get_text = _make_text_generator(json.dumps(data))
    chunker = JsonListChunker(document, get_text, max_chunk_size=512, json_path="records")

    with pytest.raises(ValueError, match="does not point to a list"):
        await _collect(chunker)


@pytest.mark.asyncio
async def test_invalid_json_path_raises():
    """json_path that doesn't exist raises ValueError."""
    data = {"records": [{"id": 1}]}
    document = _make_document()
    get_text = _make_text_generator(json.dumps(data))
    chunker = JsonListChunker(document, get_text, max_chunk_size=512, json_path="nonexistent")

    with pytest.raises(ValueError, match="not found"):
        await _collect(chunker)


@pytest.mark.asyncio
async def test_non_list_non_dict_raises():
    """A JSON string that is neither list nor dict raises ValueError."""
    data = "just a string"
    document = _make_document()
    get_text = _make_text_generator(json.dumps(data))
    chunker = JsonListChunker(document, get_text, max_chunk_size=512)

    with pytest.raises(ValueError, match="expects JSON list or dict"):
        await _collect(chunker)


@pytest.mark.asyncio
async def test_deeply_nested_list():
    """Deeply nested dict structure is handled via json_path."""
    data = {
        "level1": {
            "level2": {
                "level3": [
                    {"id": 1},
                    {"id": 2},
                ]
            }
        }
    }
    document = _make_document()
    get_text = _make_text_generator(json.dumps(data))
    chunker = JsonListChunker(
        document, get_text, max_chunk_size=512, json_path="level1.level2.level3"
    )

    chunks = await _collect(chunker)

    assert len(chunks) == 2
    assert chunks[0].metadata["json_path"] == "level1.level2.level3[0]"
    assert chunks[1].metadata["json_path"] == "level1.level2.level3[1]"


@pytest.mark.asyncio
async def test_chunk_ids_deterministic():
    """Same document + same index produces the same chunk ID."""
    data = [{"id": 1}, {"id": 2}]
    doc_id = uuid4()

    document1 = Document(
        id=doc_id,
        name="test.json",
        raw_data_location="/test/path/test.json",
        external_metadata=None,
        mime_type="application/json",
    )
    chunker1 = JsonListChunker(
        document1, _make_text_generator(json.dumps(data)), max_chunk_size=512
    )
    chunks1 = await _collect(chunker1)

    document2 = Document(
        id=doc_id,
        name="test.json",
        raw_data_location="/test/path/test.json",
        external_metadata=None,
        mime_type="application/json",
    )
    chunker2 = JsonListChunker(
        document2, _make_text_generator(json.dumps(data)), max_chunk_size=512
    )
    chunks2 = await _collect(chunker2)

    assert chunks1[0].id == chunks2[0].id
    assert chunks1[1].id == chunks2[1].id
