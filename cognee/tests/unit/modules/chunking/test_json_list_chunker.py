import json
from uuid import NAMESPACE_OID, uuid4, uuid5

import pytest

from cognee.modules.chunking.JsonListChunker import JsonListChunker
from cognee.modules.data.processing.document_types import TextDocument
from cognee.modules.data.processing.document_types.Document import Document


def make_document() -> Document:
    return Document(
        id=uuid4(),
        name="test_document",
        raw_data_location="/test/path",
        external_metadata=None,
        mime_type="application/json",
    )


def make_text_generator(*texts: str):
    async def get_text():
        for text in texts:
            yield text

    return get_text


async def collect_chunks(chunker: JsonListChunker):
    return [chunk async for chunk in chunker.read()]


@pytest.mark.asyncio
async def test_flat_list_preserves_text_and_legacy_ids():
    document = make_document()
    chunker = JsonListChunker(
        document,
        make_text_generator(json.dumps([{"name": "Alice"}, {"name": "Bob"}])),
        max_chunk_size=512,
    )

    chunks = await collect_chunks(chunker)

    assert [chunk.text for chunk in chunks] == ["{'name': 'Alice'}", "{'name': 'Bob'}"]
    assert [chunk.id for chunk in chunks] == [
        uuid5(NAMESPACE_OID, f"{document.id}-0"),
        uuid5(NAMESPACE_OID, f"{document.id}-1"),
    ]
    assert [chunk.metadata["json_path"] for chunk in chunks] == ["[0]", "[1]"]
    assert "json_context" not in chunks[0].metadata


@pytest.mark.asyncio
async def test_single_nested_array_includes_ancestor_context_in_indexed_text():
    data = {
        "tenant": "ACME",
        "records": {"batch_id": "B01", "items": [{"value": 123}]},
    }
    chunker = JsonListChunker(make_document(), make_text_generator(json.dumps(data)), 512)

    [chunk] = await collect_chunks(chunker)

    assert chunk.metadata["json_path"] == "records.items[0]"
    assert chunk.metadata["json_array_path"] == "records.items"
    assert chunk.metadata["json_context"] == {"tenant": "ACME", "batch_id": "B01"}
    assert json.loads(chunk.text) == {
        "_context": {"tenant": "ACME", "batch_id": "B01"},
        "_json_path": "records.items",
        "_record": {"value": 123},
    }


@pytest.mark.asyncio
async def test_auto_detection_traverses_lists_and_normalizes_repeated_leaf_paths():
    data = {
        "tenant": "ACME",
        "records": [
            {"source": "sensor-A", "items": [{"value": 10}]},
            {"source": "sensor-B", "items": [{"value": 20}]},
        ],
    }
    chunker = JsonListChunker(make_document(), make_text_generator(json.dumps(data)), 512)

    chunks = await collect_chunks(chunker)

    assert [chunk.metadata["json_path"] for chunk in chunks] == [
        "records[0].items[0]",
        "records[1].items[0]",
    ]
    assert [chunk.metadata["json_array_path"] for chunk in chunks] == [
        "records[*].items",
        "records[*].items",
    ]
    assert [json.loads(chunk.text)["_context"] for chunk in chunks] == [
        {"tenant": "ACME", "source": "sensor-A"},
        {"tenant": "ACME", "source": "sensor-B"},
    ]
    assert chunks[0].id != chunks[1].id


@pytest.mark.asyncio
async def test_auto_detection_requires_a_selector_for_distinct_leaf_arrays():
    data = {"users": [{"name": "Alice"}], "orders": [{"id": "order-1"}]}
    chunker = JsonListChunker(make_document(), make_text_generator(json.dumps(data)), 512)

    with pytest.raises(ValueError, match="'users', 'orders'"):
        await collect_chunks(chunker)


@pytest.mark.asyncio
async def test_nested_array_indexes_its_structural_path_without_scalar_context():
    data = {"records": {"items": [{"value": 123}]}}
    chunker = JsonListChunker(make_document(), make_text_generator(json.dumps(data)), 512)

    [chunk] = await collect_chunks(chunker)

    assert json.loads(chunk.text) == {
        "_context": {},
        "_json_path": "records.items",
        "_record": {"value": 123},
    }


@pytest.mark.asyncio
async def test_explicit_path_can_select_a_container_array():
    data = {"tenant": "ACME", "records": [{"items": [{"value": 10}]}]}
    chunker = JsonListChunker(
        make_document(), make_text_generator(json.dumps(data)), 512, json_path="records"
    )

    [chunk] = await collect_chunks(chunker)

    assert chunk.metadata["json_path"] == "records[0]"
    assert chunk.metadata["json_array_path"] == "records"
    assert json.loads(chunk.text) == {
        "_context": {"tenant": "ACME"},
        "_json_path": "records",
        "_record": {"items": [{"value": 10}]},
    }


@pytest.mark.asyncio
async def test_selector_factory_works_through_document_read(tmp_path):
    data_file = tmp_path / "records.json"
    data_file.write_text(
        json.dumps(
            {
                "tenant": "ACME",
                "records": [
                    {"items": [{"id": 1}]},
                    {"items": [{"id": 2}]},
                ],
            }
        ),
        encoding="utf-8",
    )
    document = TextDocument(
        id=uuid4(),
        name="records",
        raw_data_location=str(data_file),
        external_metadata=None,
        mime_type="application/json",
    )

    chunks = [
        chunk
        async for chunk in document.read(
            chunker_cls=JsonListChunker.with_json_path("records[*].items"), max_chunk_size=512
        )
    ]

    assert [json.loads(chunk.text)["_record"] for chunk in chunks] == [{"id": 1}, {"id": 2}]


@pytest.mark.asyncio
async def test_invalid_selector_is_rejected():
    chunker = JsonListChunker(
        make_document(),
        make_text_generator(json.dumps({"records": [{"id": 1}]})),
        512,
        json_path="records[*].items",
    )

    with pytest.raises(ValueError, match="does not point to a JSON list"):
        await collect_chunks(chunker)


def test_selector_factory_rejects_invalid_path_syntax():
    with pytest.raises(ValueError, match="dotted keys"):
        JsonListChunker.with_json_path("records..items")
