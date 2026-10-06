"""Hybrid retrieval reads the DLT graph: rows in the chunk lane, the DLT node
types in the entity lane, each merged by score into one ranked list."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError
from cognee.modules.retrieval.hybrid.chunks import (
    CHUNK_COLLECTIONS,
    merge_scored,
    retrieve_hybrid_chunks,
)
from cognee.modules.retrieval.hybrid.entities import ENTITY_COLLECTIONS, search_entities


def _hit(result_id, score, **payload):
    return SimpleNamespace(
        id=result_id, score=score, payload={"id": result_id, "text": result_id, **payload}
    )


def _vector_engine(hits_by_collection):
    async def search(collection_name, *_args, **_kwargs):
        if collection_name not in hits_by_collection:
            raise CollectionNotFoundError(f"{collection_name} missing")
        return hits_by_collection[collection_name]

    return SimpleNamespace(search=AsyncMock(side_effect=search))


def test_merge_scored_ranks_across_channels_dedupes_and_cuts():
    merged = merge_scored(
        [
            _hit("a", 0.3),
            _hit("b", 0.1),
            _hit("a", 0.2),
            _hit("c", 0.5),
            SimpleNamespace(payload={"id": "d"}),
        ],
        limit=3,
    )
    assert [hit.payload["id"] for hit in merged] == ["b", "a", "c"]


@pytest.mark.asyncio
async def test_chunk_lane_merges_dlt_rows_with_document_chunks_by_score():
    engine = _vector_engine(
        {
            "DocumentChunk_text": [_hit("chunk_far", 0.6), _hit("chunk_near", 0.1)],
            "DltRow_text": [_hit("row_mid", 0.3, type="DltRow")],
            "TextSummary_text": [],
        }
    )

    result = await retrieve_hybrid_chunks(engine, "orders in march", 3, None, None, "OR", False)

    assert [hit.payload["id"] for hit in result["chunks"]] == ["chunk_near", "row_mid", "chunk_far"]
    searched = [call.args[0] for call in engine.search.await_args_list]
    assert set(searched) == set(CHUNK_COLLECTIONS) | {"TextSummary_text"}


@pytest.mark.asyncio
async def test_chunk_lane_without_dlt_data_is_unchanged():
    engine = _vector_engine({"DocumentChunk_text": [_hit("chunk", 0.2)], "TextSummary_text": []})
    result = await retrieve_hybrid_chunks(engine, "q", 2, None, None, "OR", False)
    assert [hit.payload["id"] for hit in result["chunks"]] == ["chunk"]


@pytest.mark.asyncio
async def test_entity_lane_covers_the_dlt_node_types():
    engine = _vector_engine(
        {
            "Entity_name": [_hit("marie", 0.4)],
            "DltColumn_properties": [_hit("orders:status:active", 0.2)],
            "SchemaTable_name": [_hit("orders", 0.3)],
        }
    )

    hits = await search_entities(engine, "orders", 2, None, "OR", [0.0])

    assert [hit.payload["id"] for hit in hits] == ["orders:status:active", "orders"]
    assert {call.args[0] for call in engine.search.await_args_list} == set(ENTITY_COLLECTIONS)
