"""Hybrid retrieval reads the DLT graph when the dataset has one: rows join the
chunk lane, the DLT node types join the entity lane, each merged by score into
one ranked list. One ``DltRow_text`` existence check per search decides it, so
a dataset without relational rows searches exactly the document collections."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError
from cognee.modules.retrieval.hybrid.chunks import (
    CHUNK_COLLECTIONS,
    DLT_CHUNK_COLLECTIONS,
    merge_scored,
    retrieve_hybrid_chunks,
)
from cognee.modules.retrieval.hybrid.entities import (
    DLT_ENTITY_COLLECTIONS,
    ENTITY_COLLECTIONS,
    search_entities,
)
from cognee.modules.retrieval.hybrid_retriever import HybridRetriever


def _hit(result_id, score, **payload):
    return SimpleNamespace(
        id=result_id, score=score, payload={"id": result_id, "text": result_id, **payload}
    )


def _vector_engine(hits_by_collection, *, dlt_rows=None):
    async def search(collection_name, *_args, **_kwargs):
        if collection_name not in hits_by_collection:
            raise CollectionNotFoundError(f"{collection_name} missing")
        return hits_by_collection[collection_name]

    has_dlt = "DltRow_text" in hits_by_collection if dlt_rows is None else dlt_rows
    engine = SimpleNamespace(
        search=AsyncMock(side_effect=search),
        has_collection=AsyncMock(side_effect=lambda name: name == "DltRow_text" and has_dlt),
    )
    engine.embedding_engine = SimpleNamespace(embed_text=AsyncMock(return_value=[[0.0]]))
    return engine


def _unified(vector):
    return SimpleNamespace(
        vector=vector,
        graph=SimpleNamespace(
            is_empty=AsyncMock(return_value=False),
            get_neighborhood=AsyncMock(return_value=([], [])),
        ),
    )


def _searched(engine):
    return {call.args[0] for call in engine.search.await_args_list}


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

    result = await retrieve_hybrid_chunks(
        engine,
        "orders in march",
        3,
        None,
        None,
        "OR",
        False,
        collections=CHUNK_COLLECTIONS + DLT_CHUNK_COLLECTIONS,
    )

    assert [hit.payload["id"] for hit in result["chunks"]] == ["chunk_near", "row_mid", "chunk_far"]
    assert _searched(engine) == {"DocumentChunk_text", "DltRow_text", "TextSummary_text"}


@pytest.mark.asyncio
async def test_entity_lane_covers_the_dlt_node_types():
    engine = _vector_engine(
        {
            "Entity_name": [_hit("marie", 0.4)],
            "DltColumn_properties": [_hit("orders:status:active", 0.2)],
            "SchemaTable_name": [_hit("orders", 0.3)],
        }
    )

    hits = await search_entities(
        engine,
        "orders",
        2,
        None,
        "OR",
        [0.0],
        collections=ENTITY_COLLECTIONS + DLT_ENTITY_COLLECTIONS,
    )

    assert [hit.payload["id"] for hit in hits] == ["orders:status:active", "orders"]
    assert _searched(engine) == set(ENTITY_COLLECTIONS) | set(DLT_ENTITY_COLLECTIONS)


async def _fetch(engine):
    retriever = HybridRetriever(chunks_top_k=3, entities_top_k=3)
    with (
        patch(
            "cognee.modules.retrieval.hybrid_retriever.get_unified_engine",
            new_callable=AsyncMock,
            return_value=_unified(engine),
        ),
        patch(
            "cognee.modules.retrieval.hybrid_retriever.load_preference_weights",
            AsyncMock(return_value={}),
        ),
    ):
        return await retriever.get_retrieved_objects(query="orders in march")


@pytest.mark.asyncio
async def test_a_dataset_without_dlt_rows_searches_only_the_document_collections():
    """The gate: no DltRow_text table, so neither lane touches a DLT collection."""
    engine = _vector_engine(
        {"DocumentChunk_text": [_hit("chunk", 0.2)], "TextSummary_text": [], "Entity_name": []}
    )

    result = await _fetch(engine)

    assert [hit.payload["id"] for hit in result["chunks"]] == ["chunk"]
    engine.has_collection.assert_awaited_once_with("DltRow_text")
    assert _searched(engine) == {
        "DocumentChunk_text",
        "TextSummary_text",
        "Entity_name",
        "EdgeType_relationship_name",
    }


@pytest.mark.asyncio
async def test_a_dataset_with_dlt_rows_searches_the_dlt_collections_in_both_lanes():
    engine = _vector_engine(
        {
            "DocumentChunk_text": [_hit("chunk", 0.5)],
            "DltRow_text": [_hit("row", 0.1, type="DltRow")],
            "TextSummary_text": [],
            "Entity_name": [],
            "SchemaTable_name": [_hit("orders", 0.3)],
        }
    )

    result = await _fetch(engine)

    assert [hit.payload["id"] for hit in result["chunks"]] == ["row", "chunk"]
    engine.has_collection.assert_awaited_once_with("DltRow_text")
    assert _searched(engine) >= {"DltRow_text", *DLT_ENTITY_COLLECTIONS}
