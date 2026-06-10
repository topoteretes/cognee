import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError
from cognee.modules.retrieval.exceptions.exceptions import NoDataError, QueryValidationError
from cognee.modules.retrieval.hybrid_retriever import HybridRetriever


def _result(result_id="result-id", payload=None):
    scored_result = MagicMock()
    scored_result.id = result_id
    scored_result.payload = payload
    return scored_result


def _unified(vector=None, graph=None):
    unified = MagicMock()
    unified.vector = vector or MagicMock()
    unified.graph = graph or MagicMock()
    return unified


def _vector_search(chunks=None, entities=None, missing_collections=None):
    missing_collections = set(missing_collections or [])

    async def search(collection_name, *args, **kwargs):
        if collection_name in missing_collections:
            raise CollectionNotFoundError("missing")
        if collection_name == "DocumentChunk_text":
            return chunks or []
        if collection_name == "Entity_name":
            return entities or []
        return []

    return AsyncMock(side_effect=search)


def _search_call(vector, collection_name):
    for call in vector.search.await_args_list:
        if call.args[:1] == (collection_name,):
            return call
    raise AssertionError(f"{collection_name} was not searched")


@pytest.mark.asyncio
async def test_passage_section_formatting():
    retriever = HybridRetriever()
    context = await retriever.get_context_from_objects(
        query="q",
        retrieved_objects={
            "chunks": [
                _result(payload={"text": "First passage"}),
                _result(payload={}),
                _result(payload={"text": "Second passage"}),
            ],
            "entities": [],
        },
    )

    assert context == "## Relevant passages\nFirst passage\n---\nSecond passage"


@pytest.mark.asyncio
async def test_empty_sections_return_empty_context():
    retriever = HybridRetriever()

    context = await retriever.get_context_from_objects(
        query="q", retrieved_objects={"chunks": [], "entities": []}
    )

    assert context == ""


@pytest.mark.asyncio
async def test_empty_graph_does_not_prevent_chunk_search():
    vector = MagicMock()
    vector.search = _vector_search(
        chunks=[_result("chunk-1", {"id": "chunk-1", "text": "Chunk text"})],
        entities=[_result("entity-1", {"id": "entity-1", "name": "Entity"})],
    )
    graph = MagicMock()
    graph.is_empty = AsyncMock(return_value=True)
    graph.get_connections = AsyncMock()

    retriever = HybridRetriever()

    with patch(
        "cognee.modules.retrieval.hybrid_retriever.get_unified_engine",
        new_callable=AsyncMock,
        return_value=_unified(vector=vector, graph=graph),
    ):
        retrieved = await retriever.get_retrieved_objects(query="q")

    assert retrieved["chunks"][0].payload["text"] == "Chunk text"
    assert retrieved["entities"][0]["name"] == "Entity"
    graph.get_connections.assert_not_awaited()
    assert _search_call(vector, "DocumentChunk_text").args[:2] == ("DocumentChunk_text", "q")


@pytest.mark.asyncio
async def test_query_batch_is_rejected_before_work_starts():
    retriever = HybridRetriever()

    with pytest.raises(QueryValidationError, match="HYBRID_COMPLETION"):
        await retriever.get_retrieved_objects(query_batch=["q"])

    with pytest.raises(QueryValidationError, match="HYBRID_COMPLETION"):
        await retriever.get_context_from_objects(query_batch=["q"])

    with pytest.raises(QueryValidationError, match="HYBRID_COMPLETION"):
        await retriever.get_completion_from_context(query_batch=["q"])


@pytest.mark.asyncio
async def test_missing_document_chunk_collection_raises_no_data_error():
    vector = MagicMock()
    vector.search = _vector_search(missing_collections={"DocumentChunk_text"})
    graph = MagicMock()

    retriever = HybridRetriever()

    with patch(
        "cognee.modules.retrieval.hybrid_retriever.get_unified_engine",
        new_callable=AsyncMock,
        return_value=_unified(vector=vector, graph=graph),
    ):
        with pytest.raises(NoDataError, match="No data found"):
            await retriever.get_retrieved_objects(query="q")


@pytest.mark.asyncio
async def test_chunk_search_receives_nodeset_filters():
    vector = MagicMock()
    vector.search = AsyncMock(return_value=[])
    retriever = HybridRetriever(node_name=["KEN"], node_name_filter_operator="AND")

    with patch(
        "cognee.modules.retrieval.hybrid_retriever.get_unified_engine",
        new_callable=AsyncMock,
        return_value=_unified(vector=vector),
    ):
        await retriever.get_retrieved_objects(query="q")

    chunk_call = _search_call(vector, "DocumentChunk_text")
    assert chunk_call.args[:2] == ("DocumentChunk_text", "q")
    assert chunk_call.kwargs["node_name"] == ["KEN"]
    assert chunk_call.kwargs["node_name_filter_operator"] == "AND"


@pytest.mark.asyncio
async def test_chunk_retrieval_uses_bm25_first_then_vector_fill_with_dedupe():
    bm25_retriever = MagicMock()
    bm25_retriever.get_retrieved_objects = AsyncMock(
        return_value=[
            ({"id": "bm25-1", "text": "BM25 first"}, 2.0),
            ({"id": "shared", "text": "BM25 shared"}, 1.5),
        ]
    )
    vector = MagicMock()
    vector.search = _vector_search(
        chunks=[
            _result("shared-vector", {"id": "shared", "text": "Vector duplicate"}),
            _result("semantic-1", {"id": "semantic-1", "text": "Semantic extra"}),
            _result("semantic-2", {"id": "semantic-2", "text": "Second semantic extra"}),
        ]
    )
    retriever = HybridRetriever(chunks_top_k=4)

    with (
        patch(
            "cognee.modules.retrieval.hybrid_retriever.get_unified_engine",
            new_callable=AsyncMock,
            return_value=_unified(vector=vector),
        ),
        patch(
            "cognee.modules.retrieval.hybrid_retriever.BM25ChunksRetriever",
            return_value=bm25_retriever,
        ) as bm25_cls,
    ):
        retrieved = await retriever.get_retrieved_objects(query="q")

    bm25_cls.assert_called_once_with(top_k=2, with_scores=True)
    bm25_retriever.get_retrieved_objects.assert_awaited_once_with("q")
    assert [_payload_text(chunk) for chunk in retrieved["chunks"]] == [
        "BM25 first",
        "BM25 shared",
        "Semantic extra",
        "Second semantic extra",
    ]


@pytest.mark.asyncio
async def test_bm25_chunks_respect_nodeset_filter_before_merge():
    bm25_retriever = MagicMock()
    bm25_retriever.get_retrieved_objects = AsyncMock(
        return_value=[
            ({"id": "keep", "text": "Keep", "belongs_to_set": ["KEN", "src_type:figure"]}, 2.0),
            ({"id": "drop", "text": "Drop", "belongs_to_set": ["KEN"]}, 1.0),
        ]
    )
    vector = MagicMock()
    vector.search = AsyncMock(return_value=[])
    retriever = HybridRetriever(
        chunks_top_k=4,
        node_name=["KEN", "src_type:figure"],
        node_name_filter_operator="AND",
    )

    with (
        patch(
            "cognee.modules.retrieval.hybrid_retriever.get_unified_engine",
            new_callable=AsyncMock,
            return_value=_unified(vector=vector),
        ),
        patch(
            "cognee.modules.retrieval.hybrid_retriever.BM25ChunksRetriever",
            return_value=bm25_retriever,
        ),
    ):
        retrieved = await retriever.get_retrieved_objects(query="q")

    assert [_payload_text(chunk) for chunk in retrieved["chunks"]] == ["Keep"]


@pytest.mark.asyncio
async def test_zero_score_bm25_chunks_do_not_reserve_context_slots():
    bm25_retriever = MagicMock()
    bm25_retriever.get_retrieved_objects = AsyncMock(
        return_value=[
            ({"id": "zero", "text": "Zero lexical score"}, 0.0),
            ({"id": "positive", "text": "Positive lexical score"}, 3.0),
        ]
    )
    vector = MagicMock()
    vector.search = _vector_search(
        chunks=[_result("semantic", {"id": "semantic", "text": "Semantic fallback"})]
    )
    retriever = HybridRetriever(chunks_top_k=2)

    with (
        patch(
            "cognee.modules.retrieval.hybrid_retriever.get_unified_engine",
            new_callable=AsyncMock,
            return_value=_unified(vector=vector),
        ),
        patch(
            "cognee.modules.retrieval.hybrid_retriever.BM25ChunksRetriever",
            return_value=bm25_retriever,
        ),
    ):
        retrieved = await retriever.get_retrieved_objects(query="q")

    assert [_payload_text(chunk) for chunk in retrieved["chunks"]] == [
        "Positive lexical score",
        "Semantic fallback",
    ]


@pytest.mark.asyncio
async def test_independent_retrieval_channels_run_concurrently():
    bm25_started = asyncio.Event()
    chunk_vector_started = asyncio.Event()
    entity_started = asyncio.Event()

    bm25_retriever = MagicMock()

    async def search_bm25(query):
        bm25_started.set()
        await chunk_vector_started.wait()
        return []

    async def search_vector(collection_name, *args, **kwargs):
        if collection_name == "DocumentChunk_text":
            chunk_vector_started.set()
            await bm25_started.wait()
            await entity_started.wait()
            return []
        if collection_name == "Entity_name":
            entity_started.set()
            await chunk_vector_started.wait()
            return []
        return []

    bm25_retriever.get_retrieved_objects = AsyncMock(side_effect=search_bm25)
    vector = MagicMock()
    vector.search = AsyncMock(side_effect=search_vector)
    retriever = HybridRetriever()

    with (
        patch(
            "cognee.modules.retrieval.hybrid_retriever.get_unified_engine",
            new_callable=AsyncMock,
            return_value=_unified(vector=vector),
        ),
        patch(
            "cognee.modules.retrieval.hybrid_retriever.BM25ChunksRetriever",
            return_value=bm25_retriever,
        ),
    ):
        retrieved = await asyncio.wait_for(retriever.get_retrieved_objects(query="q"), timeout=1)

    assert retrieved == {"chunks": [], "entities": []}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "expected_name"),
    [
        ({"id": "entity-1", "name": "Named entity"}, "Named entity"),
        ({"id": "entity-1", "text": "Text entity"}, "Text entity"),
        ({"id": "entity-1"}, "entity-1"),
    ],
)
async def test_entity_fields_fall_back_from_name_to_text_to_id(payload, expected_name):
    vector = MagicMock()
    vector.search = _vector_search(entities=[_result("fallback-id", payload)])
    graph = MagicMock()
    graph.is_empty = AsyncMock(return_value=True)
    retriever = HybridRetriever()

    with patch(
        "cognee.modules.retrieval.hybrid_retriever.get_unified_engine",
        new_callable=AsyncMock,
        return_value=_unified(vector=vector, graph=graph),
    ):
        retrieved = await retriever.get_retrieved_objects(query="q")

    assert retrieved["entities"][0]["name"] == expected_name


@pytest.mark.asyncio
async def test_entity_header_omits_index_schema_type():
    retriever = HybridRetriever()

    context = await retriever.get_context_from_objects(
        query="q",
        retrieved_objects={
            "chunks": [],
            "entities": [
                {
                    "id": "entity-1",
                    "name": "lisbon office logistics intelligence project",
                    "type": "IndexSchema",
                    "edges": [],
                }
            ],
        },
    )

    assert context == "## Relevant entities\n### lisbon office logistics intelligence project"


@pytest.mark.asyncio
async def test_entity_type_prefers_domain_type_over_index_schema():
    vector = MagicMock()
    vector.search = _vector_search(
        entities=[
            _result(
                "entity-1",
                {
                    "id": "entity-1",
                    "name": "Lisbon office",
                    "type": "IndexSchema",
                    "is_a": "Office",
                },
            )
        ]
    )
    graph = MagicMock()
    graph.is_empty = AsyncMock(return_value=True)
    retriever = HybridRetriever()

    with patch(
        "cognee.modules.retrieval.hybrid_retriever.get_unified_engine",
        new_callable=AsyncMock,
        return_value=_unified(vector=vector, graph=graph),
    ):
        retrieved = await retriever.get_retrieved_objects(query="q")

    assert retrieved["entities"][0]["type"] == "Office"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("edge", "expected_text"),
    [
        ({"edge_text": "Direct text", "relationship_name": "REL"}, "Direct text"),
        (
            {"properties": {"edge_text": "Nested text"}, "relationship_name": "REL"},
            "Nested text",
        ),
        ({"relationship_name": "REL"}, "Source -- REL -- Target"),
    ],
)
async def test_edge_text_fallbacks(edge, expected_text):
    vector = MagicMock()
    vector.search = _vector_search(
        entities=[_result("entity-1", {"id": "entity-1", "name": "Entity"})]
    )
    graph = MagicMock()
    graph.is_empty = AsyncMock(return_value=False)
    graph.get_connections = AsyncMock(
        return_value=[
            (
                {"id": "source-1", "name": "Source"},
                edge,
                {"id": "target-1", "name": "Target"},
            )
        ]
    )
    retriever = HybridRetriever()

    with patch(
        "cognee.modules.retrieval.hybrid_retriever.get_unified_engine",
        new_callable=AsyncMock,
        return_value=_unified(vector=vector, graph=graph),
    ):
        retrieved = await retriever.get_retrieved_objects(query="q")

    assert retrieved["entities"][0]["edges"][0]["text"] == expected_text


@pytest.mark.asyncio
async def test_duplicate_edges_are_removed_and_max_edges_caps_results():
    vector = MagicMock()
    vector.search = _vector_search(
        entities=[_result("entity-1", {"id": "entity-1", "name": "Entity"})]
    )
    graph = MagicMock()
    graph.is_empty = AsyncMock(return_value=False)
    graph.get_connections = AsyncMock(
        return_value=[
            ({"id": "s1", "name": "S1"}, {"edge_text": "same"}, {"id": "t1", "name": "T1"}),
            ({"id": "s2", "name": "S2"}, {"edge_text": "same"}, {"id": "t2", "name": "T2"}),
            ({"id": "s3", "name": "S3"}, {"edge_text": "other"}, {"id": "t3", "name": "T3"}),
        ]
    )
    retriever = HybridRetriever(max_edges_per_entity=1)

    with patch(
        "cognee.modules.retrieval.hybrid_retriever.get_unified_engine",
        new_callable=AsyncMock,
        return_value=_unified(vector=vector, graph=graph),
    ):
        retrieved = await retriever.get_retrieved_objects(query="q")

    assert [edge["text"] for edge in retrieved["entities"][0]["edges"]] == ["same"]


@pytest.mark.asyncio
async def test_same_edge_text_does_not_collapse_distinct_relationships():
    vector = MagicMock()
    vector.search = _vector_search(
        entities=[_result("entity-1", {"id": "entity-1", "name": "Entity"})]
    )
    graph = MagicMock()
    graph.is_empty = AsyncMock(return_value=False)
    graph.get_connections = AsyncMock(
        return_value=[
            (
                {"id": "s1", "name": "S1"},
                {"edge_text": "related", "relationship_name": "REL"},
                {"id": "t1", "name": "T1"},
            ),
            (
                {"id": "s2", "name": "S2"},
                {"edge_text": "related", "relationship_name": "REL"},
                {"id": "t2", "name": "T2"},
            ),
            (
                {"id": "s1", "name": "S1"},
                {"edge_text": "related", "relationship_name": "REL"},
                {"id": "t1", "name": "T1"},
            ),
        ]
    )
    retriever = HybridRetriever(max_edges_per_entity=5)

    with patch(
        "cognee.modules.retrieval.hybrid_retriever.get_unified_engine",
        new_callable=AsyncMock,
        return_value=_unified(vector=vector, graph=graph),
    ):
        retrieved = await retriever.get_retrieved_objects(query="q")

    assert [edge["source_id"] for edge in retrieved["entities"][0]["edges"]] == ["s1", "s2"]


@pytest.mark.asyncio
async def test_is_a_edge_is_prioritized_before_edge_cap():
    vector = MagicMock()
    vector.search = _vector_search(
        entities=[_result("entity-1", {"id": "entity-1", "name": "Entity"})]
    )
    graph = MagicMock()
    graph.is_empty = AsyncMock(return_value=False)
    graph.get_connections = AsyncMock(
        return_value=[
            (
                {"id": "entity-1", "name": "Lisbon office"},
                {"edge_text": "Lisbon office owns HarborLens", "relationship_name": "owns"},
                {"id": "project-1", "name": "HarborLens"},
            ),
            (
                {"id": "entity-1", "name": "Lisbon office"},
                {"edge_text": "Lisbon office is a Office", "relationship_name": "is_a"},
                {"id": "type-1", "name": "Office"},
            ),
        ]
    )
    retriever = HybridRetriever(max_edges_per_entity=1)

    with patch(
        "cognee.modules.retrieval.hybrid_retriever.get_unified_engine",
        new_callable=AsyncMock,
        return_value=_unified(vector=vector, graph=graph),
    ):
        retrieved = await retriever.get_retrieved_objects(query="q")

    assert [edge["text"] for edge in retrieved["entities"][0]["edges"]] == [
        "Lisbon office is a Office"
    ]


@pytest.mark.asyncio
async def test_missing_entity_collection_returns_empty_channel():
    vector = MagicMock()
    vector.search = _vector_search(missing_collections={"Entity_name"})
    retriever = HybridRetriever()

    with patch(
        "cognee.modules.retrieval.hybrid_retriever.get_unified_engine",
        new_callable=AsyncMock,
        return_value=_unified(vector=vector),
    ):
        retrieved = await retriever.get_retrieved_objects(query="q")

    assert retrieved == {"chunks": [], "entities": []}


@pytest.mark.asyncio
async def test_entity_search_receives_nodeset_filters_and_expands_connections():
    vector = MagicMock()
    vector.search = _vector_search(
        entities=[_result("entity-1", {"id": "entity-1", "name": "Entity"})]
    )
    graph = MagicMock()
    graph.is_empty = AsyncMock(return_value=False)
    graph.get_connections = AsyncMock(return_value=[])
    retriever = HybridRetriever(node_name=["KEN"], node_name_filter_operator="AND")

    with patch(
        "cognee.modules.retrieval.hybrid_retriever.get_unified_engine",
        new_callable=AsyncMock,
        return_value=_unified(vector=vector, graph=graph),
    ):
        await retriever.get_retrieved_objects(query="q")

    entity_call = _search_call(vector, "Entity_name")
    assert entity_call.args[:2] == ("Entity_name", "q")
    assert entity_call.kwargs["node_name"] == ["KEN"]
    assert entity_call.kwargs["node_name_filter_operator"] == "AND"
    graph.get_connections.assert_awaited_once_with("entity-1")


@pytest.mark.asyncio
async def test_malformed_connection_rows_are_skipped_without_dropping_entity():
    vector = MagicMock()
    vector.search = _vector_search(
        entities=[_result("entity-1", {"id": "entity-1", "name": "Entity"})]
    )
    graph = MagicMock()
    graph.is_empty = AsyncMock(return_value=False)
    graph.get_connections = AsyncMock(
        return_value=[
            {"not": "a tuple"},
            ({"id": "source", "name": "Source"}, {"relationship_name": "REL"}, {"id": "target"}),
        ]
    )
    retriever = HybridRetriever()

    with patch(
        "cognee.modules.retrieval.hybrid_retriever.get_unified_engine",
        new_callable=AsyncMock,
        return_value=_unified(vector=vector, graph=graph),
    ):
        retrieved = await retriever.get_retrieved_objects(query="q")

    assert retrieved["entities"][0]["name"] == "Entity"
    assert retrieved["entities"][0]["edges"][0]["text"] == "Source -- REL -- target"


@pytest.mark.asyncio
async def test_entities_section_omits_missing_optional_fields():
    retriever = HybridRetriever()

    context = await retriever.get_context_from_objects(
        query="q",
        retrieved_objects={
            "chunks": [],
            "entities": [{"id": "entity-1", "name": "Entity", "edges": []}],
        },
    )

    assert context == "## Relevant entities\n### Entity"


@pytest.mark.asyncio
async def test_global_context_is_omitted_by_default():
    retriever = HybridRetriever()

    with patch(
        "cognee.modules.retrieval.hybrid_retriever.load_root_text",
        new_callable=AsyncMock,
    ) as load_root_text:
        context = await retriever.get_context_from_objects(
            query="q", retrieved_objects={"chunks": [], "entities": []}
        )

    assert context == ""
    load_root_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_global_context_is_prepended_when_enabled():
    vector = MagicMock()
    retriever = HybridRetriever(include_global_context_index=True)

    with (
        patch(
            "cognee.modules.retrieval.hybrid_retriever.get_unified_engine",
            new_callable=AsyncMock,
            return_value=_unified(vector=vector),
        ) as get_unified,
        patch(
            "cognee.modules.retrieval.hybrid_retriever.load_root_text",
            new_callable=AsyncMock,
            return_value="Root summary",
        ),
        patch(
            "cognee.modules.retrieval.hybrid_retriever.search_top_global_context_summaries",
            new_callable=AsyncMock,
            return_value=["Area summary"],
        ) as search_summaries,
    ):
        context = await retriever.get_context_from_objects(
            query="q",
            retrieved_objects={"chunks": [_result(payload={"text": "Chunk"})], "entities": []},
        )

    assert context.startswith("## Global context\nWorld summary:\nRoot summary")
    assert "\n\n## Relevant passages\nChunk" in context
    get_unified.assert_awaited_once()
    search_summaries.assert_awaited_once_with("q", 3, vector)


@pytest.mark.asyncio
async def test_no_session_path_calls_generate_completion():
    retriever = HybridRetriever()

    with (
        patch.object(retriever, "_use_session_cache", return_value=False),
        patch(
            "cognee.modules.retrieval.hybrid_retriever.generate_completion",
            new_callable=AsyncMock,
            return_value="answer",
        ) as generate_completion,
    ):
        completion = await retriever.get_completion_from_context(
            query="q",
            retrieved_objects={"chunks": [], "entities": []},
            context="context",
        )

    assert completion == ["answer"]
    generate_completion.assert_awaited_once()


@pytest.mark.asyncio
async def test_session_path_calls_session_manager_with_used_node_ids():
    retriever = HybridRetriever(session_id="session-1")
    session_manager = MagicMock()
    session_manager.generate_completion_with_session = AsyncMock(return_value="answer")
    retrieved_objects = {
        "chunks": [_result("chunk-result", {"id": "chunk-1", "text": "Chunk"})],
        "entities": [
            {
                "id": "entity-1",
                "name": "Entity",
                "edges": [
                    {
                        "text": "edge",
                        "source_id": "source-1",
                        "target_id": "target-1",
                    }
                ],
            }
        ],
    }

    with (
        patch.object(retriever, "_use_session_cache", return_value=True),
        patch(
            "cognee.modules.retrieval.hybrid_retriever.get_session_manager",
            return_value=session_manager,
        ),
    ):
        completion = await retriever.get_completion_from_context(
            query="q",
            retrieved_objects=retrieved_objects,
            context="context",
        )

    assert completion == ["answer"]
    call_kwargs = session_manager.generate_completion_with_session.call_args.kwargs
    assert call_kwargs["session_id"] == "session-1"
    assert call_kwargs["used_graph_element_ids"] == {
        "node_ids": ["chunk-1", "entity-1", "source-1", "target-1"]
    }


@pytest.mark.asyncio
async def test_context_object_id_extraction_skips_non_dict_entities():
    retriever = HybridRetriever(session_id="session-1")
    session_manager = MagicMock()
    session_manager.generate_completion_with_session = AsyncMock(return_value="answer")

    with (
        patch.object(retriever, "_use_session_cache", return_value=True),
        patch(
            "cognee.modules.retrieval.hybrid_retriever.get_session_manager",
            return_value=session_manager,
        ),
    ):
        await retriever.get_completion_from_context(
            query="q",
            retrieved_objects={
                "chunks": [_result("chunk-result", {"id": "chunk-1", "text": "Chunk"})],
                "entities": ["not-a-dict"],
            },
            context="context",
        )

    call_kwargs = session_manager.generate_completion_with_session.call_args.kwargs
    assert call_kwargs["used_graph_element_ids"] == {"node_ids": ["chunk-1"]}


def _payload_text(chunk):
    if isinstance(chunk, dict):
        return chunk.get("text")
    return chunk.payload.get("text")
