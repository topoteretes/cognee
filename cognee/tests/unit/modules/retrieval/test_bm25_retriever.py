import asyncio

import pytest
from unittest.mock import AsyncMock, patch

from cognee.context_global_variables import current_dataset_id
from cognee.modules.retrieval.bm25_retriever import BM25ChunksRetriever


def _patch_graph(corpus: dict[str, str]):
    """Patch the graph engine so the lexical loader sees the given {chunk_id: text} corpus."""
    nodes = [
        (chunk_id, {"id": chunk_id, "type": "DocumentChunk", "text": text})
        for chunk_id, text in corpus.items()
    ]
    engine = AsyncMock()
    engine.get_filtered_graph_data = AsyncMock(return_value=(nodes, {}))
    return patch(
        "cognee.modules.retrieval.lexical_retriever.get_graph_engine",
        AsyncMock(return_value=engine),
    )


@pytest.mark.asyncio
async def test_payloads_carry_id_when_graph_node_payload_omits_it():
    # Some graph adapters (e.g. kuzu) omit "id" from node payloads; the loader must
    # backfill it from the node id so chunks can be matched across retrieval channels.
    nodes = [("chunk_a", {"type": "DocumentChunk", "text": "alpha project"})]
    engine = AsyncMock()
    engine.get_filtered_graph_data = AsyncMock(return_value=(nodes, {}))
    retriever = BM25ChunksRetriever(top_k=1, with_scores=True)

    with patch(
        "cognee.modules.retrieval.lexical_retriever.get_graph_engine",
        AsyncMock(return_value=engine),
    ):
        results = await retriever.get_retrieved_objects("project")

    assert results[0][0]["id"] == "chunk_a"


@pytest.mark.asyncio
async def test_term_frequency_orders_results():
    corpus = {
        "chunk_a": "alpha alpha alpha project",
        "chunk_b": "alpha project project project",
        "chunk_c": "beta gamma delta",
    }
    retriever = BM25ChunksRetriever(top_k=3, with_scores=True)

    with _patch_graph(corpus):
        results = await retriever.get_retrieved_objects("project")

    ranked_ids = [payload["id"] for payload, _ in results]
    assert ranked_ids[:2] == ["chunk_b", "chunk_a"]
    # chunk_c shares no term with the query → zero score, ranked last.
    assert results[-1][0]["id"] == "chunk_c"
    assert results[-1][1] == 0.0


@pytest.mark.asyncio
async def test_rare_term_beats_common_term():
    corpus = {
        "chunk_a": "status project common",
        "chunk_b": "status common common",
        "chunk_c": "status common raremarker",
    }
    retriever = BM25ChunksRetriever(top_k=3, with_scores=True)

    with _patch_graph(corpus):
        results = await retriever.get_retrieved_objects("status raremarker")

    ranked_ids = [payload["id"] for payload, _ in results]
    # "status" is in every chunk (low IDF); "raremarker" is unique and drives the top result.
    assert ranked_ids[0] == "chunk_c"


@pytest.mark.asyncio
async def test_empty_query_returns_empty_list():
    corpus = {"chunk_a": "alpha beta", "chunk_b": "gamma delta"}
    retriever = BM25ChunksRetriever(top_k=3)

    with _patch_graph(corpus):
        results = await retriever.get_retrieved_objects("   ")

    assert results == []


@pytest.mark.asyncio
async def test_stop_words_filtered_by_default():
    # "the" is a default stop word: it is dropped from both query and corpus, so a
    # query of only stop words yields no usable tokens and returns nothing.
    corpus = {"chunk_a": "the the the", "chunk_b": "the project"}
    retriever = BM25ChunksRetriever(top_k=3)

    with _patch_graph(corpus):
        results = await retriever.get_retrieved_objects("the")

    assert results == []


@pytest.mark.asyncio
async def test_stop_words_can_be_disabled():
    corpus = {"chunk_a": "the the the", "chunk_b": "the project"}
    retriever = BM25ChunksRetriever(top_k=3, stop_words=[])

    with _patch_graph(corpus):
        results = await retriever.get_retrieved_objects("the")

    # With filtering disabled, "the" is a real term and both chunks are scorable.
    assert len(results) == 2


@pytest.mark.asyncio
async def test_no_match_query_returns_zero_scored_chunks():
    corpus = {"chunk_a": "alpha beta", "chunk_b": "gamma delta"}
    retriever = BM25ChunksRetriever(top_k=3, with_scores=True)

    with _patch_graph(corpus):
        results = await retriever.get_retrieved_objects("zzz")

    # LexicalRetriever still returns top_k payloads for a no-match query; all score 0.0.
    assert len(results) == 2
    assert all(score == 0.0 for _, score in results)


def _scoped_engine(nodes):
    engine = AsyncMock()
    engine.get_filtered_graph_data = AsyncMock(return_value=(nodes, {}))
    return engine


@pytest.fixture(autouse=True)
def clear_bm25_corpus_cache():
    BM25ChunksRetriever.invalidate_cache()
    yield
    BM25ChunksRetriever.invalidate_cache()


@pytest.mark.asyncio
async def test_hybrid_cache_reuses_tokenized_corpus_for_same_dataset_and_scope():
    engine = _scoped_engine([("chunk_a", {"type": "DocumentChunk", "text": "alpha project"})])
    dataset_token = current_dataset_id.set("dataset-a")
    try:
        with patch(
            "cognee.modules.retrieval.lexical_retriever.get_graph_engine",
            AsyncMock(return_value=engine),
        ):
            first = BM25ChunksRetriever(top_k=1, use_cache=True)
            second = BM25ChunksRetriever(top_k=1, use_cache=True)
            await first.get_retrieved_objects("project")
            await second.get_retrieved_objects("project")
    finally:
        current_dataset_id.reset(dataset_token)

    engine.get_filtered_graph_data.assert_awaited_once()
    assert first.cache_status == "miss"
    assert second.cache_status == "hit"
    assert BM25ChunksRetriever.cache_info()["entries"] == 1


@pytest.mark.asyncio
async def test_nodeset_scope_filters_corpus_before_top_k_ranking():
    nodes = [
        (
            f"global-{index}",
            {
                "type": "DocumentChunk",
                "text": "needle " * (20 - index),
                "belongs_to_set": ["global"],
            },
        )
        for index in range(4)
    ]
    nodes.append(
        (
            "scoped",
            {
                "type": "DocumentChunk",
                "text": "needle appears once",
                "belongs_to_set": ["project-a", "figure"],
            },
        )
    )
    engine = _scoped_engine(nodes)
    retriever = BM25ChunksRetriever(
        top_k=1,
        with_scores=True,
        node_name=["project-a", "figure"],
        node_name_filter_operator="AND",
    )

    with patch(
        "cognee.modules.retrieval.lexical_retriever.get_graph_engine",
        AsyncMock(return_value=engine),
    ):
        results = await retriever.get_retrieved_objects("needle")

    assert [payload["id"] for payload, _ in results] == ["scoped"]
    assert set(retriever.payloads) == {"scoped"}


@pytest.mark.asyncio
async def test_cache_key_isolated_by_nodeset_scope():
    nodes = [
        (
            "scope-a",
            {
                "type": "DocumentChunk",
                "text": "alpha",
                "belongs_to_set": ["A"],
            },
        ),
        (
            "scope-b",
            {
                "type": "DocumentChunk",
                "text": "alpha",
                "belongs_to_set": ["B"],
            },
        ),
    ]
    engine = _scoped_engine(nodes)
    dataset_token = current_dataset_id.set("dataset-scopes")
    try:
        with patch(
            "cognee.modules.retrieval.lexical_retriever.get_graph_engine",
            AsyncMock(return_value=engine),
        ):
            scope_a = BM25ChunksRetriever(top_k=1, use_cache=True, node_name=["A"])
            scope_b = BM25ChunksRetriever(top_k=1, use_cache=True, node_name=["B"])
            result_a = await scope_a.get_retrieved_objects("alpha")
            result_b = await scope_b.get_retrieved_objects("alpha")
    finally:
        current_dataset_id.reset(dataset_token)

    assert result_a[0]["id"] == "scope-a"
    assert result_b[0]["id"] == "scope-b"
    assert engine.get_filtered_graph_data.await_count == 2
    assert BM25ChunksRetriever.cache_info()["entries"] == 2


@pytest.mark.asyncio
async def test_cache_ttl_and_explicit_dataset_invalidation_reload_corpus():
    engine = _scoped_engine([("chunk_a", {"type": "DocumentChunk", "text": "alpha project"})])
    dataset_token = current_dataset_id.set("dataset-expiring")
    try:
        with patch(
            "cognee.modules.retrieval.lexical_retriever.get_graph_engine",
            AsyncMock(return_value=engine),
        ):
            first = BM25ChunksRetriever(top_k=1, use_cache=True, cache_ttl_seconds=0.01)
            await first.get_retrieved_objects("project")
            await asyncio.sleep(0.02)
            expired = BM25ChunksRetriever(top_k=1, use_cache=True, cache_ttl_seconds=10)
            await expired.get_retrieved_objects("project")

            assert BM25ChunksRetriever.invalidate_cache("dataset-expiring") == 1
            invalidated = BM25ChunksRetriever(top_k=1, use_cache=True)
            await invalidated.get_retrieved_objects("project")
    finally:
        current_dataset_id.reset(dataset_token)

    assert engine.get_filtered_graph_data.await_count == 3
    assert expired.cache_status == "miss"
    assert invalidated.cache_status == "miss"


@pytest.mark.asyncio
async def test_concurrent_cache_misses_coalesce_to_one_graph_load():
    load_started = asyncio.Event()
    release_load = asyncio.Event()
    load_count = 0

    async def load_nodes(_filters):
        nonlocal load_count
        load_count += 1
        load_started.set()
        await release_load.wait()
        return ([("chunk_a", {"type": "DocumentChunk", "text": "alpha project"})], {})

    engine = AsyncMock()
    engine.get_filtered_graph_data = AsyncMock(side_effect=load_nodes)
    dataset_token = current_dataset_id.set("dataset-concurrent")
    try:
        with patch(
            "cognee.modules.retrieval.lexical_retriever.get_graph_engine",
            AsyncMock(return_value=engine),
        ):
            first = BM25ChunksRetriever(top_k=1, use_cache=True)
            second = BM25ChunksRetriever(top_k=1, use_cache=True)
            first_task = asyncio.create_task(first.get_retrieved_objects("project"))
            await load_started.wait()
            second_task = asyncio.create_task(second.get_retrieved_objects("project"))
            await asyncio.sleep(0)
            release_load.set()
            first_result, second_result = await asyncio.gather(first_task, second_task)
    finally:
        current_dataset_id.reset(dataset_token)

    assert load_count == 1
    assert first_result[0]["id"] == second_result[0]["id"] == "chunk_a"
    assert {first.cache_status, second.cache_status} == {"miss", "wait"}


@pytest.mark.asyncio
async def test_cache_bypasses_when_dataset_context_is_missing():
    engine = _scoped_engine([("chunk_a", {"type": "DocumentChunk", "text": "alpha project"})])
    dataset_token = current_dataset_id.set(None)
    try:
        with patch(
            "cognee.modules.retrieval.lexical_retriever.get_graph_engine",
            AsyncMock(return_value=engine),
        ):
            first = BM25ChunksRetriever(top_k=1, use_cache=True)
            second = BM25ChunksRetriever(top_k=1, use_cache=True)
            await first.get_retrieved_objects("project")
            await second.get_retrieved_objects("project")
    finally:
        current_dataset_id.reset(dataset_token)

    assert engine.get_filtered_graph_data.await_count == 2
    assert first.cache_status == second.cache_status == "bypass"
    assert BM25ChunksRetriever.cache_info()["entries"] == 0


@pytest.mark.asyncio
async def test_oversized_corpus_is_not_retained_in_process_cache():
    engine = _scoped_engine([("chunk_a", {"type": "DocumentChunk", "text": "alpha project"})])
    dataset_token = current_dataset_id.set("dataset-too-large")
    try:
        with (
            patch(
                "cognee.modules.retrieval.lexical_retriever.get_graph_engine",
                AsyncMock(return_value=engine),
            ),
            patch.object(BM25ChunksRetriever, "CACHE_MAX_TOKENS", 1),
        ):
            first = BM25ChunksRetriever(top_k=1, use_cache=True)
            second = BM25ChunksRetriever(top_k=1, use_cache=True)
            await first.get_retrieved_objects("project")
            await second.get_retrieved_objects("project")
    finally:
        current_dataset_id.reset(dataset_token)

    assert engine.get_filtered_graph_data.await_count == 2
    assert BM25ChunksRetriever.cache_info()["entries"] == 0
