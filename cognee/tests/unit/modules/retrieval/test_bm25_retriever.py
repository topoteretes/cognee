import pytest
from unittest.mock import AsyncMock, patch

from cognee.modules.retrieval.bm25_retriever import BM25ChunksRetriever
from cognee.modules.retrieval.utils import lexical_corpus_cache


@pytest.fixture(autouse=True)
def _clear_corpus_cache():
    lexical_corpus_cache.invalidate()
    yield
    lexical_corpus_cache.invalidate()


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


def _engine_with_corpus(corpus: dict[str, str]):
    nodes = [
        (chunk_id, {"id": chunk_id, "type": "DocumentChunk", "text": text})
        for chunk_id, text in corpus.items()
    ]
    engine = AsyncMock()
    engine.get_filtered_graph_data = AsyncMock(return_value=(nodes, {}))
    return engine


@pytest.mark.asyncio
async def test_corpus_is_cached_across_retriever_instances():
    engine = _engine_with_corpus({"chunk_a": "alpha project", "chunk_b": "beta archive"})

    with patch(
        "cognee.modules.retrieval.lexical_retriever.get_graph_engine",
        AsyncMock(return_value=engine),
    ):
        first = await BM25ChunksRetriever(top_k=2, with_scores=True).get_retrieved_objects(
            "project"
        )
        second = await BM25ChunksRetriever(top_k=2, with_scores=True).get_retrieved_objects(
            "project"
        )

    assert engine.get_filtered_graph_data.await_count == 1
    # Scores must match exactly: the cached BM25 stats (idf, avg length) were restored.
    assert [(payload["id"], score) for payload, score in first] == [
        (payload["id"], score) for payload, score in second
    ]


@pytest.mark.asyncio
async def test_invalidate_forces_corpus_reload():
    engine = _engine_with_corpus({"chunk_a": "alpha project"})

    with patch(
        "cognee.modules.retrieval.lexical_retriever.get_graph_engine",
        AsyncMock(return_value=engine),
    ):
        await BM25ChunksRetriever(top_k=1).get_retrieved_objects("project")
        lexical_corpus_cache.invalidate()
        await BM25ChunksRetriever(top_k=1).get_retrieved_objects("project")

    assert engine.get_filtered_graph_data.await_count == 2


@pytest.mark.asyncio
async def test_different_stop_words_use_separate_cache_entries():
    engine = _engine_with_corpus({"chunk_a": "the alpha project"})

    with patch(
        "cognee.modules.retrieval.lexical_retriever.get_graph_engine",
        AsyncMock(return_value=engine),
    ):
        await BM25ChunksRetriever(top_k=1).get_retrieved_objects("project")
        await BM25ChunksRetriever(top_k=1, stop_words=[]).get_retrieved_objects("project")

    assert engine.get_filtered_graph_data.await_count == 2
