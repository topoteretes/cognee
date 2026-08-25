import pytest
from unittest.mock import AsyncMock, patch

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


def _patch_graph_nodes(nodes: list[tuple[str, dict]]):
    """Patch the graph engine with fully formed node payloads.

    Node sets are carried on the chunk payload itself: graph serialization keeps
    ``belongs_to_set`` as a property, reduced to NodeSet names, so it can be
    filtered on the same way the vector payloads are.
    """
    engine = AsyncMock()
    engine.get_filtered_graph_data = AsyncMock(return_value=(nodes, {}))
    return patch(
        "cognee.modules.retrieval.lexical_retriever.get_graph_engine",
        AsyncMock(return_value=engine),
    )


def _chunk(chunk_id: str, text: str, node_sets=None) -> tuple[str, dict]:
    payload = {"id": chunk_id, "type": "DocumentChunk", "text": text}
    if node_sets is not None:
        payload["belongs_to_set"] = node_sets
    return (chunk_id, payload)


TAGGED_CORPUS = [
    _chunk("chunk_a", "alpha project", ["set_a"]),
    _chunk("chunk_b", "alpha project", ["set_b"]),
    _chunk("chunk_ab", "alpha project", ["set_a", "set_b"]),
    _chunk("chunk_untagged", "alpha project"),
]


@pytest.mark.asyncio
async def test_node_name_scopes_the_corpus_with_or():
    retriever = BM25ChunksRetriever(top_k=10, node_name=["set_a"])

    with _patch_graph_nodes(TAGGED_CORPUS):
        results = await retriever.get_retrieved_objects("project")

    assert {payload["id"] for payload in results} == {"chunk_a", "chunk_ab"}


@pytest.mark.asyncio
async def test_node_name_or_matches_any_requested_set():
    retriever = BM25ChunksRetriever(top_k=10, node_name=["set_a", "set_b"])

    with _patch_graph_nodes(TAGGED_CORPUS):
        results = await retriever.get_retrieved_objects("project")

    assert {payload["id"] for payload in results} == {"chunk_a", "chunk_b", "chunk_ab"}


@pytest.mark.asyncio
async def test_node_name_and_requires_every_requested_set():
    retriever = BM25ChunksRetriever(
        top_k=10, node_name=["set_a", "set_b"], node_name_filter_operator="AND"
    )

    with _patch_graph_nodes(TAGGED_CORPUS):
        results = await retriever.get_retrieved_objects("project")

    # Only the chunk carrying both tags qualifies; a chunk in one of the two does not.
    assert {payload["id"] for payload in results} == {"chunk_ab"}


@pytest.mark.asyncio
async def test_no_node_name_searches_every_chunk():
    retriever = BM25ChunksRetriever(top_k=10)

    with _patch_graph_nodes(TAGGED_CORPUS):
        results = await retriever.get_retrieved_objects("project")

    assert len(results) == len(TAGGED_CORPUS)


@pytest.mark.asyncio
async def test_node_set_tags_stored_as_objects_are_matched_by_name():
    class _NodeSet:
        name = "set_a"

    corpus = [
        _chunk("chunk_mapping", "alpha project", [{"name": "set_a"}]),
        _chunk("chunk_object", "alpha project", [_NodeSet()]),
        _chunk("chunk_other", "alpha project", ["set_b"]),
    ]
    retriever = BM25ChunksRetriever(top_k=10, node_name=["set_a"])

    with _patch_graph_nodes(corpus):
        results = await retriever.get_retrieved_objects("project")

    assert {payload["id"] for payload in results} == {"chunk_mapping", "chunk_object"}


@pytest.mark.asyncio
async def test_corpus_statistics_are_built_from_the_scoped_chunks_only():
    corpus = [
        _chunk("chunk_a", "alpha shared", ["set_a"]),
        _chunk("chunk_b", "beta shared outsideterm", ["set_b"]),
    ]
    retriever = BM25ChunksRetriever(top_k=10, node_name=["set_a"])

    with _patch_graph_nodes(corpus):
        await retriever.get_retrieved_objects("shared")

    # A term that only exists outside the scope must not reach the corpus stats,
    # otherwise IDF would describe a corpus the search cannot return.
    assert "outsideterm" not in retriever.idf
    assert set(retriever.chunks) == {"chunk_a"}


@pytest.mark.asyncio
async def test_empty_node_set_returns_no_results_instead_of_reporting_no_data():
    retriever = BM25ChunksRetriever(top_k=10, node_name=["set_missing"])

    with _patch_graph_nodes(TAGGED_CORPUS):
        results = await retriever.get_retrieved_objects("project")

    # The system holds data, the requested node set is simply empty. The vector
    # chunk search returns nothing here rather than raising, so this does too.
    assert results == []


@pytest.mark.asyncio
async def test_genuinely_empty_corpus_still_raises_no_data():
    from cognee.modules.retrieval.exceptions.exceptions import NoDataError

    retriever = BM25ChunksRetriever(top_k=10, node_name=["set_a"])

    with _patch_graph_nodes([]), pytest.raises(NoDataError):
        await retriever.get_retrieved_objects("project")
