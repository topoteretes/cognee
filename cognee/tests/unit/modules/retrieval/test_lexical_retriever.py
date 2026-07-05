import pytest

from cognee.modules.retrieval.lexical_retriever import LexicalRetriever, tokenize_words


def overlap_scorer(query_tokens, chunk_tokens):
    return len(set(query_tokens) & set(chunk_tokens))


def make_retriever(with_scores: bool) -> LexicalRetriever:
    retriever = LexicalRetriever(
        tokenizer=tokenize_words, scorer=overlap_scorer, with_scores=with_scores
    )
    # Populate the caches directly, the way initialize() would.
    retriever.chunks = {
        "c1": tokenize_words("alpha beta gamma"),
        "c2": tokenize_words("delta epsilon"),
    }
    retriever.payloads = {
        "c1": {"id": "c1", "text": "alpha beta gamma", "type": "DocumentChunk"},
        "c2": {"id": "c2", "text": "delta epsilon", "type": "DocumentChunk"},
    }
    retriever._initialized = True
    return retriever


@pytest.mark.asyncio
async def test_get_context_from_objects_without_scores():
    retriever = make_retriever(with_scores=False)

    objects = await retriever.get_retrieved_objects("alpha")
    context = await retriever.get_context_from_objects("alpha", objects)

    assert "alpha beta gamma" in context


@pytest.mark.asyncio
async def test_get_context_from_objects_with_scores():
    """Both constructor modes must work with the inherited get_context flow.

    Regression test: with_scores=True makes get_retrieved_objects return
    (payload, score) tuples, and get_context_from_objects indexed them as
    dicts — TypeError: tuple indices must be integers or slices, not str.
    """
    retriever = make_retriever(with_scores=True)

    objects = await retriever.get_retrieved_objects("alpha")
    assert objects and all(isinstance(entry, tuple) for entry in objects)

    context = await retriever.get_context_from_objects("alpha", objects)

    assert "alpha beta gamma" in context


@pytest.mark.asyncio
async def test_get_context_from_objects_empty():
    retriever = make_retriever(with_scores=True)

    assert await retriever.get_context_from_objects("alpha", []) == ""
