"""Journey 2: memory correctness on the golden corpus.

A user remembers twenty documents and asks thirty questions whose answers are
invented facts. Every search path a user is likely to touch must surface the
fact, not merely return something.

Mock mode asserts exact-or-near-exact hit rates (retrieval is deterministic).
Real-LLM mode asserts thresholds and additionally checks that concise answers
do not leak facts from unrelated documents.
"""

from __future__ import annotations

import pytest

import cognee
from cognee.modules.search.types import SearchType
from cognee.tests.journeys import _support

DATASET = "journey_golden_corpus"

# Minimum share of questions each path must answer, per mode.
THRESHOLDS = {
    # Observed in mock mode on the default stack: recall 90%, CHUNKS 100%,
    # RAG 100%, GRAPH_COMPLETION 97%, SUMMARIES 87%. Thresholds sit one or two
    # questions below so a hashing-collision tie cannot flake the gate, while a
    # real retrieval regression (several questions) still fails it.
    "mock": {
        "recall_default": 0.85,
        SearchType.CHUNKS: 0.95,
        SearchType.RAG_COMPLETION: 0.95,
        SearchType.GRAPH_COMPLETION: 0.90,
        SearchType.SUMMARIES: 0.80,
    },
    "llm": {
        "recall_default": 0.85,
        SearchType.CHUNKS: 0.90,
        SearchType.RAG_COMPLETION: 0.85,
        SearchType.GRAPH_COMPLETION: 0.85,
        SearchType.SUMMARIES: 0.70,
    },
}


@pytest.mark.journey
@pytest.mark.asyncio
async def test_golden_corpus_is_answerable_through_every_user_path(
    clean_env, default_user, corpus, questions, journey_mode
):
    thresholds = THRESHOLDS[journey_mode]
    check_forbidden = journey_mode == "llm"

    # --- remember the whole corpus through the public API --------------------
    result = await cognee.remember([d.text for d in corpus], dataset_name=DATASET)
    assert result.status == "completed", f"remember did not complete: {result!r}"
    assert result.dataset_id, "remember returned no dataset id"

    snapshot = await _support.snapshot_dataset(result.dataset_id, default_user)
    assert snapshot.data_rows == len(corpus), f"expected one data row per document, got {snapshot}"
    assert snapshot.graph_nodes >= len(corpus) * 3, f"graph too small for the corpus: {snapshot}"
    assert snapshot.graph_edges >= len(corpus) * 2, f"graph has too few edges: {snapshot}"
    if snapshot.vector_rows is not None:
        assert snapshot.vector_rows.get("DocumentChunk_text", 0) >= len(corpus), (
            f"chunks were not embedded: {snapshot.vector_rows}"
        )

    # Each question runs in its own session. Without a session id every turn
    # shares one per-dataset default session and the conversational lane merges
    # the previous turn's retrievals into this one, so thirty unrelated questions
    # asked back to back would contaminate each other's context. Isolation here
    # measures retrieval; the session journey covers conversational behaviour.
    cards: list[_support.Scorecard] = []

    # --- recall with the product default (auto-routed) -----------------------
    card = _support.Scorecard("recall() default routing")
    for q in questions:
        results = await cognee.recall(q.question, datasets=[DATASET], session_id=f"{q.id}-recall")
        assert results, f"recall returned nothing for {q.id}: {q.question!r}"
        card.record(q, _support.result_text(results), check_forbidden)
    cards.append(card)
    assert card.rate >= thresholds["recall_default"], card.report()

    # --- every explicit retrieval path --------------------------------------
    for search_type in (
        SearchType.CHUNKS,
        SearchType.RAG_COMPLETION,
        SearchType.GRAPH_COMPLETION,
        SearchType.SUMMARIES,
    ):
        card = _support.Scorecard(f"search({search_type.value})")
        for q in questions:
            results = await cognee.search(
                query_text=q.question,
                query_type=search_type,
                datasets=[DATASET],
                session_id=f"{q.id}-{search_type.value}",
            )
            assert results, f"{search_type.value} returned nothing for {q.id}: {q.question!r}"
            # Completions echo retrieved context in mock mode; a leak check there is meaningless.
            card.record(q, _support.result_text(results), check_forbidden)
        cards.append(card)
        assert card.rate >= thresholds[search_type], card.report()

    # Leaks are a correctness failure in real-LLM mode: the answer cited the wrong document.
    if check_forbidden:
        leaking = [c.report() for c in cards if c.leaks]
        assert not leaking, "answers cited facts from unrelated documents:\n" + "\n".join(leaking)

    print("\n".join(c.report() for c in cards))


@pytest.mark.journey
@pytest.mark.asyncio
async def test_questions_about_unknown_topics_do_not_fabricate_corpus_facts(
    clean_env, default_user, corpus, journey_mode
):
    """Asking about something the corpus never mentions must not return corpus facts
    as if they were the answer. In mock mode the echo has no matching words to
    surface; in LLM mode the model has to say it does not know."""
    result = await cognee.remember([d.text for d in corpus], dataset_name=DATASET)
    assert result.status == "completed"

    results = await cognee.search(
        query_text="Zzqx plimble frobnicate glorp",  # no token overlap with any document
        query_type=SearchType.CHUNKS,
        datasets=[DATASET],
        top_k=1,
    )
    # CHUNKS always returns the nearest neighbours; the contract here is only that
    # the pipeline does not crash and returns a well-formed list.
    assert isinstance(results, list)

    if journey_mode == "llm":
        answer = await cognee.recall(
            "What is the population of the city of Zzqx Plimble?",
            datasets=[DATASET],
            session_id="unknown-topic",
        )
        text = _support.result_text(answer)
        assert "14,200" not in text and "14200" not in text, (
            "model answered an unknown city with Kestrel Hollow's population"
        )
