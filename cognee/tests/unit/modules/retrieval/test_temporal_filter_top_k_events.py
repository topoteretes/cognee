"""Regression test: TEMPORAL event ranking must apply the vector scores.

``TemporalRetriever.filter_top_k_events`` ranks the time-filtered events by the
similarity score of the ``Event_name`` vector search. It built the score lookup
keyed by ``ScoredResult.id`` (a ``uuid.UUID``) but looked it up with
``event["id"]`` (a plain string from the graph). ``UUID(x) != str(x)`` and they
hash differently, so every lookup returned the ``inf`` default — all events tied,
the sort was a no-op, and the events came back in arbitrary graph order with the
vector ranking discarded. Keys are now normalized to ``str``.
"""

import uuid

import pytest

from cognee.infrastructure.databases.vector.models.ScoredResult import ScoredResult
from cognee.modules.retrieval.temporal_retriever import TemporalRetriever


@pytest.mark.asyncio
async def test_events_are_ranked_by_vector_score():
    id_a = uuid.uuid4()
    id_b = uuid.uuid4()
    id_c = uuid.uuid4()

    # Graph events carry string ids and are returned in a non-score order.
    relevant_events = [
        {
            "events": [
                {"id": str(id_a), "name": "A", "description": "a"},
                {"id": str(id_b), "name": "B", "description": "b"},
                {"id": str(id_c), "name": "C", "description": "c"},
            ]
        }
    ]
    # Lower score is better; B is most relevant, then C, then A.
    scored_results = [
        ScoredResult(id=id_a, score=0.9, payload={}),
        ScoredResult(id=id_b, score=0.1, payload={}),
        ScoredResult(id=id_c, score=0.5, payload={}),
    ]

    retriever = TemporalRetriever(top_k=3)
    top = await retriever.filter_top_k_events(relevant_events, scored_results)

    # The scores must actually be applied (not the inf fallback)...
    assert [e["score"] for e in top] == [0.1, 0.5, 0.9]
    # ...and the events ordered by relevance.
    assert [e["name"] for e in top] == ["B", "C", "A"]


@pytest.mark.asyncio
async def test_top_k_truncates_to_best_matches():
    ids = [uuid.uuid4() for _ in range(3)]
    relevant_events = [
        {"events": [{"id": str(i), "name": n} for i, n in zip(ids, ["A", "B", "C"])]}
    ]
    scored_results = [
        ScoredResult(id=ids[0], score=0.9, payload={}),
        ScoredResult(id=ids[1], score=0.1, payload={}),
        ScoredResult(id=ids[2], score=0.5, payload={}),
    ]

    retriever = TemporalRetriever(top_k=2)
    top = await retriever.filter_top_k_events(relevant_events, scored_results)

    # The two best (lowest score) events, in order.
    assert [e["name"] for e in top] == ["B", "C"]
