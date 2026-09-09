"""Journey 3: session memory.

A user remembers something inside a session, gets it back from the session
cache immediately, then sees it bridged into the permanent graph after improve,
and can drop the session without losing the bridged knowledge.
"""

from __future__ import annotations

import pytest

import cognee
from cognee.tests.journeys import _support

DATASET = "journey_sessions"
SESSION = "journey-session-alpha"
OTHER_SESSION = "journey-session-beta"

FACT = (
    "The Quillon lighthouse keeper is Adaeze Wren, and the lamp was relit on 3 March 2024 "
    "after a nine-year silence."
)
QUESTION = "Who is the Quillon lighthouse keeper?"
EXPECTED = ("wren",)


def _texts(results) -> str:
    return _support.result_text(results)


def _sources(results) -> set[str]:
    return {getattr(r, "source", None) for r in results}


@pytest.mark.journey
@pytest.mark.asyncio
async def test_session_memory_roundtrip_bridge_and_cleanup(clean_env, default_user):
    from cognee.infrastructure.session.get_session_manager import get_session_manager

    # A dataset must exist for the session to attach to; seed it with one unrelated fact.
    seed = await cognee.remember(
        "Title: Harbour notes\n\nThe harbour master of Quillon is Tomas Ferreira.",
        dataset_name=DATASET,
    )
    assert seed.status == "completed"

    # --- 1. remember inside a session -----------------------------------------
    stored = await cognee.remember(FACT, dataset_name=DATASET, session_id=SESSION)
    assert stored.status == "session_stored", f"session remember did not store: {stored!r}"
    assert stored.session_id == SESSION

    # --- 2. recall inside the session hits the cache first ---------------------
    in_session = await cognee.recall(QUESTION, session_id=SESSION)
    assert in_session, "session recall returned nothing"
    assert "session" in _sources(in_session), (
        f"expected a session-sourced hit, got sources {_sources(in_session)}"
    )
    assert any(t in _texts(in_session) for t in EXPECTED), (
        f"session recall did not return the remembered fact: {_texts(in_session)[:300]}"
    )

    # --- 3. a different session does not see it -------------------------------
    other = await cognee.recall(QUESTION, session_id=OTHER_SESSION)
    assert "session" not in _sources(other), "fact leaked into an unrelated session"

    # --- 4. the session is listable through the public session API -------------
    entries = await cognee.session.get_session(SESSION, user=default_user)
    assert len(entries) >= 1, "session has no entries"
    assert any("wren" in (e.answer or "").lower() for e in entries), "session entry lost the fact"

    # --- 5. bridge into the permanent graph and recall without a session -------
    await stored  # wait for the background self-improvement bridge, if any
    await cognee.improve(DATASET, session_ids=[SESSION])

    from_graph = await cognee.recall(QUESTION, datasets=[DATASET])
    assert from_graph, "graph recall returned nothing after bridging"
    assert any(t in _texts(from_graph) for t in EXPECTED), (
        f"bridged fact not recalled from the graph: {_texts(from_graph)[:300]}"
    )

    # --- 6. dropping the session keeps the bridged knowledge -------------------
    sm = get_session_manager()
    deleted = await sm.delete_session(user_id=str(default_user.id), session_id=SESSION)
    assert deleted, "delete_session reported failure"

    after_delete = await cognee.recall(QUESTION, session_id=SESSION)
    assert "session" not in _sources(after_delete), "session entries survived deletion"

    still_in_graph = await cognee.recall(QUESTION, datasets=[DATASET])
    assert any(t in _texts(still_in_graph) for t in EXPECTED), (
        "bridged fact disappeared from the graph when the session was deleted"
    )


@pytest.mark.journey
@pytest.mark.asyncio
async def test_session_recall_falls_through_to_graph_when_session_has_no_match(
    clean_env, default_user
):
    seed = await cognee.remember(
        "Title: Bell tower\n\nThe Quillon bell tower was cast by the foundry of Halvard Ness in 1871.",
        dataset_name=DATASET,
    )
    assert seed.status == "completed"
    await cognee.remember("Reminder: buy oat milk.", dataset_name=DATASET, session_id=SESSION)

    results = await cognee.recall("Who cast the Quillon bell tower?", session_id=SESSION)
    assert results, "recall returned nothing"
    assert "ness" in _texts(results), (
        f"graph fact not reached when the session had no keyword match: {_texts(results)[:300]}"
    )
