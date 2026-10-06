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
    return _support.content_text(results)


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


# ---------------------------------------------------------------------------
# node_set pinned on a session (SDK-336)
# ---------------------------------------------------------------------------

PROJECT_A = "project-journey-a"
PROJECT_B = "project-journey-b"
SESSION_A = "journey-session-project-a"
SESSION_B = "journey-session-project-b"
FACT_A = (
    "The Quillon ferry schedule is kept by Marisol Okonkwo, and the last crossing "
    "leaves at nine in the evening."
)
FACT_B = (
    "The Quillon ferry schedule is kept by Bartholomew Lindqvist, and the last crossing "
    "leaves at six in the morning."
)
FERRY_QUESTION = "Who keeps the Quillon ferry schedule?"
TOKEN_A = "okonkwo"
TOKEN_B = "lindqvist"


async def _ferry_chunks(node_name: list[str] | None) -> str:
    """Raw chunk retrieval of the ferry question, optionally scoped to a node set.

    CHUNKS returns stored text and nothing else, so what comes back is exactly
    what the node-set filter let through: no completion, no conversation
    history around it.
    """
    from cognee import SearchType

    kwargs = {"datasets": [DATASET], "query_type": SearchType.CHUNKS}
    if node_name is not None:
        kwargs["node_name"] = node_name
    return _texts(await cognee.recall(FERRY_QUESTION, **kwargs))


async def _ferry_answer(node_name: list[str]) -> str:
    """The default completion route, scoped, the way the plugins call it.

    Each probe runs in its own fresh session with ``scope="graph"``: the mock
    LLM echoes its whole prompt, and a sessionless completion would carry the
    previous probe's turn as conversation history, which is not a filter leak.
    """
    from uuid import uuid4

    return _texts(
        await cognee.recall(
            FERRY_QUESTION,
            datasets=[DATASET],
            node_name=node_name,
            scope="graph",
            session_id=f"journey-probe-{uuid4().hex[:8]}",
        )
    )


@pytest.mark.journey
@pytest.mark.asyncio
async def test_session_node_set_scopes_what_improve_bridges_into_the_graph(clean_env, default_user):
    """Two sessions pinned to two node sets; after improve, node_name-scoped recall keeps them apart.

    This is the contract the integrations plugins build per-project memory on:
    the set a typed entry carries must pin the session, survive the bridge into
    the graph, and be the thing ``recall(node_name=...)`` filters on.
    """
    from cognee.infrastructure.session.get_session_manager import get_session_manager
    from cognee.infrastructure.session.session_node_set import get_session_node_set

    seed = await cognee.remember(
        "Title: Harbour notes\n\nThe harbour master of Quillon is Tomas Ferreira.",
        dataset_name=DATASET,
    )
    assert seed.status == "completed"

    # --- 1. the first typed entry pins the session ----------------------------------
    stored = await cognee.remember(
        cognee.QAEntry(question=FERRY_QUESTION, answer=FACT_A, node_set=[PROJECT_A]),
        dataset_name=DATASET,
        session_id=SESSION_A,
    )
    assert stored.status == "session_stored", repr(stored)

    sm = get_session_manager()
    user_id = str(default_user.id)
    assert await get_session_node_set(sm, user_id, SESSION_A) == (PROJECT_A,)

    # --- 2. later entries may repeat it, omit it, or send an empty list ------------
    await cognee.remember(
        cognee.TraceEntry(
            origin_function="ferry_lookup",
            status="success",
            method_return_value={"keeper": "Marisol Okonkwo"},
            node_set=[PROJECT_A],
        ),
        dataset_name=DATASET,
        session_id=SESSION_A,
    )
    await cognee.remember(
        cognee.QAEntry(question="Anything else?", answer="No.", node_set=[]),
        dataset_name=DATASET,
        session_id=SESSION_A,
    )
    # The call-level kwarg is the same thing as the field.
    await cognee.remember(
        cognee.QAEntry(question="Noted?", answer="Noted."),
        dataset_name=DATASET,
        session_id=SESSION_A,
        node_set=[PROJECT_A],
    )
    assert await get_session_node_set(sm, user_id, SESSION_A) == (PROJECT_A,)

    # --- 3. a different set on the same session is refused and writes nothing ------
    before = len(await cognee.session.get_session(SESSION_A, user=default_user))
    with pytest.raises(cognee.SessionNodeSetConflictError) as refused:
        await cognee.remember(
            cognee.QAEntry(question="q", answer="a", node_set=[PROJECT_B]),
            dataset_name=DATASET,
            session_id=SESSION_A,
        )
    assert refused.value.status_code == 409
    assert PROJECT_A in str(refused.value) and PROJECT_B in str(refused.value)
    assert len(await cognee.session.get_session(SESSION_A, user=default_user)) == before

    # --- 4. a second session is pinned to the other set through the kwarg ----------
    stored_b = await cognee.remember(
        FACT_B, dataset_name=DATASET, session_id=SESSION_B, node_set=[PROJECT_B]
    )
    assert await get_session_node_set(sm, user_id, SESSION_B) == (PROJECT_B,)
    # Plain-text session writes start the background self-improvement bridge;
    # wait for it so the explicit improve below is not refused with lock_held.
    await stored_b

    # --- 5. bridge both sessions, then recall scoped by node set -------------------
    improved = await cognee.improve(DATASET, session_ids=[SESSION_A, SESSION_B])
    by_stage = {stage.stage: stage for stage in improved.stages}
    assert by_stage["persist_session_qa"].status == "completed", by_stage["persist_session_qa"]
    assert by_stage["persist_agent_traces"].status != "errored", by_stage["persist_agent_traces"]

    # The filter itself: scoped chunk retrieval returns one session's text, never the other's.
    chunks_a = await _ferry_chunks([PROJECT_A])
    assert TOKEN_A in chunks_a, f"project A scope lost its own fact: {chunks_a[:300]}"
    assert TOKEN_B not in chunks_a, f"project A scope leaked project B: {chunks_a[:300]}"

    chunks_b = await _ferry_chunks([PROJECT_B])
    assert TOKEN_B in chunks_b, f"project B scope lost its own fact: {chunks_b[:300]}"
    assert TOKEN_A not in chunks_b, f"project B scope leaked project A: {chunks_b[:300]}"

    unscoped = await _ferry_chunks(None)
    assert TOKEN_A in unscoped and TOKEN_B in unscoped, (
        f"unscoped retrieval should see both sessions: {unscoped[:300]}"
    )

    # The route the plugins use: auto-routed completion with node_name.
    answer_a = await _ferry_answer([PROJECT_A])
    assert TOKEN_A in answer_a and TOKEN_B not in answer_a, (
        f"scoped completion for project A: {answer_a[:300]}"
    )
    answer_b = await _ferry_answer([PROJECT_B])
    assert TOKEN_B in answer_b and TOKEN_A not in answer_b, (
        f"scoped completion for project B: {answer_b[:300]}"
    )

    # --- 6. the pin is session state: it goes when the session goes ----------------
    assert await sm.delete_session(user_id=user_id, session_id=SESSION_A)
    assert await get_session_node_set(sm, user_id, SESSION_A) == ()
    assert TOKEN_A in await _ferry_chunks([PROJECT_A]), (
        "bridged, node-set-scoped knowledge disappeared with the session"
    )
