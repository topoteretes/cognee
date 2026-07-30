"""Unit tests for modules.integrations.slack.handle_cognee_ask.

``/cognee-ask`` validates the workspace/query/authorization and starts the
search immediately in the background, acking with an in_channel placeholder
— splitting into two groups: handle_cognee_ask's own fast-path behavior
(validation, immediate ack, scheduling) and _search_and_respond's behavior
in isolation (reachable only through the background task). Once the search
finishes, the answer is shown back ephemerally first (Share/Discard) rather
than posted straight to the channel — handle_cognee_ask_share/_discard
cover that review step.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from urllib.parse import urlencode
from uuid import uuid4

import pytest

from cognee.infrastructure.databases.exceptions import EntityNotFoundError
from cognee.modules.integrations.credentials import STATUS_ACTIVE
from cognee.modules.integrations.slack.handle_cognee_ask import (
    ASK_DISCARD_ACTION_ID,
    ASK_SHARE_ACTION_ID,
    _format_answer,
    _pending_answers,
    _search_and_respond,
    handle_cognee_ask,
    handle_cognee_ask_discard,
    handle_cognee_ask_share,
)
from cognee.modules.search.types import SearchType

MODULE = "cognee.modules.integrations.slack.handle_cognee_ask"


def _body(
    team_id: str = "T123",
    text: str = "what happened in Q3?",
    response_url: str = "https://hooks.slack.com/x",
    user_id: str = "U100",
) -> bytes:
    return urlencode(
        {"team_id": team_id, "text": text, "response_url": response_url, "user_id": user_id}
    ).encode()


def _credential(user_id=None):
    return SimpleNamespace(
        tenant_id=uuid4(), user_id=user_id or uuid4(), provider_metadata={}, status=STATUS_ACTIVE
    )


# ── handle_cognee_ask: fast path ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_unconnected_workspace_is_rejected():
    with patch(f"{MODULE}.get_by_team", new=AsyncMock(return_value=None)):
        response = await handle_cognee_ask(_body())

    assert "not connected" in response["text"]
    assert response["response_type"] == "ephemeral"


@pytest.mark.asyncio
async def test_unauthorized_member_is_rejected():
    credential = _credential()
    with (
        patch(f"{MODULE}.get_by_team", new=AsyncMock(return_value=credential)),
        patch(f"{MODULE}.resolve_owner_user_id", new=AsyncMock(return_value=None)),
    ):
        response = await handle_cognee_ask(_body())

    assert "/cognee-link" in response["text"]
    assert response["response_type"] == "ephemeral"


@pytest.mark.asyncio
async def test_empty_query_returns_usage():
    credential = _credential()
    with (
        patch(f"{MODULE}.get_by_team", new=AsyncMock(return_value=credential)),
        patch(f"{MODULE}.resolve_owner_user_id", new=AsyncMock(return_value=uuid4())),
    ):
        response = await handle_cognee_ask(_body(text=""))

    assert response["text"].startswith("Usage:")
    assert response["response_type"] == "ephemeral"


@pytest.mark.asyncio
async def test_valid_query_acks_immediately_without_the_final_answer():
    credential = _credential()
    with (
        patch(f"{MODULE}.get_by_team", new=AsyncMock(return_value=credential)),
        patch(f"{MODULE}.resolve_owner_user_id", new=AsyncMock(return_value=uuid4())),
        patch(f"{MODULE}._search_and_respond", new=AsyncMock()),
    ):
        response = await handle_cognee_ask(_body(text="what do we know?"))

    # Ephemeral — nothing is visible to the channel until Share is clicked
    # on the answer, so even the placeholder ack must stay private.
    assert response["response_type"] == "ephemeral"
    assert "what do we know?" in response["text"]
    assert "replace_original" not in response  # the placeholder, not a replacement


@pytest.mark.asyncio
async def test_valid_query_schedules_the_background_search_with_right_args():
    credential = _credential()
    owner_id = uuid4()
    search_and_respond = AsyncMock()
    with (
        patch(f"{MODULE}.get_by_team", new=AsyncMock(return_value=credential)),
        patch(f"{MODULE}.resolve_owner_user_id", new=AsyncMock(return_value=owner_id)),
        patch(f"{MODULE}._search_and_respond", new=search_and_respond),
    ):
        await handle_cognee_ask(
            _body(
                team_id="T123", text="what do we know?", response_url="https://hooks.slack.com/xyz"
            )
        )
        await asyncio.sleep(0)  # let the scheduled task actually start

    search_and_respond.assert_awaited_once_with(
        "https://hooks.slack.com/xyz", "T123", owner_id, "what do we know?"
    )


@pytest.mark.asyncio
async def test_long_query_is_truncated_in_the_ack_message():
    credential = _credential()
    long_text = "x" * 500
    with (
        patch(f"{MODULE}.get_by_team", new=AsyncMock(return_value=credential)),
        patch(f"{MODULE}.resolve_owner_user_id", new=AsyncMock(return_value=uuid4())),
        patch(f"{MODULE}._search_and_respond", new=AsyncMock()),
    ):
        response = await handle_cognee_ask(_body(text=long_text))

    assert len(response["text"]) < 500 + 50
    assert "…" in response["text"]


# ── _search_and_respond: the actual work, now off the request/response cycle ──


@pytest.mark.asyncio
async def test_search_and_respond_posts_the_formatted_answer():
    owner = SimpleNamespace(id=uuid4())
    # This is the real runtime shape with backend access control enabled —
    # cognee.search() returns dicts, not SearchResult objects, despite its
    # type hint (see _extract_fact's docstring).
    results = [{"search_result": "fact one"}]
    with (
        patch(f"{MODULE}.get_user", new=AsyncMock(return_value=owner)),
        patch(f"{MODULE}.cognee_search", new=AsyncMock(return_value=results)),
        patch(f"{MODULE}.post_to_response_url", new=AsyncMock()) as post,
    ):
        await _search_and_respond("https://hooks.slack.com/x", "T123", owner.id, "what do we know?")

    post.assert_awaited_once()
    url, payload = post.call_args[0]
    assert url == "https://hooks.slack.com/x"
    assert payload["replace_original"] is True
    # A found answer is shown back privately first (Share/Discard) — never
    # posted straight to the channel before the asker has seen it.
    assert payload["response_type"] == "ephemeral"
    assert "fact one" in payload["text"]
    assert any(
        b["type"] == "section" and b["text"]["text"] == "fact one" for b in payload["blocks"]
    )
    actions = next(b for b in payload["blocks"] if b["type"] == "actions")
    action_ids = {el["action_id"] for el in actions["elements"]}
    assert action_ids == {ASK_SHARE_ACTION_ID, ASK_DISCARD_ACTION_ID}


@pytest.mark.asyncio
async def test_search_and_respond_uses_hybrid_completion():
    # Matches the frontend recall page's default — graph-only search can
    # come back empty on a thin graph where the vector fallback still hits.
    owner = SimpleNamespace(id=uuid4())
    search_mock = AsyncMock(return_value=[])
    with (
        patch(f"{MODULE}.get_user", new=AsyncMock(return_value=owner)),
        patch(f"{MODULE}.cognee_search", new=search_mock),
        patch(f"{MODULE}.post_to_response_url", new=AsyncMock()),
    ):
        await _search_and_respond("https://hooks.slack.com/x", "T123", owner.id, "q")

    _, kwargs = search_mock.call_args
    assert kwargs["query_type"] == SearchType.HYBRID_COMPLETION


@pytest.mark.asyncio
async def test_search_and_respond_reports_deleted_owner():
    with (
        patch(f"{MODULE}.get_user", new=AsyncMock(side_effect=EntityNotFoundError(message="gone"))),
        patch(f"{MODULE}.post_to_response_url", new=AsyncMock()) as post,
    ):
        await _search_and_respond("https://hooks.slack.com/x", "T123", uuid4(), "q")

    _, payload = post.call_args[0]
    assert "not fully configured" in payload["text"]
    assert payload["response_type"] == "ephemeral"  # errors are private, not shared


@pytest.mark.asyncio
async def test_search_and_respond_reports_search_failure():
    owner = SimpleNamespace(id=uuid4())
    with (
        patch(f"{MODULE}.get_user", new=AsyncMock(return_value=owner)),
        patch(f"{MODULE}.cognee_search", new=AsyncMock(side_effect=RuntimeError("boom"))),
        patch(f"{MODULE}.post_to_response_url", new=AsyncMock()) as post,
    ):
        await _search_and_respond("https://hooks.slack.com/x", "T123", owner.id, "q")

    _, payload = post.call_args[0]
    assert payload["text"] == "Search failed. Please try again."


# ── handle_cognee_ask_share / handle_cognee_ask_discard ─────────────────────


@pytest.mark.asyncio
async def test_share_posts_the_question_and_answer_in_channel():
    _pending_answers["abc123"] = (
        "what happened in Q3?",
        "*Answer:*\nfact one",
        [{"type": "section"}],
    )

    response = await handle_cognee_ask_share("https://hooks.slack.com/x", "abc123")

    assert response["response_type"] == "in_channel"
    assert response["replace_original"] is True
    assert "what happened in Q3?" in response["text"]
    assert "fact one" in response["text"]
    # The question is its own block, ahead of the answer's own blocks —
    # nothing about either was visible to the channel before this.
    assert response["blocks"][0]["text"]["text"] == '*Asked:* "what happened in Q3?"'
    assert any(b.get("type") == "section" for b in response["blocks"][1:])
    assert "abc123" not in _pending_answers  # popped, not left behind


@pytest.mark.asyncio
async def test_share_of_unknown_answer_id_is_a_clean_ephemeral_message():
    # Server restart, double-click, or an expired/already-handled id.
    response = await handle_cognee_ask_share("https://hooks.slack.com/x", "does-not-exist")

    assert response["response_type"] == "ephemeral"
    assert "no longer available" in response["text"]


@pytest.mark.asyncio
async def test_discard_drops_the_answer_and_replies_ephemeral():
    _pending_answers["abc456"] = (
        "what happened in Q3?",
        "*Answer:*\nfact one",
        [{"type": "section"}],
    )

    response = await handle_cognee_ask_discard("abc456")

    assert response == {
        "response_type": "ephemeral",
        "text": "Discarded.",
        "replace_original": True,
    }
    assert "abc456" not in _pending_answers


# ── _format_answer ──────────────────────────────────────────────────────────
#
# cognee.search() returns dicts with a "search_result" key when backend
# access control is enabled, or bare completion strings otherwise — never
# SearchResult objects (see _extract_fact's docstring). Both shapes are
# covered below, plus a shape this code doesn't recognize.


def test_format_answer_fallback_text_joins_up_to_three_results():
    results = [{"search_result": f"fact {i}"} for i in range(5)]
    fallback_text, _ = _format_answer(results)
    assert fallback_text.startswith("*Answer:*")
    assert "fact 0" in fallback_text and "fact 2" in fallback_text
    assert "fact 3" not in fallback_text


def test_format_answer_blocks_have_one_section_per_fact_with_dividers_between():
    results = [{"search_result": f"fact {i}"} for i in range(3)]
    _, blocks = _format_answer(results)

    assert blocks[0] == {
        "type": "header",
        "text": {"type": "plain_text", "text": "Answer", "emoji": True},
    }
    sections = [b for b in blocks if b["type"] == "section"]
    assert [s["text"]["text"] for s in sections] == ["fact 0", "fact 1", "fact 2"]
    # Dividers sit between facts, never trailing after the last one.
    assert blocks[-1]["type"] == "context"
    assert blocks[-2]["type"] == "section"


def test_format_answer_blocks_context_reports_result_count():
    results = [{"search_result": "fact 0"}, {"search_result": "fact 1"}]
    _, blocks = _format_answer(results)
    context = next(b for b in blocks if b["type"] == "context")
    assert "2 results" in context["elements"][0]["text"]


def test_format_answer_handles_bare_completion_strings():
    # The shape returned when backend access control is disabled.
    results = ["fact 0", "fact 1"]
    fallback_text, blocks = _format_answer(results)
    sections = [b for b in blocks if b["type"] == "section"]
    assert [s["text"]["text"] for s in sections] == ["fact 0", "fact 1"]
    assert "fact 0" in fallback_text


def test_format_answer_unwraps_search_result_lists():
    # HYBRID_COMPLETION (and others) wrap the answer in a single-element
    # list even inside the "search_result" dict key.
    results = [{"search_result": ["fact 0"]}, {"search_result": ["fact 1"]}]
    fallback_text, blocks = _format_answer(results)
    sections = [b for b in blocks if b["type"] == "section"]
    assert [s["text"]["text"] for s in sections] == ["fact 0", "fact 1"]
    assert "fact 0" in fallback_text


def test_format_answer_empty_results():
    fallback_text, blocks = _format_answer([])
    assert fallback_text == "No relevant information found."
    assert blocks == []


def test_format_answer_results_without_search_result_field():
    fallback_text, blocks = _format_answer([{"search_result": None}])
    assert fallback_text == "No relevant information found."
    assert blocks == []


def test_format_answer_skips_unrecognized_result_shapes():
    fallback_text, blocks = _format_answer([object(), 42, None])
    assert fallback_text == "No relevant information found."
    assert blocks == []


def test_format_answer_filters_out_refusal_facts():
    # A chunk irrelevant to the question produces its own "I can't answer"
    # completion alongside a real answer from a relevant chunk — the
    # refusal must not show up as if it were a second fact.
    results = [
        {"search_result": "LLMs are neural language models built on Transformers."},
        {
            "search_result": (
                "I can't—the provided context only covers the 2026 FIFA World Cup "
                "and contains no information about LLMs."
            )
        },
    ]
    fallback_text, blocks = _format_answer(results)
    sections = [b for b in blocks if b["type"] == "section"]
    assert len(sections) == 1
    assert "Transformers" in sections[0]["text"]["text"]
    assert "World Cup" not in fallback_text


def test_format_answer_all_refusals_reads_as_no_relevant_information():
    results = [{"search_result": "I cannot answer this from the given context."}]
    fallback_text, blocks = _format_answer(results)
    assert fallback_text == "No relevant information found."
    assert blocks == []
