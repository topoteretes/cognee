"""Unit tests for modules.integrations.slack.handle_cognee_ask.

/cognee-ask now acks fast (workspace/query validation only) and does the
real search in a background task, delivering the answer via response_url —
so these tests split into two groups: handle_cognee_ask's own fast-path
behavior (validation, immediate ack, scheduling), and _search_and_respond's
behavior in isolation (the part that used to be handle_cognee_ask's whole
body, now reachable only through the background task).
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
    _format_answer,
    _search_and_respond,
    handle_cognee_ask,
)
from cognee.modules.search.types import SearchType

MODULE = "cognee.modules.integrations.slack.handle_cognee_ask"


INVOKING_USER = "U100"


def _body(
    team_id: str = "T123",
    text: str = "what happened in Q3?",
    response_url: str = "https://hooks.slack.com/x",
    user_id: str = INVOKING_USER,
) -> bytes:
    return urlencode(
        {"team_id": team_id, "text": text, "response_url": response_url, "user_id": user_id}
    ).encode()


def _credential(user_id=None, installed_by_slack_user_id=INVOKING_USER):
    return SimpleNamespace(
        tenant_id=uuid4(),
        user_id=user_id or uuid4(),
        provider_metadata={"installed_by_slack_user_id": installed_by_slack_user_id},
        status=STATUS_ACTIVE,
    )


# ── handle_cognee_ask: fast path ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_unconnected_workspace_is_rejected():
    with patch(f"{MODULE}.get_by_team", new=AsyncMock(return_value=None)):
        response = await handle_cognee_ask(_body())

    assert "not connected" in response["text"]
    assert response["response_type"] == "ephemeral"


@pytest.mark.asyncio
async def test_rejects_non_installing_user():
    credential = _credential(installed_by_slack_user_id=INVOKING_USER)
    with patch(f"{MODULE}.get_by_team", new=AsyncMock(return_value=credential)):
        response = await handle_cognee_ask(_body(user_id="U999-someone-else"))

    assert "Only the Cognee account" in response["text"]
    assert response["response_type"] == "ephemeral"


@pytest.mark.asyncio
async def test_rejects_when_installed_by_is_unset():
    # Pre-existing connections made before installed_by_slack_user_id was
    # captured — must fail closed, not silently allow everyone.
    credential = _credential(installed_by_slack_user_id=None)
    with patch(f"{MODULE}.get_by_team", new=AsyncMock(return_value=credential)):
        response = await handle_cognee_ask(_body())

    assert "Only the Cognee account" in response["text"]
    assert response["response_type"] == "ephemeral"


@pytest.mark.asyncio
async def test_empty_query_returns_usage():
    credential = _credential()
    with patch(f"{MODULE}.get_by_team", new=AsyncMock(return_value=credential)):
        response = await handle_cognee_ask(_body(text=""))

    assert response["text"].startswith("Usage:")
    assert response["response_type"] == "ephemeral"


@pytest.mark.asyncio
async def test_valid_query_acks_immediately_without_the_final_answer():
    credential = _credential()
    with (
        patch(f"{MODULE}.get_by_team", new=AsyncMock(return_value=credential)),
        patch(f"{MODULE}._search_and_respond", new=AsyncMock()),
    ):
        response = await handle_cognee_ask(_body(text="what do we know?"))

    # in_channel here (not ephemeral) is what makes Slack show the
    # "<@user> used /cognee-ask ..." invocation line to everyone.
    assert response["response_type"] == "in_channel"
    assert "what do we know?" in response["text"]
    assert "replace_original" not in response  # the placeholder, not a replacement


@pytest.mark.asyncio
async def test_valid_query_schedules_the_background_search_with_right_args():
    credential = _credential()
    search_and_respond = AsyncMock()
    with (
        patch(f"{MODULE}.get_by_team", new=AsyncMock(return_value=credential)),
        patch(f"{MODULE}._search_and_respond", new=search_and_respond),
    ):
        await handle_cognee_ask(
            _body(
                team_id="T123", text="what do we know?", response_url="https://hooks.slack.com/xyz"
            )
        )
        await asyncio.sleep(0)  # let the scheduled task actually start

    search_and_respond.assert_awaited_once_with(
        "https://hooks.slack.com/xyz", "T123", credential.user_id, "what do we know?"
    )


@pytest.mark.asyncio
async def test_long_query_is_truncated_in_the_ack_message():
    credential = _credential()
    long_text = "x" * 500
    with (
        patch(f"{MODULE}.get_by_team", new=AsyncMock(return_value=credential)),
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
    assert payload["response_type"] == "in_channel"  # a found answer is shared, not private
    assert "fact one" in payload["text"]  # fallback text, for notifications/screen readers
    assert any(
        b["type"] == "section" and b["text"]["text"] == "fact one" for b in payload["blocks"]
    )


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
