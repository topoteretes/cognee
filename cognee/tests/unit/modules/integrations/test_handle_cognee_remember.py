"""Unit tests for cognee.modules.integrations.slack.handle_cognee_remember.

Credential lookup, owner resolution, the ingest call and the response_url
delivery are all patched out — this is about the handler's contract, not
about cognee.remember.

Invariants: an unconnected workspace and an unlinked member are both refused
before anything is written; empty text gets usage rather than an empty note;
a valid call acks inside Slack's 3-second window and does the save detached,
confirming (or reporting failure) afterwards via response_url.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from urllib.parse import urlencode
from uuid import uuid4

import pytest

from cognee.modules.integrations.credentials import STATUS_ACTIVE
from cognee.modules.integrations.slack.handle_cognee_remember import (
    REMEMBERED_MESSAGE,
    SAVE_FAILED_MESSAGE,
    handle_cognee_remember,
)

MODULE = "cognee.modules.integrations.slack.handle_cognee_remember"


def _body(text="we chose Neon for v2", team_id="T123", user_id="U1"):
    return urlencode(
        {
            "command": "/cognee-remember",
            "team_id": team_id,
            "channel_id": "C1",
            "user_id": user_id,
            "text": text,
            "response_url": "https://hooks.slack.test/response",
        }
    ).encode()


def _credential():
    return SimpleNamespace(provider_metadata={}, status=STATUS_ACTIVE)


async def _drain_pending_saves():
    """Let the detached save task run to completion before asserting on it."""
    from cognee.modules.integrations.slack.handle_cognee_remember import _pending_saves

    if _pending_saves:
        await asyncio.gather(*list(_pending_saves), return_exceptions=True)


@pytest.mark.asyncio
async def test_unconnected_workspace_is_refused_without_writing():
    with (
        patch(f"{MODULE}.get_by_team", new=AsyncMock(return_value=None)),
        patch(f"{MODULE}.remember_note", new=AsyncMock()) as note,
    ):
        response = await handle_cognee_remember(_body())

    note.assert_not_awaited()
    assert "not connected" in response["text"]


@pytest.mark.asyncio
async def test_unlinked_member_is_refused_without_writing():
    with (
        patch(f"{MODULE}.get_by_team", new=AsyncMock(return_value=_credential())),
        patch(f"{MODULE}.resolve_owner_user_id", new=AsyncMock(return_value=None)),
        patch(f"{MODULE}.remember_note", new=AsyncMock()) as note,
    ):
        response = await handle_cognee_remember(_body())

    note.assert_not_awaited()
    # The point of the message: name the command that fixes it.
    assert "/cognee-link" in response["text"]


@pytest.mark.asyncio
async def test_empty_text_returns_usage_rather_than_saving_nothing():
    with (
        patch(f"{MODULE}.get_by_team", new=AsyncMock(return_value=_credential())),
        patch(f"{MODULE}.resolve_owner_user_id", new=AsyncMock(return_value=uuid4())),
        patch(f"{MODULE}.remember_note", new=AsyncMock()) as note,
    ):
        response = await handle_cognee_remember(_body(text="   "))

    note.assert_not_awaited()
    assert "/cognee-remember" in response["text"]
    # Points at the other way in, which is the right answer more often.
    assert "Remember this" in response["text"]


@pytest.mark.asyncio
async def test_valid_call_acks_immediately_and_saves_detached():
    user_id = uuid4()
    with (
        patch(f"{MODULE}.get_by_team", new=AsyncMock(return_value=_credential())),
        patch(f"{MODULE}.resolve_owner_user_id", new=AsyncMock(return_value=user_id)),
        patch(f"{MODULE}.remember_note", new=AsyncMock()) as note,
        patch(f"{MODULE}.post_to_response_url", new=AsyncMock()) as respond,
    ):
        response = await handle_cognee_remember(_body(text="we chose Neon for v2"))

        # The ack is the return value; the save has not necessarily run yet.
        assert response["response_type"] == "ephemeral"
        assert "we chose Neon for v2" in response["text"]

        await _drain_pending_saves()

    note.assert_awaited_once()
    assert note.await_args.kwargs["text"] == "we chose Neon for v2"
    assert note.await_args.kwargs["author_id"] == "U1"

    respond.assert_awaited_once()
    assert respond.await_args.args[1]["text"] == REMEMBERED_MESSAGE


@pytest.mark.asyncio
async def test_save_failure_reports_back_instead_of_raising():
    with (
        patch(f"{MODULE}.get_by_team", new=AsyncMock(return_value=_credential())),
        patch(f"{MODULE}.resolve_owner_user_id", new=AsyncMock(return_value=uuid4())),
        patch(f"{MODULE}.remember_note", new=AsyncMock(side_effect=RuntimeError("boom"))),
        patch(f"{MODULE}.post_to_response_url", new=AsyncMock()) as respond,
    ):
        await handle_cognee_remember(_body())
        await _drain_pending_saves()

    respond.assert_awaited_once()
    assert respond.await_args.args[1]["text"] == SAVE_FAILED_MESSAGE


@pytest.mark.asyncio
async def test_long_note_is_truncated_in_the_ack_only():
    long_text = "x" * 500
    with (
        patch(f"{MODULE}.get_by_team", new=AsyncMock(return_value=_credential())),
        patch(f"{MODULE}.resolve_owner_user_id", new=AsyncMock(return_value=uuid4())),
        patch(f"{MODULE}.remember_note", new=AsyncMock()) as note,
        patch(f"{MODULE}.post_to_response_url", new=AsyncMock()),
    ):
        response = await handle_cognee_remember(_body(text=long_text))
        await _drain_pending_saves()

    assert len(response["text"]) < len(long_text)
    # Truncation is presentation only — the note keeps every character.
    assert note.await_args.kwargs["text"] == long_text
