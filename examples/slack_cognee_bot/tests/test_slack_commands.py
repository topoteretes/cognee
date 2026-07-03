"""Unit tests for the forget + opt-in/opt-out commands (issue #3609, commit 6).

Bolt (ack/say) and the buffer/adapter are mocked — no cognee, no keys, no
sockets. These tests also verify the opt-in set is a single source of truth
shared with the Commit-4 message handler.
"""

import asyncio
from unittest.mock import AsyncMock

from src.slack_app import (
    handle_forget_command,
    handle_message_event,
    handle_optin_command,
    handle_optout_command,
    should_ingest,
)


def _cmd(channel_id="C1", team_id="T1"):
    return {"channel_id": channel_id, "team_id": team_id, "text": ""}


# --------------------------------------------------------------------------- #
# /cognee-forget                                                              #
# --------------------------------------------------------------------------- #


def test_forget_command_deletes_channel_dataset_and_confirms():
    buffer = AsyncMock()
    ack = AsyncMock()
    say = AsyncMock()

    asyncio.run(handle_forget_command(_cmd("C1", "T1"), ack, say, buffer, default_team_id="T0"))

    ack.assert_awaited_once()
    buffer.forget.assert_awaited_once()
    ref = buffer.forget.await_args.args[0]
    assert ref.channel_id == "C1"
    assert ref.dataset_name == "slack_C1"

    say.assert_awaited_once()
    reply = say.await_args.args[0]
    assert "slack_C1" in reply  # names the deleted dataset


def test_forget_reply_surfaces_channel_level_limitation_honestly():
    buffer = AsyncMock()
    say = AsyncMock()
    asyncio.run(handle_forget_command(_cmd("C1"), AsyncMock(), say, buffer))

    reply = say.await_args.args[0].lower()
    assert "channel level" in reply
    assert "forget me" in reply  # explicitly states per-user forget is unsupported


# --------------------------------------------------------------------------- #
# /cognee-optin                                                               #
# --------------------------------------------------------------------------- #


def test_optin_adds_channel_and_posts_disclosure_first_time():
    opted_in: set[str] = set()
    say = AsyncMock()
    ack = AsyncMock()

    asyncio.run(handle_optin_command(_cmd("C1"), ack, say, opted_in))

    ack.assert_awaited_once()
    assert "C1" in opted_in
    reply = say.await_args.args[0]
    assert "remember" in reply.lower()  # disclosure describes what is recorded
    assert "optout" in reply.lower()  # tells the user how to stop


def test_optin_second_time_does_not_repeat_disclosure():
    opted_in = {"C1"}
    say = AsyncMock()

    asyncio.run(handle_optin_command(_cmd("C1"), AsyncMock(), say, opted_in))

    assert "C1" in opted_in
    assert "already" in say.await_args.args[0].lower()


# --------------------------------------------------------------------------- #
# /cognee-optout                                                              #
# --------------------------------------------------------------------------- #


def test_optout_removes_channel_and_confirms():
    opted_in = {"C1", "C2"}
    ack = AsyncMock()
    say = AsyncMock()

    asyncio.run(handle_optout_command(_cmd("C1"), ack, say, opted_in))

    ack.assert_awaited_once()
    assert "C1" not in opted_in
    assert "C2" in opted_in  # other channels untouched
    assert "forget" in say.await_args.args[0].lower()  # points at /cognee-forget


def test_optout_of_unknown_channel_is_safe():
    opted_in: set[str] = set()
    # discard() never raises on a missing key.
    asyncio.run(handle_optout_command(_cmd("C_missing"), AsyncMock(), AsyncMock(), opted_in))
    assert opted_in == set()


# --------------------------------------------------------------------------- #
# single source of truth: commands + message handler share one opt-in set     #
# --------------------------------------------------------------------------- #


def test_optin_optout_drive_the_same_set_the_message_handler_reads():
    opted_in: set[str] = set()
    buffer = AsyncMock()
    client = AsyncMock()
    client.chat_getPermalink.return_value = {"permalink": "https://slack.example/x"}
    event = {"channel": "C1", "ts": "1.0", "text": "hello", "user": "U_alice"}

    # Before opt-in: message is skipped.
    ingested = asyncio.run(handle_message_event(event, client, buffer, opted_in=opted_in))
    assert ingested is False
    buffer.add_message.assert_not_awaited()

    # Opt in via the command -> mutates the same set.
    asyncio.run(handle_optin_command(_cmd("C1"), AsyncMock(), AsyncMock(), opted_in))
    ingested = asyncio.run(handle_message_event(event, client, buffer, opted_in=opted_in))
    assert ingested is True
    buffer.add_message.assert_awaited_once()

    # Opt out -> the message handler now skips again.
    asyncio.run(handle_optout_command(_cmd("C1"), AsyncMock(), AsyncMock(), opted_in))
    ingested = asyncio.run(handle_message_event(event, client, buffer, opted_in=opted_in))
    assert ingested is False
    assert buffer.add_message.await_count == 1  # unchanged since opt-out


def test_should_ingest_reflects_live_optin_set():
    opted_in: set[str] = set()
    event = {"channel": "C1", "text": "x", "user": "U"}
    assert should_ingest(event, opted_in, None) is False
    opted_in.add("C1")
    assert should_ingest(event, opted_in, None) is True
