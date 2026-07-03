"""Unit tests for the Slack Bolt handlers (issue #3609, commit 4).

Two mock layers, no real cognee and no real Slack:

* Slack SDK — ``client.chat_getPermalink``, ``say``, ``ack`` are AsyncMocks and
  events/commands are plain dicts. This mirrors cognee's precedent for mocking
  external async SDK clients (``httpx.AsyncClient.post`` patched in
  cognee/tests/integration/infrastructure/session/test_tapes_cache_adapter.py;
  ``aiohttp.ClientSession.post`` in cognee/tests/test_telemetry.py).
* Buffer/adapter — an AsyncMock, so no cognee is touched.

No sockets are opened; slack_bolt is not imported by these tests.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from src.memory_adapter import Answer, Citation
from src.slack_app import (
    format_reply,
    handle_app_mention,
    handle_message_event,
    handle_recall_command,
    should_ingest,
)


def _permalink_client(permalink="https://slack.example/archives/C1/p1"):
    client = AsyncMock()
    client.chat_getPermalink.return_value = {"ok": True, "permalink": permalink}
    return client


def _fake_buffer():
    buffer = AsyncMock()
    buffer.answer.return_value = Answer(text="We shipped Friday.", citations=[])
    return buffer


# --------------------------------------------------------------------------- #
# message ingestion                                                           #
# --------------------------------------------------------------------------- #


def test_normal_message_is_ingested_with_metadata():
    buffer = _fake_buffer()
    client = _permalink_client("https://slack.example/archives/C1/p1700000000000100")
    event = {
        "type": "message",
        "channel": "C1",
        "ts": "1700000000.000100",
        "user": "U_alice",
        "team": "T1",
        "text": "we decided to ship on Friday",
    }

    ingested = asyncio.run(
        handle_message_event(
            event, client, buffer, opted_in={"C1"}, bot_user_id="U_bot", default_team_id="T0"
        )
    )

    assert ingested is True
    client.chat_getPermalink.assert_awaited_once_with(channel="C1", message_ts="1700000000.000100")
    buffer.add_message.assert_awaited_once()
    call = buffer.add_message.await_args
    ref = call.args[0]
    assert ref.channel_id == "C1"
    assert ref.team_id == "T1"
    assert call.kwargs["ts"] == "1700000000.000100"
    assert call.kwargs["text"] == "we decided to ship on Friday"
    assert call.kwargs["permalink"] == "https://slack.example/archives/C1/p1700000000000100"
    assert call.kwargs["author"] == "U_alice"


def test_bot_message_is_skipped():
    buffer = _fake_buffer()
    client = _permalink_client()
    event = {"channel": "C1", "ts": "1.0", "text": "beep", "bot_id": "B999"}

    ingested = asyncio.run(handle_message_event(event, client, buffer, opted_in={"C1"}))

    assert ingested is False
    buffer.add_message.assert_not_awaited()
    client.chat_getPermalink.assert_not_awaited()


def test_edited_message_is_skipped():
    buffer = _fake_buffer()
    client = _permalink_client()
    event = {"channel": "C1", "ts": "1.0", "text": "edited", "subtype": "message_changed"}

    ingested = asyncio.run(handle_message_event(event, client, buffer, opted_in={"C1"}))

    assert ingested is False
    buffer.add_message.assert_not_awaited()


def test_own_message_is_skipped():
    buffer = _fake_buffer()
    client = _permalink_client()
    event = {"channel": "C1", "ts": "1.0", "text": "hi", "user": "U_bot"}

    ingested = asyncio.run(
        handle_message_event(event, client, buffer, opted_in={"C1"}, bot_user_id="U_bot")
    )

    assert ingested is False
    buffer.add_message.assert_not_awaited()


def test_message_from_non_opted_in_channel_is_skipped():
    buffer = _fake_buffer()
    client = _permalink_client()
    event = {"channel": "C_other", "ts": "1.0", "text": "secret", "user": "U_alice"}

    ingested = asyncio.run(handle_message_event(event, client, buffer, opted_in={"C1"}))

    assert ingested is False
    buffer.add_message.assert_not_awaited()


def test_empty_text_message_is_skipped():
    buffer = _fake_buffer()
    client = _permalink_client()
    event = {"channel": "C1", "ts": "1.0", "text": "   ", "user": "U_alice"}

    ingested = asyncio.run(handle_message_event(event, client, buffer, opted_in={"C1"}))

    assert ingested is False


def test_ingestion_survives_permalink_failure():
    buffer = _fake_buffer()
    client = AsyncMock()
    client.chat_getPermalink.side_effect = RuntimeError("slack down")
    event = {"channel": "C1", "ts": "1.0", "text": "still ingest me", "user": "U_alice"}

    ingested = asyncio.run(handle_message_event(event, client, buffer, opted_in={"C1"}))

    assert ingested is True
    assert buffer.add_message.await_args.kwargs["permalink"] == ""


def test_should_ingest_predicate_direct():
    assert should_ingest({"channel": "C1", "text": "x", "user": "U"}, {"C1"}, "U_bot") is True
    assert (
        should_ingest({"channel": "C1", "text": "x", "subtype": "bot_message"}, {"C1"}, None)
        is False
    )


# --------------------------------------------------------------------------- #
# @mention answer                                                             #
# --------------------------------------------------------------------------- #


def test_app_mention_answers_extracted_question_and_replies():
    buffer = _fake_buffer()
    say = AsyncMock()
    event = {
        "channel": "C1",
        "team": "T1",
        "ts": "5.0",
        "text": "<@U0BOT123> what did we decide about the launch?",
    }

    asyncio.run(handle_app_mention(event, say, buffer, default_team_id="T0"))

    buffer.answer.assert_awaited_once()
    call = buffer.answer.await_args
    assert call.args[0].channel_id == "C1"
    assert call.kwargs["query"] == "what did we decide about the launch?"
    say.assert_awaited_once()
    assert "We shipped Friday." in say.await_args.args[0]


# --------------------------------------------------------------------------- #
# /recall slash command                                                       #
# --------------------------------------------------------------------------- #


def test_recall_command_acks_answers_and_replies():
    buffer = _fake_buffer()
    say = AsyncMock()
    ack = AsyncMock()
    command = {"channel_id": "C1", "team_id": "T1", "text": "who owns billing?"}

    asyncio.run(handle_recall_command(command, ack, say, buffer, default_team_id="T0"))

    ack.assert_awaited_once()
    buffer.answer.assert_awaited_once()
    call = buffer.answer.await_args
    assert call.args[0].channel_id == "C1"
    assert call.kwargs["query"] == "who owns billing?"
    say.assert_awaited_once()


# --------------------------------------------------------------------------- #
# reply rendering (minimal; commit 5 replaces with Block Kit)                 #
# --------------------------------------------------------------------------- #


def test_format_reply_includes_answer_and_ok_citation_link():
    answer = Answer(
        text="We shipped Friday.",
        citations=[
            Citation(
                channel_id="C1",
                ts="1.0",
                permalink="https://slack.example/x",
                author="alice",
                snippet="ship friday",
                ok=True,
            )
        ],
    )
    reply = format_reply(answer)
    assert "We shipped Friday." in reply
    assert "<https://slack.example/x|alice>" in reply


def test_format_reply_falls_back_for_stale_citation():
    answer = Answer(
        text="Answer.",
        citations=[
            Citation(
                channel_id="C1",
                ts="1.0",
                permalink="",
                author="",
                snippet="fallback text",
                ok=False,
            )
        ],
    )
    reply = format_reply(answer)
    assert "fallback text" in reply
    assert "<|" not in reply  # no broken link


def test_format_reply_handles_empty_answer():
    reply = format_reply(Answer(text="", citations=[]))
    assert reply  # non-empty placeholder, no crash


# --------------------------------------------------------------------------- #
# slack_bolt import isolation                                                 #
# --------------------------------------------------------------------------- #


def test_non_slack_modules_import_without_slack_bolt():
    # These import cleanly even though slack_bolt is not installed.
    import importlib

    for mod in (
        "src.memory_adapter",
        "src.citation_index",
        "src.cognee_memory",
        "src.ingestion_buffer",
        "src.config",
        "src.slack_app",  # imports without slack_bolt because the import is deferred
    ):
        assert importlib.import_module(mod) is not None


def test_build_app_raises_clear_error_when_slack_bolt_missing():
    # slack_bolt is not installed in the test env, so build_app must fail loudly
    # with an actionable message rather than at import time.
    import importlib.util

    if importlib.util.find_spec("slack_bolt") is not None:
        pytest.skip("slack_bolt is installed; import-isolation error path not exercised")

    from src.config import SlackSettings
    from src.slack_app import build_app

    with pytest.raises(ImportError, match="slack_bolt"):
        build_app(AsyncMock(), SlackSettings(bot_token="x", app_token="y"), set())
