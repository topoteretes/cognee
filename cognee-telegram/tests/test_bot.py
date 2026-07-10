"""Handler wiring with Telegram mocked via unittest.mock (PTB has no test harness)."""

from unittest.mock import AsyncMock, MagicMock

from cognee_telegram.adapter import Answer, CogneeMemoryAdapter
from cognee_telegram.bot import (
    INTRO,
    ask_command,
    forget_command,
    greet_on_join,
    ingest_message,
    optin_command,
    optout_command,
    render_answer,
    start_command,
)
from cognee_telegram.citations import MessageRef


def make_update(
    *, text=None, chat_id=7, chat_type="private", user_id=7, message_id=1, thread_id=None
):
    update = MagicMock()
    update.effective_chat.id = chat_id
    update.effective_chat.type = chat_type
    update.effective_user.id = user_id
    update.effective_user.full_name = "Ada"
    update.effective_user.username = "ada"
    message = MagicMock()
    message.message_id = message_id
    message.text = text
    message.message_thread_id = thread_id
    message.reply_text = AsyncMock()
    update.effective_message = message
    return update


def make_context(adapter, args=None):
    context = MagicMock()
    context.bot_data = {"adapter": adapter}
    context.args = args or []
    context.bot.send_message = AsyncMock()
    return context


# -- render_answer -------------------------------------------------------
def test_render_answer_empty():
    assert "don't have anything in memory" in render_answer(Answer(text=""))


def test_render_answer_with_deep_link_citation():
    answer = Answer(
        text="It's due Friday.",
        source_tag="graph",
        citations=[MessageRef(chat_id=-1001234567890, message_id=99, text="report due friday")],
    )
    out = render_answer(answer)
    assert "It's due Friday." in out
    assert "https://t.me/c/1234567890/99" in out
    assert "from graph memory" in out


def test_render_answer_quotes_when_no_public_link():
    answer = Answer(
        text="Lunch is at noon.",
        citations=[MessageRef(chat_id=7, message_id=1, text="lunch at noon")],
    )
    out = render_answer(answer)
    assert '"lunch at noon"' in out


def test_render_answer_forum_topic_deep_link():
    answer = Answer(
        text="It's Friday.",
        citations=[
            MessageRef(chat_id=-1001234567890, message_id=99, text="q3 friday", thread_id=12)
        ],
    )
    assert "https://t.me/c/1234567890/12/99" in render_answer(answer)


# -- handlers ------------------------------------------------------------
async def test_ingest_message_stores_to_memory(mock_cognee):
    adapter = CogneeMemoryAdapter()
    update = make_update(text="remember the wifi password is hunter2")
    await ingest_message(update, make_context(adapter))

    mock_cognee.remember.assert_awaited_once()
    args, kwargs = mock_cognee.remember.call_args
    assert args[0] == ["Ada: remember the wifi password is hunter2"]
    assert kwargs["dataset_name"] == "telegram_dm_7"


async def test_ingest_skips_when_opted_out(mock_cognee):
    adapter = CogneeMemoryAdapter()
    adapter.opt_out(7)
    update = make_update(text="should not be stored")
    await ingest_message(update, make_context(adapter))
    mock_cognee.remember.assert_not_awaited()


async def test_ask_command_replies_with_answer_and_sources(mock_cognee, graph_result):
    mock_cognee.recall.return_value = [graph_result("The wifi password is hunter2.")]
    adapter = CogneeMemoryAdapter()
    # seed the ledger so the citation resolves
    update_seed = make_update(text="the wifi password is hunter2", message_id=5)
    await ingest_message(update_seed, make_context(adapter))

    update = make_update(text="/ask")
    await ask_command(update, make_context(adapter, args=["what's", "the", "wifi", "password?"]))

    update.effective_message.reply_text.assert_awaited_once()
    reply = update.effective_message.reply_text.call_args.args[0]
    assert "hunter2" in reply


async def test_ask_without_query_shows_usage(mock_cognee):
    adapter = CogneeMemoryAdapter()
    update = make_update(text="/ask")
    await ask_command(update, make_context(adapter, args=[]))
    reply = update.effective_message.reply_text.call_args.args[0]
    assert "Usage" in reply
    mock_cognee.recall.assert_not_awaited()


async def test_ask_replies_friendly_error_on_backend_failure(mock_cognee):
    # Live test exposed: a rate-limited/failed backend must not leave /ask silent.
    mock_cognee.recall.side_effect = RuntimeError("rate limited")
    adapter = CogneeMemoryAdapter()
    update = make_update(text="/ask")
    await ask_command(update, make_context(adapter, args=["anything"]))
    reply = update.effective_message.reply_text.call_args.args[0]
    assert "couldn't reach the memory backend" in reply


async def test_forget_command_clears_memory(mock_cognee):
    adapter = CogneeMemoryAdapter()
    update = make_update(text="/forget")
    await forget_command(update, make_context(adapter))

    mock_cognee.forget.assert_awaited_once()
    _, kwargs = mock_cognee.forget.call_args
    assert kwargs["dataset"] == "telegram_dm_7"


async def test_optout_then_optin_toggles_capture(mock_cognee):
    adapter = CogneeMemoryAdapter()
    await optout_command(make_update(text="/optout"), make_context(adapter))
    assert adapter.is_opted_out(7) is True
    await optin_command(make_update(text="/optin"), make_context(adapter))
    assert adapter.is_opted_out(7) is False


async def test_start_command_sends_intro(mock_cognee):
    update = make_update(text="/start")
    await start_command(update, make_context(CogneeMemoryAdapter()))
    reply = update.effective_message.reply_text.call_args.args[0]
    assert "cognee memory bot" in reply


async def test_greet_on_join_posts_intro_when_added(mock_cognee):
    update = MagicMock()
    update.my_chat_member.old_chat_member.status = "left"
    update.my_chat_member.new_chat_member.status = "member"
    update.my_chat_member.chat.id = -1001234567890
    context = make_context(CogneeMemoryAdapter())
    await greet_on_join(update, context)
    context.bot.send_message.assert_awaited_once()
    _, kwargs = context.bot.send_message.call_args
    assert kwargs["chat_id"] == -1001234567890
    assert kwargs["text"] == INTRO


async def test_greet_on_join_silent_when_already_member(mock_cognee):
    update = MagicMock()
    update.my_chat_member.old_chat_member.status = "member"
    update.my_chat_member.new_chat_member.status = "administrator"
    context = make_context(CogneeMemoryAdapter())
    await greet_on_join(update, context)
    context.bot.send_message.assert_not_awaited()
