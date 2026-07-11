"""A Telegram chat-memory bot on the adapter. The whole platform layer is here.

This is the "thin bot" the adapter is designed for. It does two things only:
map a Telegram update to a ``Conversation`` + ``Message``, and wire commands to
the three adapter primitives. All memory behaviour (scoping, consent, citations,
forget) is inherited from the core, so the file stays ~100 lines and every other
platform bot looks just like it.

Unlike ``console_bot.py`` (which runs with no keys), this talks to real cognee
and real Telegram, so it needs:

    pip install "cognee[anthropic]" python-telegram-bot
    export LLM_API_KEY=...            # for cognee's graph memory
    export TELEGRAM_BOT_TOKEN=...     # from @BotFather
    python examples/integrations/chat_memory/telegram_bot.py

Commands: ``/ask <q>``, ``/forgetme``, ``/opt_in``, ``/opt_out``. Any other
message is remembered in the background.
"""

import os

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from cognee.integrations.chat_memory import (
    ChatMemoryAdapter,
    CogneeMemoryBackend,
    Conversation,
    Message,
    per_channel_scope,
)

# One adapter for the whole bot. Per-channel memory: a group chat's messages
# share one connected graph; use per_user_scope for a personal-brain bot.
memory = ChatMemoryAdapter(scope=per_channel_scope, backend=CogneeMemoryBackend())


def conversation_of(update: Update) -> Conversation:
    """Map a Telegram update to a Conversation. The only platform-specific bit."""
    chat = update.effective_chat
    user = update.effective_user
    return Conversation(
        platform="telegram",
        workspace="",  # Telegram has no workspace above the chat
        channel=str(chat.id),
        user=str(user.id),
        thread=str(update.message.message_thread_id) if update.message else None,
    )


async def on_message(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    convo = conversation_of(update)
    await memory.ingest(
        convo,
        Message(
            text=update.message.text or "",
            user=str(update.effective_user.id),
            timestamp=str(update.message.date.timestamp()),
            permalink=f"https://t.me/c/{update.effective_chat.id}/{update.message.message_id}",
        ),
    )


async def on_ask(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = " ".join(context.args)
    if not query:
        await update.message.reply_text("Usage: /ask <question>")
        return
    answer = await memory.answer(conversation_of(update), query)
    if answer.is_empty:
        await update.message.reply_text("I don't have anything on that yet.")
        return
    lines = [answer.text]
    if answer.citations:
        lines.append("\nSources:")
        lines += [f"• {c.permalink or c.text[:60]}" for c in answer.citations[:5]]
    await update.message.reply_text("\n".join(lines))


async def on_forget_me(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    result = await memory.forget(
        conversation=conversation_of(update), user=str(update.effective_user.id)
    )
    await update.message.reply_text(f"Forgotten {result['items_removed']} message(s) of yours.")


async def on_opt(update: Update, on: bool) -> None:
    memory.set_consent(str(update.effective_user.id), on)
    await update.message.reply_text("You're opted in." if on else "You're opted out.")


def main() -> None:
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("ask", on_ask))
    app.add_handler(CommandHandler("forgetme", on_forget_me))
    app.add_handler(CommandHandler("opt_in", lambda u, c: on_opt(u, True)))
    app.add_handler(CommandHandler("opt_out", lambda u, c: on_opt(u, False)))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))
    app.run_polling()


if __name__ == "__main__":
    main()
