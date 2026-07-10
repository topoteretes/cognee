"""python-telegram-bot handlers — Telegram I/O only; memory lives in the adapter."""

from __future__ import annotations

import logging

from telegram import Update
from telegram.ext import (
    Application,
    ChatMemberHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from .adapter import Answer, CogneeMemoryAdapter
from .citations import MessageRef
from .config import Settings
from .scoping import Scope

logger = logging.getLogger(__name__)

_BACKEND_DOWN = (
    "⚠️ I couldn't reach the memory backend just now — it may be rate-limited, "
    "starting up, or missing an LLM key. Please try again in a moment."
)

INTRO = (
    "👋 I'm a cognee memory bot. I quietly remember what's shared here so you can ask "
    "about it later.\n\n"
    "• Just chat — I capture messages into this chat's private memory.\n"
    "• /ask <question> — I answer from this chat's memory, with sources.\n"
    "• /forget — wipe this chat's memory.\n"
    "• /optout — stop capturing here (/optin to resume).\n\n"
    "Nothing leaves this chat's own memory. An LLM is used to build and query it."
)

_EMPTY = (
    "I don't have anything in memory for this chat yet. Send a few messages or forward "
    "something, then try /ask again."
)


def _author(update: Update) -> str | None:
    user = update.effective_user
    if user is None:
        return None
    return user.full_name or user.username or str(user.id)


def _scope(adapter: CogneeMemoryAdapter, update: Update) -> Scope:
    chat = update.effective_chat
    user = update.effective_user
    message = update.effective_message
    thread_id = getattr(message, "message_thread_id", None) if message else None
    return adapter.scope_for(
        chat_type=chat.type, chat_id=chat.id, user_id=user.id, thread_id=thread_id
    )


def render_answer(answer: Answer) -> str:
    """Format a recall answer with tappable sources back to the original messages."""
    if not answer.text:
        return _EMPTY
    lines = [answer.text]
    if answer.citations:
        lines += ["", "Sources:"]
        for ref in answer.citations:
            label = ref.text if len(ref.text) <= 60 else ref.text[:59] + "…"
            link = ref.deep_link()
            lines.append(f"• {label} — {link}" if link else f'• "{label}"')
    return "\n".join(lines)


# -- handlers ------------------------------------------------------------
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.effective_message.reply_text(INTRO)


async def ingest_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    adapter: CogneeMemoryAdapter = context.bot_data["adapter"]
    message = update.effective_message
    if message is None or not message.text:
        return
    scope = _scope(adapter, update)
    if adapter.is_opted_out(scope.chat_id):
        return
    ref = MessageRef(
        chat_id=scope.chat_id,
        message_id=message.message_id,
        text=message.text,
        thread_id=scope.thread_id,
        author=_author(update),
    )
    try:
        await adapter.ingest(scope, ref)
    except Exception:
        # Capture is passive — log and move on rather than nagging the chat.
        logger.exception("ingest failed for chat %s", scope.chat_id)


async def ask_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    adapter: CogneeMemoryAdapter = context.bot_data["adapter"]
    query = " ".join(context.args).strip() if context.args else ""
    if not query:
        await update.effective_message.reply_text("Usage: /ask <your question>")
        return
    scope = _scope(adapter, update)
    try:
        answer = await adapter.answer(scope, query)
    except Exception:
        logger.exception("ask failed for chat %s", scope.chat_id)
        await update.effective_message.reply_text(_BACKEND_DOWN)
        return
    await update.effective_message.reply_text(render_answer(answer), disable_web_page_preview=True)


async def forget_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    adapter: CogneeMemoryAdapter = context.bot_data["adapter"]
    scope = _scope(adapter, update)
    try:
        await adapter.forget(scope)
    except Exception:
        logger.exception("forget failed for chat %s", scope.chat_id)
        await update.effective_message.reply_text(_BACKEND_DOWN)
        return
    await update.effective_message.reply_text("🧹 Cleared this chat's memory (graph + vectors).")


async def optout_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    adapter: CogneeMemoryAdapter = context.bot_data["adapter"]
    adapter.opt_out(update.effective_chat.id)
    await update.effective_message.reply_text(
        "🔕 Paused capturing in this chat. Existing memory is kept — use /forget to clear it, "
        "or /optin to resume."
    )


async def optin_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    adapter: CogneeMemoryAdapter = context.bot_data["adapter"]
    adapter.opt_in(update.effective_chat.id)
    await update.effective_message.reply_text("🔔 Resumed capturing in this chat.")


async def greet_on_join(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Post the calm intro once, when the bot is added to a group."""
    member = update.my_chat_member
    if member is None:
        return
    was_in = member.old_chat_member.status in {"member", "administrator"}
    now_in = member.new_chat_member.status in {"member", "administrator"}
    if not was_in and now_in:
        await context.bot.send_message(chat_id=member.chat.id, text=INTRO)


def build_application(
    settings: Settings, adapter: CogneeMemoryAdapter | None = None
) -> Application:
    """Wire the adapter and handlers into a python-telegram-bot Application."""
    adapter = adapter or CogneeMemoryAdapter()
    # concurrent_updates keeps the bot responsive: a slow ingest (cognify) on one
    # message won't block /ask or other commands.
    app = Application.builder().token(settings.bot_token).concurrent_updates(True).build()
    app.bot_data["adapter"] = adapter

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", start_command))
    app.add_handler(CommandHandler("ask", ask_command))
    app.add_handler(CommandHandler("forget", forget_command))
    app.add_handler(CommandHandler("optout", optout_command))
    app.add_handler(CommandHandler("optin", optin_command))
    app.add_handler(ChatMemberHandler(greet_on_join, ChatMemberHandler.MY_CHAT_MEMBER))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, ingest_message))
    app.add_error_handler(_on_error)
    return app


async def _on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Last-resort handler so no exception goes completely unnoticed."""
    logger.exception("Unhandled handler error", exc_info=context.error)
