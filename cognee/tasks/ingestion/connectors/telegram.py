# Adapted from topoteretes/cognee-community (Apache-2.0), PR #355, commit 76ff1a4.
# Modified to ship in the Cognee SDK with SDK-local imports and test paths.
"""Telegram connector for cognee: a ``dlt`` source that turns the messages a bot sees into memory.

One source is one bot token. It yields one document per message of the selected chats and is
meant to be handed directly to :func:`cognee.remember`::

    import cognee
    from cognee.tasks.ingestion.connectors import telegram_source

    await cognee.remember(
        telegram_source(token="123456789:AA...", chats=[-1001234567890, "@my_channel"]),
        dataset_name="telegram",
        primary_key="id",
        write_disposition="merge",   # REQUIRED, see below
        max_rows_per_table=0,        # compare forget-on-delete against every row
    )

Design
------
* **Auth** is a bot token from @BotFather, passed as ``token=``. There is no OAuth for bots. The
  token's format is checked when the source is built, so a pasted fragment never reaches a table
  or state name, and ``getMe`` checks the token itself at the start of every run. The token is
  part of every request URL, so errors are raised outside the ``except`` that caught the HTTP
  error: neither ``__cause__`` nor ``__context__`` carries the URL.
* **One page per run.** Asking ``getUpdates`` for the next page confirms the previous one on
  Telegram's side, and confirmed updates are gone for good. A run makes one ``getUpdates`` call
  (up to 100 updates), and those updates are confirmed only by the *next* run, whose offset comes
  from the state dlt saved with this run's rows. A host revoking access mid-run saves no state,
  so the next run reads the same page; a load that fails keeps dlt's package (rows and cursor),
  which the next run of the pipeline loads first. A run that read a full page reports
  ``backlog=1``.
* **Incremental cursor** is the Bot API ``update_id`` in dlt resource state. After a week without
  updates Telegram may restart ``update_id`` below the saved cursor, so after six quiet days the
  connector asks without an offset (which confirms nothing). Telegram drops updates nobody
  collected within about 24 hours; a run that comes more than 24 hours after the last update it
  saw and finds a hole in ``update_id`` counts the missing ids in ``gap`` (possible loss, since
  Telegram also skips ids of update types this bot does not ask for).
* **Edits** (``edited_message`` / ``edited_channel_post``) carry the same ``message_id`` and upsert
  the row (``merge`` on ``chat_id:message_id``). An edit that leaves no text (a caption removed)
  tombstones the stored row.
* **Forget-on-delete.** The Bot API sends no event when a message is deleted. It sends
  ``my_chat_member`` with status ``left``/``kicked`` (or ``restricted`` and no longer a member)
  when the bot is removed from a group, a supergroup or a channel; every message stored for that
  chat is then emitted as a ``_deleted`` row. A private chat is never forgotten this way: there ``kicked`` means the user blocked the
  bot. Narrowing ``chats`` (or turning ``include_private`` off) forgets the chats that left the
  selection on the next run. Which messages are stored is read from the resource's own staging
  table (the rows loaded so far), once per run and only when a tombstone needs it, so the state
  does not grow with the corpus.
* **Group upgrades** give a chat a new id (``migrate_to_chat_id``). Old rows keep their key; the
  chat keeps syncing under the new id, both ids count as one chat for the selection, for removal
  and for the node set, which names the oldest id of the chain.
* **Several bots** in one chat see the same keys, so each bot needs its own ``resource_name``
  (``telegram_<bot id>`` by default). A sync state written by another bot is refused.

.. important::
   ``write_disposition="merge"`` is mandatory. The add pipeline defaults to ``"replace"``, which
   would wipe every message from earlier syncs on the second run, because ``getUpdates`` only
   returns *new* updates, and Telegram never sends an old one again.
"""

import http.client
import json
import math
import re
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Iterator
from datetime import datetime, timezone
from typing import Any

from cognee.shared.logging_utils import get_logger
from cognee.tasks.ingestion import dlt_utils
from cognee.tasks.ingestion.dlt_utils import (
    DOCUMENT_SOURCE_ATTR,
    NODE_SET_COLUMN,
    STRUCTURE_COLUMN,
    with_retries,
)

logger = get_logger("telegram_connector")

TELEGRAM_SOURCE_NAME = "telegram"
# Version of the resource state; a later release reads it to migrate an older layout.
STATE_VERSION = 1
# A separator line before the message text: everything after the first one is the text, so a
# member cannot write a line that reads as another header.
TEXT_SEPARATOR = "--- message ---"

_API_BASE = "https://api.telegram.org"
# The Bot API's own range for getUpdates ``limit`` ("Values between 1-100").
_PAGE_SIZE = 100
# Attempts per request on a 429 or a 5xx or no answer. Our own choice: it only delays a run.
_ATTEMPTS = 5
# Seconds one run may sleep between retries, all requests together. Our own choice (OneNote
# also waits at most 60 s a run): the walk holds the process-wide staging lock while it sleeps.
# A wait Telegram dictates (``retry_after``) that does not fit cuts the run with
# ``failed_rate_limit`` instead; it only delays, the next run reads the same page.
_WAIT_BUDGET_SECONDS = 60.0
# Socket timeout of one request. Our own choice, Telegram documents no latency.
_TIMEOUT_SECONDS = 30.0
# ``update_id`` may restart at a random value after a week without updates.
_QUIET_RESET_SECONDS = 6 * 24 * 3600
# Telegram keeps undelivered updates for about a day: "they will not be kept longer than 24 hours"
# (core.telegram.org/bots/api, Getting updates). Nothing newer than that can have been dropped.
_UPDATE_RETENTION_SECONDS = 24 * 3600
# Display length of a title; the full first line stays in the content.
_TITLE_LENGTH = 80
# Display length of a quoted reply; the quoted message is its own document.
_QUOTE_LENGTH = 200
# A bot token as @BotFather issues it: the bot's numeric id, a colon and the secret.
_TOKEN_RE = re.compile(r"\d{1,20}:[A-Za-z0-9_-]{20,}")
# dlt snake-cases table names, so only names it leaves unchanged are accepted.
_RESOURCE_NAME_RE = re.compile(r"[a-z][a-z0-9_]{0,62}")
_CHAT_ID_RE = re.compile(r"-?\d+")
_USERNAME_RE = re.compile(r"@?[A-Za-z][A-Za-z0-9_]*")
_ALLOWED_UPDATES = [
    "message",
    "edited_message",
    "channel_post",
    "edited_channel_post",
    "my_chat_member",
]
_MESSAGE_KEYS = ("message", "edited_message", "channel_post", "edited_channel_post")
_EDIT_KEYS = ("edited_message", "edited_channel_post")
_REMOVED_STATUSES = {"left", "kicked"}
# A restricted member with ``is_member`` false is out of the chat too.
_RESTRICTED = "restricted"
# What a message is besides text: the Message fields of Bot API 10.5 that say so
# (core.telegram.org/bots/api#message, read 2026-11-24). Of all of it only a caption is read,
# labelled with the first kind listed here that the message carries: ``animation`` comes before
# ``document``, which Telegram sets on every animation too. A kind Telegram adds later counts as
# a message without content until it is listed here.
_CONTENT_KINDS = (
    "animation",
    "audio",
    "document",
    "live_photo",
    "paid_media",
    "photo",
    "sticker",
    "story",
    "video",
    "video_note",
    "voice",
    "widget",
    "rich_message",
    "checklist",
    "contact",
    "dice",
    "game",
    "poll",
    "venue",
    "location",
    "invoice",
    "giveaway",
    "giveaway_winners",
)


# ---------------------------------------------------------------------------
# Errors: messages carry the method and the status only, never the token or a chat.
# ---------------------------------------------------------------------------
class TelegramSourceError(RuntimeError):
    """Base class of the errors this source raises."""


class TelegramAPIError(TelegramSourceError):
    """An error answer from the Bot API (``status``), or no answer at all (``status`` 0).

    Telegram's own ``description`` is kept on the error for the caller, never in the message.
    """

    def __init__(self, message: str, *, status: int = 0, description: str = ""):
        super().__init__(message)
        self.status = status
        self.description = description


class TelegramAuthError(TelegramAPIError):
    """Telegram rejected the bot token (HTTP 401 or 404)."""


class TelegramRateLimitError(TelegramAPIError):
    """Telegram answered HTTP 429; ``retry_after`` is the wait it named, when it named a usable
    one (``math.inf`` for a wait beyond the largest float, ``None`` for NaN or a negative)."""

    def __init__(self, message: str, *, retry_after: float | None = None, description: str = ""):
        super().__init__(message, status=429, description=description)
        self.retry_after = retry_after


class TelegramConflictError(TelegramAPIError):
    """``getUpdates`` is unavailable (409): a webhook is set, or another process polls."""


class TelegramWebhookConflictError(TelegramConflictError):
    """``getUpdates`` is unavailable because a webhook is set on the bot."""


class TelegramUnavailableError(TelegramAPIError):
    """Telegram answered a 5xx, or did not answer."""


class TelegramBotMismatchError(TelegramSourceError):
    """The sync state of this ``resource_name`` was written by another bot."""


class TelegramUnexpectedError(TelegramSourceError):
    """A run failed on something this source did not expect, a defect of its own rather than
    Telegram's answer (``__cause__`` is the original), counted as ``failed_unexpected``."""


class TelegramStagingError(TelegramSourceError):
    """The ids of the stored messages could not be read from the staging table."""


def new_stats() -> dict[str, int]:
    """The counts of a run, every key present from its start so a host finds them on any exit.

    ``scanned`` message rows emitted, ``deleted`` tombstones emitted, ``skipped`` message and
    unread updates that became no row (a ``my_chat_member`` update is never counted), always the
    sum of ``skipped_unselected`` (a chat outside the selection, private chats included unless
    ``include_private``), ``skipped_unsupported_type`` (content this source does not read: a
    file, a sticker, a poll, a location and every other kind but text, unless it has a caption),
    ``skipped_empty_content`` (a message with no content: a service message, a blank text) and
    ``skipped_unread_update`` (an update type this source does not read), ``backlog`` 1 when the
    run read a full page, ``gap`` update ids that may have expired at Telegram before this run
    (see :func:`count_gap`), ``privacy_mode`` 1 when ``getMe`` said the bot sees only commands
    and replies in groups. ``privacy_mode`` means something only on a run with ``failed == 0``:
    a run that failed before or at ``getMe``, or on a bot mismatch, leaves it 0, and so does a
    ``can_read_all_group_messages`` that is anything but JSON ``false``.
    ``failed`` is 1 when the run stopped or raised, with one reason key
    set: ``failed_unexpected`` for a failure that is not Telegram's (a defect of this source),
    ``failed_provider`` for any other failure without a reason of its own, a failed staging read
    included.
    """
    return dict.fromkeys(
        (
            "scanned",
            "deleted",
            "skipped",
            "failed",
            "failed_auth",
            "failed_rate_limit",
            "failed_conflict",
            "failed_unavailable",
            "failed_provider",
            "failed_unexpected",
            "failed_bot_mismatch",
            "backlog",
            "gap",
            "skipped_unselected",
            "skipped_unsupported_type",
            "skipped_empty_content",
            "skipped_unread_update",
            "privacy_mode",
        ),
        0,
    )


# ---------------------------------------------------------------------------
# Transport (stdlib only)
# ---------------------------------------------------------------------------
class _RefuseRedirects(urllib.request.HTTPRedirectHandler):
    """A 3xx is an error answer, never a new target: the token is in the URL."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class _WaitBudget:
    """The seconds a run may still sleep between retries."""

    def __init__(self, seconds: float = _WAIT_BUDGET_SECONDS):
        self.left = seconds

    def take(self, seconds: float) -> float | None:
        """``seconds`` when they fit (and spend them), else ``None``."""
        if seconds > self.left:
            return None
        self.left -= seconds
        return seconds


def _check_token(token: Any) -> str:
    """The token, stripped, or ``ValueError``. The message never echoes the value."""
    value = token.strip() if isinstance(token, str) else ""
    if not _TOKEN_RE.fullmatch(value):
        raise ValueError(
            "The Telegram bot token has an invalid format: expected the bot id, a colon and the "
            "secret, as @BotFather issues it."
        )
    return value


class TelegramBotClient:
    """Minimal Bot API client: ``call(method, params) -> result``.

    A ``service`` handed to :func:`telegram_source` instead must offer the same ``call`` and run
    ``check_active`` before each attempt it makes.
    """

    def __init__(self, token: str, *, sleep: Callable[[float], None] = time.sleep):
        self._token = _check_token(token)
        self._sleep = sleep
        self._opener = urllib.request.build_opener(_RefuseRedirects())

    def __repr__(self) -> str:
        return "TelegramBotClient(token=<redacted>)"

    @property
    def bot_id(self) -> str:
        """The bot's public numeric id, the part of the token before the colon."""
        return self._token.split(":", 1)[0]

    def call(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        check_active: Callable[[], None] | None = None,
        budget: _WaitBudget | None = None,
    ) -> Any:
        """The ``result`` of a successful answer, retrying a 429, a 5xx or no answer.

        ``check_active`` runs before each attempt, a retry included, and before each wait
        (``with_retries``); what it raises propagates as it is. Waits come out of ``budget`` (a
        fresh one per call when none is given); a wait that does not fit raises the last error.
        """
        body = json.dumps({k: v for k, v in (params or {}).items() if v is not None}).encode()
        waits = budget if budget is not None else _WaitBudget()

        def attempt() -> Any:
            status, raw = self._send(method, body)
            return self._answer(method, status, raw)

        if check_active is not None:
            check_active()
        return with_retries(
            attempt,
            retry_on=(TelegramRateLimitError, TelegramUnavailableError),
            delay=lambda error, number: waits.take(_retry_wait(error, number)),
            attempts=_ATTEMPTS,
            check_active=check_active,
            sleep=self._sleep,
        )

    def _open(self, request: urllib.request.Request):
        return self._opener.open(request, timeout=_TIMEOUT_SECONDS)

    def _send(self, method: str, body: bytes) -> tuple[int, bytes]:
        """``(status, body)``; status 0 when nothing came back. Never raises a transport error,
        so the typed error is raised later, where the HTTPError (its URL holds the token) is no
        one's ``__context__``."""
        request = urllib.request.Request(
            f"{_API_BASE}/bot{self._token}/{method}",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with self._open(request) as response:
                return 200, response.read()
        except urllib.error.HTTPError as exc:
            return exc.code, _error_body(exc)
        except (OSError, http.client.HTTPException):
            # URLError, timeouts, resets, RemoteDisconnected, IncompleteRead.
            return 0, b""

    def _answer(self, method: str, status: int, raw: bytes) -> Any:
        payload = _json_object(raw)
        description = str(payload.get("description") or "").replace(self._token, "[token]")
        description = description[:_QUOTE_LENGTH]
        if status == 0:
            raise TelegramUnavailableError(f"Telegram {method}: no answer")
        if status == 200:
            if payload.get("ok") is not True:
                raise TelegramAPIError(
                    f"Telegram {method}: HTTP 200 without a successful JSON answer",
                    status=200,
                    description=description,
                )
            return payload.get("result")
        if status == 409 and "webhook" in description.lower():
            raise TelegramWebhookConflictError(
                f"Telegram {method}: HTTP 409, this bot has a webhook set, so getUpdates is "
                "unavailable. Use a dedicated bot for cognee, or remove the webhook yourself "
                "(deleteWebhook) if it is not in use.",
                status=409,
                description=description,
            )
        if status == 409:
            raise TelegramConflictError(
                f"Telegram {method}: HTTP 409, another process is polling this bot with "
                "getUpdates. Stop it, or use a dedicated bot for cognee.",
                status=409,
                description=description,
            )
        if status in (401, 404):
            raise TelegramAuthError(
                f"Telegram {method}: HTTP {status}, the bot token was rejected. Create one with "
                "@BotFather and pass it as token=.",
                status=status,
                description=description,
            )
        if status == 429:
            raise TelegramRateLimitError(
                f"Telegram {method}: HTTP 429",
                retry_after=_retry_after(payload),
                description=description,
            )
        if status >= 500:
            raise TelegramUnavailableError(
                f"Telegram {method}: HTTP {status}", status=status, description=description
            )
        raise TelegramAPIError(
            f"Telegram {method}: HTTP {status}", status=status, description=description
        )


def _error_body(exc: urllib.error.HTTPError) -> bytes:
    try:
        return exc.read()
    except (OSError, http.client.HTTPException):
        return b""


def _json_object(raw: bytes) -> dict:
    """The JSON object in ``raw``, or ``{}`` (a proxy's error page, a cut-off reply)."""
    try:
        value = json.loads(raw.decode("utf-8")) if raw else {}
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def _retry_after(payload: dict) -> float | None:
    value = (payload.get("parameters") or {}).get("retry_after")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    # NaN and a negative wait are no wait, so the backoff applies. Anything beyond the largest
    # float (Infinity included) is longer than any budget, so the run stops on it.
    if not 0 <= value:
        return None
    return float(value) if value <= sys.float_info.max else math.inf


def _retry_wait(error: BaseException, number: int) -> float:
    """Telegram's ``retry_after`` on a 429, else a backoff of 1, 2, 4, 8 seconds."""
    retry_after = getattr(error, "retry_after", None)
    return retry_after if retry_after is not None else float(2**number)


def build_telegram_service(bot_token: str) -> TelegramBotClient:
    """Wrap a bot token in a client, the way Apollo wraps its API key.

    A bot token is not an OAuth access token: Telegram has no OAuth for bots.
    """
    return TelegramBotClient(bot_token)


def telegram_account_id(service: Any, *, check_active: Callable[[], None] | None = None) -> str:
    """The bot's id (``getMe``): the account a host keys a connection by.

    It is the id the source checks its state against, so a host that keys the connection by it
    gives a second bot its own table and state, instead of a table whose state belongs to the
    first bot (which the source refuses with ``failed_bot_mismatch`` on every run).
    """
    return _bot_id(service.call("getMe", check_active=check_active))


def _bot_id(me: Any) -> str:
    """The bot id of a ``getMe`` answer, read one way wherever it is read."""
    bot_id = me.get("id") if isinstance(me, dict) else None
    if isinstance(bot_id, bool) or not isinstance(bot_id, int):
        raise TelegramAPIError("Telegram getMe: HTTP 200 without the bot's id", status=200)
    return str(bot_id)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def row_key(chat_id: int, message_id: int) -> str:
    """Primary key of a message row: message ids are only unique inside a chat."""
    return f"{chat_id}:{message_id}"


def _one_line(value: Any) -> str:
    """User-controlled text as one line, so it cannot add a header line of its own."""
    return " ".join(str(value or "").split())


def _person_name(person: dict) -> str:
    return _one_line(" ".join(p for p in (person.get("first_name"), person.get("last_name")) if p))


def _chat_label(chat: dict) -> str:
    """A chat's name. A private chat is the person's name, never their handle or user id."""
    if chat.get("type") == "private":
        return _person_name(chat) or "private chat"
    return (
        _one_line(chat.get("title"))
        or (f"@{_one_line(chat['username'])}" if chat.get("username") else "")
        or _person_name(chat)
        or str(chat.get("id"))
    )


def _user_name(user: dict) -> str:
    """A person's display name; the handle goes on its own line. Never the numeric user id."""
    return _person_name(user) or _one_line(user.get("username")) or "unknown"


def _author(message: dict) -> str:
    """Who wrote the message. Channel posts have no ``from``: the channel is the author."""
    if message.get("from"):
        return _user_name(message["from"])
    if message.get("sender_chat"):
        return _chat_label(message["sender_chat"])
    return _chat_label(message.get("chat") or {})


def _forwarded_from(message: dict) -> str | None:
    """Original author of a forwarded message, so it is not credited to the forwarder."""
    origin = message.get("forward_origin")
    if not origin:
        return None
    kind = origin.get("type")
    if kind == "user" and origin.get("sender_user"):
        return _user_name(origin["sender_user"])
    if kind == "hidden_user":
        return _one_line(origin.get("sender_user_name")) or "a hidden user"
    if kind == "chat" and origin.get("sender_chat"):
        return _chat_label(origin["sender_chat"])
    if kind == "channel" and origin.get("chat"):
        return _chat_label(origin["chat"])
    return "unknown"


def _message_url(chat: dict, message_id: int) -> str | None:
    """t.me link: public for chats with a username, members-only for private supergroups
    and channels (``t.me/c/<id>``). Basic groups and private chats have no message links."""
    if chat.get("username") and chat.get("type") != "private":
        return f"https://t.me/{chat['username']}/{message_id}"
    chat_id = str(chat.get("id", ""))
    if chat.get("type") in ("supergroup", "channel") and chat_id.startswith("-100"):
        return f"https://t.me/c/{chat_id[4:]}/{message_id}"
    return None


def _content_kind(message: dict) -> str | None:
    """The kind of content a message carries besides text, named as its Bot API field."""
    return next((kind for kind in _CONTENT_KINDS if message.get(kind) is not None), None)


def _message_text(message: dict) -> str:
    """Text of a message, or its caption labelled with the kind it is on; empty when neither
    says anything, so a blank caption never becomes a document of its label alone."""
    if message.get("text"):
        return message["text"]
    caption = message.get("caption")
    if not caption or not caption.strip():
        return ""
    kind = _content_kind(message)
    return f"[{kind}] {caption}" if kind else caption


def _iso(ts: int | None) -> str:
    if not ts:
        return ""
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _title(text: str, chat: dict, message_id: int) -> str:
    """``<first line> (<chat> #<message id>)``: the first line says what the message is about,
    the chat and message id make the title unique, so two messages that both say "ok" are
    two sources in a search's references. Only the first line is shortened; the full text
    stays in the content. A private chat is named by the person's name, never their handle.
    ``text`` is never blank (``message_to_row`` returns no row for it)."""
    first = next(_one_line(line) for line in text.splitlines() if line.strip())
    if len(first) > _TITLE_LENGTH:
        first = first[: _TITLE_LENGTH - 1].rstrip() + "…"
    return f"{first} ({_chat_label(chat)} #{message_id})"


def message_to_row(message: dict, root_chat_id: int | None = None) -> dict[str, Any] | None:
    """Render one Bot API message as a document row, or ``None`` when it has no text.

    The row is deterministic (no fetch timestamps), so an unchanged message keeps its content
    hash. ``root_chat_id`` is the oldest id of the chat's upgrade chain (the chat's own id when
    omitted); it names the node set and the chat container.
    """
    text = _message_text(message)
    if not text.strip():
        return None  # service messages (joins, pins, ...) and any content without a caption

    # Every Bot API message has a chat with an id and a message_id.
    chat = message["chat"]
    message_id: int = message["message_id"]
    root = str(chat["id"] if root_chat_id is None else root_chat_id)
    chat_name = _chat_label(chat)
    lines = [f"Chat: {chat_name}", f"From: {_author(message)}"]
    sender = message.get("from") or {}
    if sender.get("username"):
        lines.append(f"Handle: @{_one_line(sender['username'])}")
    lines.append(f"Date: {_iso(message.get('date'))}")
    forwarded_from = _forwarded_from(message)
    if forwarded_from:
        lines.append(f"Forwarded from: {forwarded_from}")
    if message.get("edit_date"):
        lines.append(f"Edited: {_iso(message['edit_date'])}")
    reply = message.get("reply_to_message")
    if reply:
        quoted = _one_line(_message_text(reply))
        lines.append(
            f"In reply to {_author(reply)}: {quoted[:_QUOTE_LENGTH]}"
            if quoted
            else f"In reply to message {reply.get('message_id')}"
        )
    lines += [TEXT_SEPARATOR, text]

    return {
        "id": row_key(chat["id"], message_id),
        "title": _title(text, chat, message_id),
        "content": "\n".join(lines),
        "url": _message_url(chat, message_id),
        "_deleted": False,
        NODE_SET_COLUMN: [f"telegram:chat:{root}"],
        STRUCTURE_COLUMN: {"ancestors": [{"kind": "chat", "id": root, "name": chat_name}]},
    }


def deleted_row(key: str) -> dict[str, Any]:
    """Hard-delete marker for a stored message. A tombstone carries no node set."""
    return {"id": key, "_deleted": True}


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------
class _Selection:
    """Which chats a run ingests. ``ids is None`` means every chat the bot is in."""

    def __init__(self, ids: list[int] | None, usernames: list[str], include_private: bool):
        self.ids = ids
        self.usernames = usernames
        self.include_private = include_private

    def stored(self) -> dict[str, Any]:
        """The selection as it is kept in state, so a later run can tell it changed."""
        chats = None if self.ids is None else {"ids": self.ids, "usernames": self.usernames}
        return {"chats": chats, "include_private": self.include_private}

    def selects(
        self,
        chat_id: int,
        migrated_from: dict[str, str],
        meta: dict[str, dict],
        bound: dict[str, list[str]],
    ) -> bool:
        """Whether the chat is selected. A chat and the chats it was upgraded from or to count as
        one; a username selects the chat it is bound to (``bind``), whatever it is called now."""
        if self.ids is None:
            return self.include_private or meta.get(str(chat_id), {}).get("type") != "private"
        root = _root(chat_id, migrated_from)
        chosen = [*self.ids, *(int(key) for name in self.usernames for key in bound.get(name, ()))]
        return any(_root(key, migrated_from) == root for key in chosen)

    def bind(self, bound: dict[str, list[str]], chat: dict) -> None:
        """Bind a selected username to the chat that carries it, the first time one is seen.

        From then on the username names that chat: a chat may change its username, and another
        chat may take the old one, which does not move the selection to it. A private chat is
        never bound (it is selected by id only, its username is a person's handle).
        """
        username = str(chat.get("username") or "").lower()
        if chat.get("type") == "private" or username not in self.usernames or bound.get(username):
            return
        bound[username] = [str(chat["id"])]

    def bound_usernames(self, state: dict) -> dict[str, list[str]]:
        """The saved bindings of the usernames still selected. A selected username with no saved
        binding is bound to the chat the saved chat meta knows under that name (a new entry of
        ``chats``, or a state written before bindings were kept), only when exactly one chat
        carries it: meta written before a chat released the name may still hold it, and a
        selection never pulls in a chat on a stale name. Read before the page updates the meta,
        so a rename in the page is not missed."""
        saved = state.get("username_chats", {})
        meta = state.get("chats_meta", {})
        bound = {}
        for name in self.usernames:
            if saved.get(name):
                bound[name] = list(saved[name])
                continue
            holders = [key for key, info in meta.items() if info.get("username") == name]
            if len(holders) == 1:
                bound[name] = holders
        return bound


def _selection(chats: Any, include_private: Any) -> _Selection:
    """``chats`` validated: ``None``, or a non-empty list or tuple of chat ids and @usernames.

    A bare string would be read as one username per letter and select nothing, so it is refused.
    """
    if not isinstance(include_private, bool):
        raise ValueError("include_private must be True or False")
    if chats is None:
        return _Selection(None, [], include_private)
    if not isinstance(chats, (list, tuple)) or not chats:
        raise ValueError(
            "chats must be None (every chat) or a non-empty list or tuple, like "
            '[-1001234567890, "@my_channel"].'
        )
    ids, usernames = set(), set()
    for chat in chats:
        value = chat.strip() if isinstance(chat, str) else chat
        if isinstance(value, int) and not isinstance(value, bool):
            ids.add(value)
        elif isinstance(value, str) and _CHAT_ID_RE.fullmatch(value):
            ids.add(int(value))
        elif isinstance(value, str) and _USERNAME_RE.fullmatch(value):
            usernames.add(value.lstrip("@").lower())
        else:
            raise ValueError("Each entry of chats must be a chat id or an @username.")
    return _Selection(sorted(ids), sorted(usernames), include_private)


def _root(chat_id: int | str, migrated_from: dict[str, str]) -> int:
    """The oldest id of a chat's upgrade chain (a cycle stops the walk)."""
    current, seen = str(chat_id), set()
    while current in migrated_from and current not in seen:
        seen.add(current)
        current = migrated_from[current]
    return int(current)


def _link_upgrades(updates: list, migrated_from: dict[str, str]) -> None:
    """Link the ids of every group upgrade the page announces, before any message is read.

    Telegram announces an upgrade twice: ``migrate_to_chat_id`` in the old group and
    ``migrate_from_chat_id`` in the new supergroup; either one links the ids. Read first, a
    supergroup message that comes before the announcement in the same page still counts as the
    old chat, for the selection and for the node set.
    """
    for update in updates:
        message = next((update[key] for key in _MESSAGE_KEYS if update.get(key)), None)
        if message is None:
            continue
        chat_id = message["chat"]["id"]
        if message.get("migrate_to_chat_id"):
            migrated_from[str(message["migrate_to_chat_id"])] = str(chat_id)
        if message.get("migrate_from_chat_id"):
            migrated_from[str(chat_id)] = str(message["migrate_from_chat_id"])


def _page_chats(updates: list) -> Iterator[dict]:
    """The chat of every update of the page, in order: messages, posts, edits, membership."""
    for update in updates:
        member = update.get("my_chat_member")
        if member:
            yield member["chat"]
            continue
        message = next((update[key] for key in _MESSAGE_KEYS if update.get(key)), None)
        if message is not None:
            yield message["chat"]


def _remember_chat(meta: dict[str, dict], chat: dict) -> None:
    """Keep a chat's type, and its username unless it is a private chat (a person's handle).

    A username belongs to one chat at a time, so the chat seen claiming it takes it from every
    other chat's meta: a chat that released the name is not known under it any more.
    """
    key = str(chat["id"])
    info = {"type": str(chat.get("type") or "")}
    if chat.get("username") and chat.get("type") != "private":
        info["username"] = str(chat["username"]).lower()
        for other, entry in meta.items():
            if other != key and entry.get("username") == info["username"]:
                del entry["username"]
    meta[key] = info


def _removed(member: dict) -> bool:
    """Whether a ``my_chat_member`` update takes the bot out of the chat."""
    new = member["new_chat_member"]
    status = new["status"]
    return status in _REMOVED_STATUSES or (status == _RESTRICTED and new.get("is_member") is False)


def _update_item(update: dict) -> dict | None:
    """The message, post, edit or ``my_chat_member`` an update carries, if it is one read here."""
    return update.get("my_chat_member") or next(
        (update[key] for key in _MESSAGE_KEYS if update.get(key)), None
    )


def _update_date(update: dict) -> int | None:
    """When Telegram produced the update: an edit's ``edit_date``, else the ``date`` every
    message and ``my_chat_member`` carries."""
    item = _update_item(update)
    if not item:
        return None
    return item.get("edit_date") or item.get("date") or None


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_timestamp(value: Any) -> bool:
    """An integer the date rendering can convert. The range is the platform's, not ours."""
    if not _is_int(value):
        return False
    try:
        datetime.fromtimestamp(value, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return False
    return True


def _check_page(updates: Any, offset: int | None) -> None:
    """Refuse a page the source cannot store, before any of it is planned: a list of objects
    whose ``update_id`` values are positive, rise strictly and start at ``offset`` or above
    when one was sent, each carrying an object whose ``date`` and ``edit_date``, unless missing
    or null, are integers the platform can turn into a time. Counted as ``failed_provider``.
    """
    if not isinstance(updates, list):
        raise _malformed_page("is not a list")
    # Telegram answers only ids at or above the offset it was sent.
    previous = 0 if offset is None else offset - 1
    for update in updates:
        if not isinstance(update, dict):
            raise _malformed_page("holds an update that is not an object")
        update_id = update.get("update_id")
        if not _is_int(update_id) or update_id <= previous:
            raise _malformed_page(
                "holds an update_id that is not a rising positive integer at or above the offset"
            )
        previous = update_id
        item = _update_item(update)
        if item is None:
            continue
        if not isinstance(item, dict):
            raise _malformed_page("holds an update whose message is not an object")
        for field in ("date", "edit_date"):
            if item.get(field) is not None and not _is_timestamp(item[field]):
                raise _malformed_page(f"holds an update whose {field} is not a timestamp")


def _malformed_page(what: str) -> TelegramAPIError:
    return TelegramAPIError(f"Telegram getUpdates: HTTP 200 with a page that {what}", status=200)


# ---------------------------------------------------------------------------
# Sync
# ---------------------------------------------------------------------------
def _quiet(state: dict, now: float) -> bool:
    """Whether the last update seen is over six days old, so ``update_id`` may have restarted."""
    seen_at = state.get("last_update_seen_at")
    return seen_at is not None and now - seen_at > _QUIET_RESET_SECONDS


def fetch_page(
    client: Any,
    state: dict,
    now: float,
    *,
    check_active: Callable[[], None] | None = None,
    budget: _WaitBudget | None = None,
) -> list:
    """The one ``getUpdates`` call of a run.

    The offset confirms everything before it, which is safe: those updates were loaded by an
    earlier run whose state dlt saved. After six days without updates the offset is left out,
    because Telegram may restart ``update_id`` below the saved cursor, and an offset above the
    new ids would confirm (drop) them. Without an offset nothing is confirmed. A page that is
    not a list of well-formed updates with rising ids, at or above the offset when one was
    sent, raises (:func:`_check_page`) before any of it is used.
    """
    last = state.get("last_update_id")
    quiet = _quiet(state, now)
    offset = last + 1 if last is not None and not quiet else None
    if quiet:
        logger.info("Telegram: no updates for over six days; reading without an offset.")
    params = {
        "offset": offset,
        "limit": _PAGE_SIZE,
        "timeout": 0,
        "allowed_updates": _ALLOWED_UPDATES,
    }
    updates = client.call("getUpdates", params, check_active=check_active, budget=budget) or []
    _check_page(updates, offset)
    return updates


def _chat_not_found(error: TelegramAPIError) -> bool:
    """Telegram's answer for a ``@username`` no chat carries: HTTP 400 "chat not found"."""
    return error.status == 400 and "chat not found" in error.description.lower()


def resolve_usernames(
    client: Any,
    state: dict,
    selection: _Selection,
    stored: "_StoredMessages",
    *,
    check_active: Callable[[], None] | None = None,
    budget: _WaitBudget | None = None,
) -> dict[str, dict | None]:
    """Ask ``getChat`` for each selected username no chat is bound to yet.

    The chat Telegram names, or ``None`` when it answers "chat not found". That answer alone is
    no evidence the name is a typo: a rename may not be visible to it yet. So when a name is not
    found and the selection changed, every stored chat this run would drop is asked for by its
    id, under the key ``#<chat id>``: its current username binds it again when it is the new
    selected name (the same chat, renamed), and its history stays. One rule, whatever the saved
    selection named (ids, usernames or every chat). A chat Telegram does not answer for by id (a
    400 such as not found, or 403: the bot is no longer in it) names nothing and is forgotten as
    deselected. Every other error propagates like any other request's, and the run stops or
    raises on it.
    """
    bound = selection.bound_usernames(state)
    resolved: dict[str, dict | None] = {}

    def ask(key: str, chat_id: int | str) -> None:
        try:
            resolved[key] = client.call(
                "getChat", {"chat_id": chat_id}, check_active=check_active, budget=budget
            )
        except TelegramAPIError as error:
            if not (_chat_not_found(error) or (key.startswith("#") and error.status in (400, 403))):
                raise
            resolved[key] = None

    for name in selection.usernames:
        if not bound.get(name):
            ask(name, f"@{name}")
    saved = state.get("selection")
    if None in resolved.values() and saved is not None and saved != selection.stored():
        for chat in filter(None, list(resolved.values())):
            selection.bind(bound, chat)
        migrated_from = state.get("migrated_from", {})
        meta = state.get("chats_meta", {})
        for key in sorted(stored.by_chat()):
            # A private chat (a positive id: a person) is never bound by a username, so no
            # answer could keep it; it is forgotten as deselected without asking.
            if int(key) < 0 and not selection.selects(int(key), migrated_from, meta, bound):
                ask(f"#{key}", int(key))
    return resolved


def count_gap(updates: list, state: dict, now: float) -> int:
    """Update ids skipped between the saved cursor and this page that may have expired unread.

    Telegram keeps an undelivered update for about 24 hours, so the ids skipped since the last
    update this connection saw can only have been dropped when that was longer ago. A hole is not
    proof: Telegram also skips the ids of update types the bot does not ask for. Only the hole
    before the page is counted. After six quiet days ids may restart, but only lower: a first id
    above the cursor shows the sequence went on, so its hole is counted too.
    """
    last = state.get("last_update_id")
    seen_at = state.get("last_update_seen_at")
    if not updates or last is None or seen_at is None:
        return 0
    if now - seen_at <= _UPDATE_RETENTION_SECONDS:
        return 0
    return max(updates[0]["update_id"] - last - 1, 0)


def _chat_of(key: str) -> str:
    """The chat id part of a row key ``<chat id>:<message id>``."""
    return key.split(":", 1)[0]


class _StoredMessages:
    """The keys of the stored message rows per chat, read from staging at most once per run.

    The staging table is the one record of which rows exist (a tombstone removes its row there),
    so a run reads it only when it has to name stored messages: the bot removed from a chat, a
    chat leaving the selection, an edit that takes the text away. A run of plain messages never
    reads it.
    """

    def __init__(self, read: Callable[[], Iterable[str]]):
        self._read = read
        self._by_chat: dict[str, set[str]] | None = None

    def by_chat(self) -> dict[str, set[str]]:
        if self._by_chat is None:
            by_chat: dict[str, set[str]] = {}
            for key in self._read():
                by_chat.setdefault(_chat_of(key), set()).add(key)
            self._by_chat = by_chat
        return self._by_chat


def plan_page(
    updates: list,
    state: dict,
    selection: _Selection,
    *,
    stored: _StoredMessages,
    stats: dict[str, int],
    now: float,
    resolved: dict[str, dict | None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """One page of updates as message rows and tombstones, plus the state to save after them.

    ``state`` is only read: the caller writes the returned state once every row is out, and dlt
    persists it only after the load succeeds, so a failed run is retried from the same cursor.
    A tombstone names the rows already staged (``stored``) and the rows emitted earlier in this
    page, which the load has not written yet.
    """
    migrated_from: dict[str, str] = dict(state.get("migrated_from", {}))
    _link_upgrades(updates, migrated_from)
    meta = {key: dict(info) for key, info in state.get("chats_meta", {}).items()}
    bound = selection.bound_usernames(state)
    # The chats getChat named first, then the page's: a username swapped for the same chat's
    # new one binds before the selection is compared, so the chat is not taken for deselected,
    # and a chat's message ahead of the first one carrying the selected username is selected.
    for chat in [*filter(None, (resolved or {}).values()), *_page_chats(updates)]:
        selection.bind(bound, chat)
    rows: dict[str, dict] = {}

    def forget_chats(chat_keys: set[str]) -> int:
        staged = stored.by_chat()
        keys = {key for chat_key in chat_keys for key in staged.get(chat_key, ())}
        keys |= {key for key in rows if _chat_of(key) in chat_keys}
        fresh = keys - {key for key, row in rows.items() if row["_deleted"]}
        rows.update((key, deleted_row(key)) for key in sorted(fresh))
        return len(fresh)

    def selected(chat_key: str) -> bool:
        return selection.selects(int(chat_key), migrated_from, meta, bound)

    def skip(reason: str) -> None:
        # ``skipped`` stays the total that hosts already read; the reason splits it.
        stats["skipped"] += 1
        stats[reason] += 1

    saved = state.get("selection")
    if saved is not None and saved != selection.stored():
        dropped = {chat_key for chat_key in stored.by_chat() if not selected(chat_key)}
        forgotten = forget_chats(dropped)
        logger.info(
            "Telegram: %d chat(s) left the selection; forgetting %d stored message(s).",
            len(dropped),
            forgotten,
        )

    for update in updates:
        member = update.get("my_chat_member")
        if member:
            # A ChatMemberUpdated always has the chat and the new member with a status.
            chat = member["chat"]
            _remember_chat(meta, chat)
            root = _root(chat["id"], migrated_from)
            # In a private chat ``kicked`` means the user blocked the bot: the chat stays.
            if _removed(member) and chat.get("type") != "private" and selected(str(chat["id"])):
                candidates = set(stored.by_chat()) | {_chat_of(key) for key in rows}
                chain = {key for key in candidates if _root(key, migrated_from) == root}
                logger.info(
                    "Telegram: the bot was removed from a chat; forgetting %d stored message(s).",
                    forget_chats(chain),
                )
            continue

        kind = next((k for k in _MESSAGE_KEYS if update.get(k)), None)
        if kind is None:
            skip("skipped_unread_update")  # an update type this connector does not read
            continue
        message = update[kind]
        chat = message["chat"]
        chat_id: int = chat["id"]
        _remember_chat(meta, chat)
        if not selection.selects(chat_id, migrated_from, meta, bound):
            skip("skipped_unselected")
            continue
        row = message_to_row(message, _root(chat_id, migrated_from))
        if row is not None:
            rows[row["id"]] = row  # a later edit in the same page wins
            continue
        # Only an edit can take the text from a stored row: a new message has a new id.
        key = row_key(chat_id, message["message_id"])
        if kind in _EDIT_KEYS and (key in rows or key in stored.by_chat().get(str(chat_id), ())):
            # The old text must not stay in memory.
            rows[key] = deleted_row(key)
        elif _content_kind(message):
            skip("skipped_unsupported_type")  # content cognee does not read, a caption aside
        else:
            skip("skipped_empty_content")  # a service message, or a blank text

    live = sum(1 for row in rows.values() if not row["_deleted"])
    stats["scanned"] += live
    stats["deleted"] += len(rows) - live
    if len(updates) >= _PAGE_SIZE:
        # Telegram drops updates it has not delivered after about 24 hours.
        stats["backlog"] = 1
        logger.warning(
            "Telegram: a full page of %d updates was read and more may be waiting. The next run "
            "continues; sync more often if this repeats, Telegram drops undelivered updates after "
            "about 24 hours.",
            len(updates),
        )
    logger.info(
        "Telegram: %d message row(s), %d deletion(s), %d update(s) skipped.",
        live,
        len(rows) - live,
        stats["skipped"],
    )

    # The meta is kept only for the chats selected, where a later run reads it (the type
    # decides a private chat). Every other chat seen would only make the state grow.
    meta = {key: info for key, info in meta.items() if selected(key)}
    new_state: dict[str, Any] = {
        "v": STATE_VERSION,
        "migrated_from": migrated_from,
        "chats_meta": meta,
        "selection": selection.stored(),
        "username_chats": bound,
    }
    if updates:
        # The ids of a page rise (_check_page), so the last one is the highest.
        new_state["last_update_id"] = updates[-1]["update_id"]
        # Telegram's restart clock runs from its last update, not from when this run read it.
        # A date of 0 or before 1970 counts as missing, and none is later than this run.
        dates = [date for date in map(_update_date, updates) if date and date > 0]
        new_state["last_update_seen_at"] = min(max(dates), now) if dates else now
    return list(rows.values()), new_state


def _staging_reader(table: str) -> Callable[[], list[str]]:
    """Reads the row ids of ``table`` from the destination of the dlt pipeline that is running.

    The resource name is the table name (the factory accepts only names dlt keeps unchanged).
    Before the first load the table does not exist, which means nothing is stored.
    """

    def read() -> list[str]:
        import dlt
        from dlt.common.configuration.container import Container
        from dlt.common.exceptions import DltException
        from dlt.common.pipeline import PipelineContext
        from dlt.destinations.exceptions import DatabaseUndefinedRelation

        # Asked first: with no pipeline running, dlt.current.pipeline() creates a default one.
        if not Container()[PipelineContext].is_active():
            raise TelegramStagingError(
                "Telegram: forgetting messages reads the stored rows from the staging table, so "
                "the source must run in a dlt pipeline (cognee.remember or cognee.add)."
            )
        try:
            # dlt annotates SqlClientBase.__exit__ without Optional parameters, so a type checker
            # rejects every ``with`` over it; the context manager itself is dlt's documented use.
            with dlt.current.pipeline().sql_client() as client:  # ty: ignore[invalid-context-manager]
                table_name = client.make_qualified_table_name(table)
                with client.execute_query(f"SELECT id FROM {table_name}") as cursor:
                    return [row[0] for row in cursor.fetchall()]
        except DatabaseUndefinedRelation:
            return []
        except DltException as error:
            raise TelegramStagingError(
                f"Telegram: the stored message ids could not be read from staging "
                f"({type(error).__name__})."
            ) from error

    return read


# A run stopped by one of these keeps its cursor and reports the reason; nothing was emitted, so
# the next run reads the same page.
_STOP_REASONS: tuple[tuple[type[TelegramSourceError], str], ...] = (
    (TelegramAuthError, "failed_auth"),
    (TelegramConflictError, "failed_conflict"),
    (TelegramRateLimitError, "failed_rate_limit"),
    (TelegramUnavailableError, "failed_unavailable"),
)
# A run that hits one of these raises, after counting it.
_RAISE_REASONS: tuple[tuple[type[TelegramSourceError], str], ...] = (
    (TelegramUnexpectedError, "failed_unexpected"),
    (TelegramBotMismatchError, "failed_bot_mismatch"),
    (TelegramAPIError, "failed_provider"),
)


def _reason(error: TelegramSourceError, reasons) -> str | None:
    return next((key for kind, key in reasons if isinstance(error, kind)), None)


def classify_sync_error(error: BaseException) -> str | None:
    """The stats key a raised run was counted under, read through the exceptions that wrap it
    (dlt's extraction error, the pipeline's), so a host counts one failure once."""
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, TelegramSourceError):
            return _reason(current, _RAISE_REASONS) or "failed_provider"
        current = current.__cause__ or current.__context__
    return None


def _counted(stats: dict[str, int], error: Exception) -> TelegramSourceError:
    """Count a failure that ends the run and return it typed, for the caller to raise: the stats
    are complete on every exit and a host classifies it under the key counted here (and counts
    it once). A failure that is not Telegram's is a defect of this source: ``failed_unexpected``."""
    typed = (
        error
        if isinstance(error, TelegramSourceError)
        else TelegramUnexpectedError(
            f"Telegram: the run failed unexpectedly ({type(error).__name__})."
        )
    )
    stats["failed"] = 1
    if not any(stats[key] for key in stats if key.startswith("failed_")):
        stats[_reason(typed, _RAISE_REASONS) or "failed_provider"] = 1
    return typed


def _check_owner(state: dict, me: Any, resource_name: str) -> None:
    owner, current = state.get("bot_id"), _bot_id(me)
    if owner is not None and owner != current:
        raise TelegramBotMismatchError(
            f"The sync state of resource_name '{resource_name}' belongs to another bot. Give "
            "each bot its own resource_name."
        )
    state["bot_id"] = current


def sync_rows(
    client: Any,
    state: dict,
    stats: dict[str, int],
    *,
    selection: _Selection,
    resource_name: str,
    staged_ids: Callable[[], Iterable[str]],
    check_active: Callable[[], None] | None = None,
    now: float | None = None,
) -> Iterator[dict[str, Any]]:
    """One run: ``getMe``, ``getChat`` per unbound selected username, one ``getUpdates`` page,
    its rows. Pure of dlt, tests drive it with
    a dict and a ``staged_ids`` of their own (the row ids staged so far, called at most once).

    A rejected token, a 409, a rate limit, a 5xx or no answer stops the run cleanly with its
    reason in ``stats``; any other failure is counted and raised."""
    if state.get("v", STATE_VERSION) != STATE_VERSION:
        stats["failed"] += 1
        stats["failed_provider"] = 1
        raise TelegramSourceError(
            f"This Telegram sync state has version {state['v']}, written by a newer cognee. "
            "Upgrade cognee before syncing."
        )
    budget = _WaitBudget()
    host_stops: list[BaseException] = []
    host_check = check_active

    def active() -> None:
        # The host's own stop (a revoked connection) is not a failure of the run: kept apart so
        # it passes through unwrapped and uncounted.
        try:
            host_check()
        except BaseException as stop:
            host_stops.append(stop)
            raise

    check_active = active if host_check is not None else None
    try:
        me = client.call("getMe", check_active=check_active, budget=budget)
        _check_owner(state, me, resource_name)
        # An int, not a bool: hosts keep only int counts.
        stats["privacy_mode"] = int(me.get("can_read_all_group_messages") is False)
        if stats["privacy_mode"]:
            logger.warning(
                "Telegram: privacy mode is on, so in groups this bot only sees commands and "
                "replies to it. Disable it in @BotFather (/setprivacy) or make the bot an admin "
                "to sync every message."
            )
        now = time.time() if now is None else now
        stored = _StoredMessages(staged_ids)
        resolved = resolve_usernames(
            client, state, selection, stored, check_active=check_active, budget=budget
        )
        updates = fetch_page(client, state, now, check_active=check_active, budget=budget)
    except Exception as error:
        if any(error is stop for stop in host_stops):
            raise
        stop = _reason(error, _STOP_REASONS) if isinstance(error, TelegramSourceError) else None
        if stop is None:
            typed = _counted(stats, error)
            if typed is error:
                raise
            raise typed from error
        stats["failed"] = 1
        stats[stop] = 1
        logger.warning("Telegram: run stopped (%s); the next sync reads the same updates.", stop)
        return
    stats["gap"] = count_gap(updates, state, now)
    if stats["gap"]:
        logger.warning(
            "Telegram: %d update id(s) are missing since the last sync, which saw its last update "
            "over 24 hours ago. Telegram drops updates it has not delivered after about 24 hours, "
            "so they may be lost (or they are update types this connector does not read). Sync "
            "at least daily.",
            stats["gap"],
        )
    try:
        rows, new_state = plan_page(
            updates,
            state,
            selection,
            stored=stored,
            stats=stats,
            now=now,
            resolved=resolved,
        )
        yield from rows
    except Exception as error:
        typed = _counted(stats, error)
        if typed is error:
            raise
        raise typed from error
    # Only reached once every row is out: a run stopped mid-way saves nothing.
    state.update(new_state)


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------
def telegram_source(
    *,
    resource_name: str | None = None,
    chats: list[int | str] | tuple[int | str, ...] | None = None,
    token: str | None = None,
    service: Any = None,
    check_active: Callable[[], None] | None = None,
    include_private: bool = False,
):
    """Return a ``dlt`` resource that yields the messages a Telegram bot sees as documents.

    Args:
        resource_name: Names the staging table and keys the sync state, so two bots in one
            dataset must never share one. Lowercase letters, digits and underscores, starting
            with a letter. Defaults to ``telegram_<bot id>`` when ``token`` is given; required
            with ``service``.
        chats: Chats to ingest, as numeric ids (``-1001234567890``) or ``@username``. ``None``
            ingests every group, supergroup and channel the bot is in. A chat dropped from the
            list is forgotten on the next run. A ``@username`` is bound to the chat ``getChat``
            names for it (or that is first seen with it) and stays bound through renames; one
            Telegram answers "chat not found" selects nothing. A private chat is selected by its id only.
        token: Bot token from @BotFather. Never read from the environment.
        service: Pre-built client (see :func:`build_telegram_service`), instead of ``token``. It
            must accept ``call(method, params, *, check_active, budget)`` and run
            ``check_active`` before each attempt it makes.
        check_active: Optional host checkpoint, called before every request attempt and wait and
            around every row (``dlt_utils.guarded_rows``). What it raises stops the run as it is.
        include_private: With ``chats=None``, also ingest private chats (messages people send
            the bot). Off by default: they are personal conversations.

    Returns:
        A ``dlt`` resource with ``primary_key="id"``, ``write_disposition="merge"`` and a
        ``_deleted`` hard-delete column. ``resource.cognee_sync_stats`` holds counts only, see
        :func:`new_stats`.
    """
    import dlt

    if getattr(dlt_utils, "DOCUMENT_SYNC_VERSION", 0) < 3:
        raise RuntimeError(
            "Telegram sync requires a Cognee build that reads per-row structure "
            "(DOCUMENT_SYNC_VERSION >= 3). Upgrade Cognee before syncing."
        )
    selection = _selection(chats, include_private)
    if service is not None and token is not None:
        raise ValueError("Pass token= or service=, not both.")
    if service is None and token is None:
        raise ValueError("Telegram bot token required: pass token= (from @BotFather).")
    # Built here so a malformed token fails at construction and the closure holds no raw token.
    client = service if service is not None else TelegramBotClient(token or "")
    if resource_name is None:
        if service is not None:
            raise ValueError("telegram_source needs a resource_name when service= is given.")
        resource_name = f"telegram_{client.bot_id}"
    if not isinstance(resource_name, str) or not _RESOURCE_NAME_RE.fullmatch(resource_name):
        raise ValueError(
            "resource_name must be lowercase letters, digits and underscores, starting with a "
            "letter, so dlt uses it unchanged as the table name."
        )
    name = resource_name
    stats: dict[str, int] = new_stats()

    @dlt.resource(
        name=name,
        primary_key="id",
        write_disposition="merge",
        columns={
            "_deleted": {"data_type": "bool", "hard_delete": True},
            # Typed up front so a column that is empty in a run (``url`` for private chats,
            # the text columns in a deletion-only run) still exists.
            "title": {"data_type": "text", "nullable": True},
            "content": {"data_type": "text", "nullable": True},
            "url": {"data_type": "text", "nullable": True},
        },
    )
    def telegram_messages():
        stats.clear()
        stats.update(new_stats())
        rows = sync_rows(
            client,
            dlt.current.resource_state(),
            stats,
            selection=selection,
            resource_name=name,
            staged_ids=_staging_reader(name),
            check_active=check_active,
        )
        yield from dlt_utils.guarded_rows(rows, check_active)

    resource = telegram_messages()
    # Route rows through the document path (text -> cognify).
    setattr(resource, DOCUMENT_SOURCE_ATTR, TELEGRAM_SOURCE_NAME)
    # The scope is the table name, so empty_resource's reset runs in the same dlt pipeline.
    setattr(resource, dlt_utils.PIPELINE_SCOPE_ATTR, name)
    # Host-readable diagnostics: counts only, never chat names or text.
    resource.cognee_sync_stats = stats
    return resource
