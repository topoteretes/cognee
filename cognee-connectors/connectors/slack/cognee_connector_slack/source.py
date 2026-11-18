"""Slack connector for cognee: a ``dlt`` source that turns one channel into memory.

Copied from cognee.tasks.ingestion.connectors.slack (cognee dev, 2026-11-12).

One source is one channel of one workspace. A thread (a message with replies) is
one document, and the messages without replies of one UTC day are one document for
that day. The source is meant to be handed directly to :func:`cognee.remember`::

    import cognee
    from cognee_connector_slack import slack_source

    await cognee.remember(
        slack_source(
            access_token="xoxb-...",
            team_id="T0123",
            channel_id="C0456",
            resource_name="slack_channel_general",
        ),
        dataset_name="slack_acme",
        primary_key="id",
        write_disposition="merge",   # REQUIRED, the add pipeline defaults to "replace"
        max_rows_per_table=0,
    )

Design
------
* **Rows** are ``{id, title, content, url, _deleted, cognee_node_set,
  cognee_structure}``. The id is ``<channel_id>:<root_ts>`` for a thread and
  ``<channel_id>:day:<YYYY-MM-DD>`` for a day, the node set is
  ``slack:channel:<channel_id>`` and the structure names the channel as the
  container of every row. Authors are rendered by name (``users.info``, one
  lookup per person and run). Attachments are named, never downloaded.
* **Incremental** by a re-walk with fingerprints, with no time window.
  ``conversations.history`` is read newest first and paged with ``latest``, so a
  walk cut short resumes from the end of the oldest day it reached. Every root keeps a
  fingerprint ``(reply_count, latest_reply, edited.ts, subtype)``, and
  ``conversations.replies`` runs only for a root whose fingerprint changed. A
  channel with no changes yields nothing.
* **Deletion**: a root missing from a *complete* walk is emitted as a tombstone.
  A walk cut short forgets nothing. A root older than the oldest message Slack
  still shows is forgotten too: Slack does not tell a deletion, a retention policy
  and a free plan's hidden history apart, and memory follows what Slack shows. A
  thread Slack reports as gone while the walk reads it is a tombstone too.
* **A run is bounded and resumable.** A spent request or wall-clock budget, a rate
  limit asking for more than 60 s or outlasting the retries, or an expired token or an
  outage after rows were emitted stops the run
  cleanly: the state reached is kept, ``cognee_sync_stats`` reports
  ``failed_<reason>``, and the next run resumes. ``dlt`` rolls the state of a
  failed run back, so these never raise.
* **Token** is a plain bot token, never refreshed here. It lives in the client
  object only; errors carry the API method and Slack's error code, never the
  token, a channel id, a URL or a response body.

Privacy
-------
This reads the messages of the channel, the names of their authors and the names
of attached files. The text goes to the configured LLM and embedding providers,
and anyone with read access to the target dataset can see it.
"""

import logging
import re
import time
from bisect import bisect_left
from collections.abc import Callable, Iterator
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any, NoReturn
from urllib.parse import urlparse

from cognee_connectors_core import dlt_conventions as dlt_utils
from cognee_connectors_core.dlt_conventions import (
    DOCUMENT_SOURCE_ATTR,
    NODE_SET_COLUMN,
    STRUCTURE_COLUMN,
    with_retries,
)

logger = logging.getLogger(__name__)

API_URL = "https://slack.com/api/"
# Slack's recommended page size for conversations.history and conversations.replies.
PAGE_SIZE = 200
# One run's request budget; a bigger channel finishes over several runs.
DEFAULT_MAX_REQUESTS = 1500
# Attempts per request on a 5xx, a network error or a rate limit (the former
# history client's bound).
_ATTEMPTS = 4
# The longest Retry-After a run waits out: Tier 1 history asks for about a minute, and a
# shorter bound cut every channel after the first on every sync (G46). The walk holds
# its dataset's dlt staging lock (other datasets stage meanwhile), so a longer wait, or one
# past the run's own deadline (``max_wait``), stops the run instead, and it resumes later.
_MAX_RETRY_AFTER_SECONDS = 60
# One run's wall-clock budget, checked before each request. The other connectors never
# sleep and have none; this one waits out short rate limits, so the budget keeps one run
# from holding the staging lock longer than the 600 s after which the integration resumes
# a cut run.
MAX_RUN_SECONDS = 600
# The Cognee app's command words. A message that starts by mentioning the app, a command or
# a question, is talk to the app and no document carries it (``_Walker._is_command``,
# Goran, 2026-11-10); the app's mention handler answers these words.
APP_COMMANDS = frozenset({"help", "link", "unlink", "status", "remember", "forget"})
# Stats key of a run that was cut where the previous one was (G56).
STALLED_KEY = "failed_day_too_large"
_TIMEOUT = 30.0
_STATE_VERSION = 1
# Marks the fingerprint of a root whose thread rendered nothing (the app's own thread, a
# join with replies), so it is not read again until it changes. Fingerprints start with a
# digit, so the mark cannot collide with one.
_EMPTY = "~"

_TS = re.compile(r"\d+\.\d{6}")
# A mention as Slack writes it in a message's text, ``<@U123>`` or ``<@U123|name>``.
_MENTION = re.compile(r"<@([A-Z0-9]+)(?:\|[^>]*)?>")
_CHANNEL = re.compile(r"[CG][A-Z0-9]+")
_CODE = re.compile(r"[a-z0-9_]+")
_SCOPE = re.compile(r"[a-z0-9:._,]+")
# Slack's answers for an invalid, revoked or expired token.
_AUTH_CODES = frozenset(
    {"invalid_auth", "not_authed", "token_revoked", "token_expired", "account_inactive"}
)
# A thread that was deleted between the history page and the replies call.
_THREAD_GONE = frozenset({"thread_not_found", "message_not_found"})
# Answers Slack gave with ``ok`` that the walk cannot read: not a dict, a page that is not
# a list, a timestamp that is not one, a cursor that is missing, repeats or does not move.
_MALFORMED = frozenset(
    {
        "invalid_response",
        "missing_cursor",
        "repeated_cursor",
        "invalid_timestamp",
        "pagination_stalled",
    }
)
# Stats key of the threads whose replies Slack answered malformed; each counts as failed.
UNREADABLE_THREAD_KEY = "failed_thread_unreadable"
# A user who is deleted from the workspace directory or hidden from the bot.
_UNKNOWN_USER = frozenset({"user_not_found", "user_not_visible"})
# Subtypes that carry what someone wrote. Every other subtype (joins, leaves, topic
# and purpose changes, pins, deleted-message tombstones) is left out of the text.
_CONTENT_SUBTYPES = frozenset({"bot_message", "file_share", "me_message", "thread_broadcast"})


# ---------------------------------------------------------------------------
# Errors: messages carry the API method and Slack's code, never tokens, ids or URLs.
# ---------------------------------------------------------------------------
class SlackSourceError(RuntimeError):
    """Base class of the errors this source raises."""


class SlackAPIError(SlackSourceError):
    """Slack answered a call with an error, or could not be reached."""

    def __init__(self, method: str, code: str, message: str | None = None):
        super().__init__(message or f"Slack {method} failed: {code}")
        self.method = method
        self.code = code


class SlackAuthError(SlackAPIError):
    """Slack rejected the token (invalid, revoked or expired)."""


class SlackMissingScopeError(SlackAPIError):
    """The token lacks a scope the call needs; only a reconnect grants it."""

    def __init__(self, method: str, needed: str):
        super().__init__(
            method,
            "missing_scope",
            f"Slack {method} needs {needed}. Reconnect Slack to grant {needed}.",
        )
        self.needed = needed


class SlackUnavailableError(SlackAPIError):
    """Slack answered 5xx, or could not be reached, on every attempt."""


class SlackRateLimitedError(SlackAPIError):
    """Slack kept answering ``ratelimited`` after the retries, or asked for a long wait."""

    def __init__(self, method: str, *, retry_after: int | None = None):
        super().__init__(method, "ratelimited", f"Slack {method} is rate limited")
        # Seconds Slack asked to wait (Retry-After), for the client's retry policy.
        self.retry_after = retry_after


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------
class SlackClient:
    """Minimal synchronous Slack Web API client.

    ``call`` returns the decoded answer of one method. Parameters go in a form body,
    so channel ids never appear in a request URL or in httpx's request log. Hosts and
    tests can pass any object with the same ``call(method, *, check_active, **params)``
    instead; it must call ``check_active`` before each attempt it makes.
    """

    def __init__(
        self,
        access_token: str,
        *,
        transport: Any = None,
        sleep: Callable[[float], None] = time.sleep,
        max_retry_after: int = _MAX_RETRY_AFTER_SECONDS,
    ):
        token = (access_token or "").strip()
        # Reject early and without echoing the value: httpx would put an invalid
        # header value, token included, into its exception.
        if not token or not token.isascii() or not token.isprintable() or " " in token:
            raise ValueError("The Slack access token is empty or has an invalid format")
        self._access_token = token
        self._transport = transport
        self._sleep = sleep
        # A rate limit asking for a longer wait raises at once; 0 never waits one out.
        self._max_retry_after = max_retry_after

    def __repr__(self) -> str:
        return "SlackClient(access_token=<redacted>)"

    def call(
        self,
        method: str,
        *,
        check_active: Callable[[], None] | None = None,
        max_wait: float | None = None,
        **params: Any,
    ) -> dict[str, Any]:
        """The decoded answer of one Slack method, retrying a 5xx, a network error and a
        short rate limit; ``check_active`` runs before the first attempt here and before
        every wait and retry in ``with_retries``, so a revoked connection never sends a
        request or waits out a retry. ``max_wait`` is what the caller's run has left: a
        rate limit asking for longer is raised."""
        if check_active is not None:
            check_active()
        return with_retries(
            lambda: self._attempt(method, params),
            retry_on=(SlackUnavailableError, SlackRateLimitedError),
            delay=lambda error, number: self._retry_delay(error, number, max_wait),
            attempts=_ATTEMPTS,
            check_active=check_active,
            sleep=self._sleep,
        )

    def _retry_delay(
        self, error: BaseException, number: int, max_wait: float | None = None
    ) -> float | None:
        if isinstance(error, SlackRateLimitedError):
            # A longer wait than this client, or the caller's run, allows is raised.
            limit = (
                self._max_retry_after if max_wait is None else min(self._max_retry_after, max_wait)
            )
            wait = error.retry_after
            return wait if wait is not None and wait <= limit else None
        return 2**number

    def _attempt(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        import httpx

        try:
            with httpx.Client(transport=self._transport, timeout=_TIMEOUT) as http:
                response = http.post(
                    f"{API_URL}{method}",
                    data=params,
                    headers={"Authorization": f"Bearer {self._access_token}"},
                )
        except httpx.HTTPError as exc:
            raise SlackUnavailableError(method, f"unreachable_{type(exc).__name__}") from None
        status = response.status_code
        if status >= 500:
            raise SlackUnavailableError(method, f"http_{status}")
        if status == 401:
            raise SlackAuthError(method, "http_401")
        payload = _payload(response, method) if status == 200 else {}
        if status == 429 or payload.get("error") == "ratelimited":
            raise SlackRateLimitedError(
                method, retry_after=_retry_after(response.headers.get("Retry-After"))
            )
        if status != 200:
            raise SlackAPIError(method, f"http_{status}")
        if payload.get("ok") is True:
            return payload
        code = _safe(payload.get("error"), _CODE, "unknown_error")
        if code in _AUTH_CODES:
            raise SlackAuthError(method, code)
        if code == "missing_scope":
            raise SlackMissingScopeError(
                method, _safe(payload.get("needed"), _SCOPE, "the missing scope")
            )
        raise SlackAPIError(method, code)


def _payload(response: Any, method: str) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if not isinstance(payload, dict):
        raise SlackAPIError(method, "invalid_response")
    return payload


def _retry_after(value: Any) -> int:
    return max(1, int(value)) if isinstance(value, str) and value.isdigit() else 1


def _safe(value: Any, pattern: re.Pattern, default: str) -> str:
    """A provider string that is safe to put into a message, or ``default``."""
    return value if isinstance(value, str) and pattern.fullmatch(value) else default


# ---------------------------------------------------------------------------
# Rendering: only the document columns, nothing volatile, messages sorted by ts.
# ---------------------------------------------------------------------------
def _ts(message: Any, method: str) -> str:
    ts = message.get("ts") if isinstance(message, dict) else None
    if not isinstance(ts, str) or not _TS.fullmatch(ts):
        raise SlackAPIError(method, "invalid_timestamp")
    return ts


def _when(ts: str) -> str:
    seconds = int(ts.split(".", 1)[0])
    return datetime.fromtimestamp(seconds, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _fingerprint(message: dict[str, Any]) -> str:
    """What changes on a root when its thread does: replies, edits, deletion."""
    edited = (message.get("edited") or {}).get("ts") or ""
    return "|".join(
        (
            str(int(message.get("reply_count") or 0)),
            str(message.get("latest_reply") or ""),
            str(edited),
            str(message.get("subtype") or ""),
        )
    )


_MARKUP = re.compile(r"<([^<>]+)>")
_TITLE_WORDS = 10


def _unescape(text: str) -> str:
    """Slack escapes only these three in message text."""
    return text.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")


def _title(text: str, author: str, ts: str) -> str:
    """The first words, the author and the day: unique enough, and free of the channel
    name, so a rename re-cognifies nothing (G11)."""
    words = text.split()
    snippet = " ".join(words[:_TITLE_WORDS]) + ("…" if len(words) > _TITLE_WORDS else "")
    day = datetime.fromtimestamp(int(ts.split(".", 1)[0]), timezone.utc).strftime("%Y-%m-%d")
    return f"{snippet} — {author}, {day}"


_TITLE_TAIL = re.compile(
    r"\A(?P<snippet>.*) — (?P<author>.+), (?P<day>\d{4}-\d{2}-\d{2})\Z", re.DOTALL
)


def split_title(title: str) -> tuple[str, str, date] | None:
    """The first words, the author and the day of a title ``_title`` wrote; ``None`` for
    any other title."""
    match = _TITLE_TAIL.match(title)
    if match is None:
        return None
    try:
        day = date.fromisoformat(match["day"])
    except ValueError:
        return None
    return match["snippet"], match["author"], day


def _attachment_texts(message: dict[str, Any]) -> list[str]:
    """The text of a shared message or a bot's attachment. A link preview (an attachment
    Slack unfurled from a URL in the text) is left out: the URL is already in the text."""
    texts = []
    for attachment in message.get("attachments") or []:
        if not isinstance(attachment, dict):
            continue
        if attachment.get("from_url") and not attachment.get("is_share"):
            continue
        parts = [attachment.get(key) for key in ("pretext", "title", "text")]
        parts += [
            f"{field.get('title')}: {field.get('value')}"
            for field in attachment.get("fields") or []
            if isinstance(field, dict) and field.get("title") and field.get("value")
        ]
        texts += [part for part in parts if isinstance(part, str) and part.strip()]
    return texts


def _block_texts(blocks: Any) -> list[str]:
    """The text of Block Kit blocks: headers, sections, context and rich text."""
    texts: list[str] = []
    for block in blocks if isinstance(blocks, list) else []:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind in ("header", "section"):
            texts.append((block.get("text") or {}).get("text") or "")
            texts += [(item or {}).get("text") or "" for item in block.get("fields") or []]
        elif kind == "context":
            texts += [(item or {}).get("text") or "" for item in block.get("elements") or []]
        elif kind == "rich_text":
            texts.append("".join(_rich_text(block.get("elements"))))
    return [text for text in texts if isinstance(text, str) and text.strip()]


def _rich_text(elements: Any) -> Iterator[str]:
    for element in elements if isinstance(elements, list) else []:
        if not isinstance(element, dict):
            continue
        kind = element.get("type")
        if kind == "text":
            yield element.get("text") or ""
        elif kind == "link":
            yield element.get("text") or element.get("url") or ""
        elif kind == "user":
            yield f"<@{element.get('user_id', '')}>"
        elif kind == "channel":
            yield f"<#{element.get('channel_id', '')}>"
        elif "elements" in element:
            yield from _rich_text(element["elements"])
            if kind in ("rich_text_section", "rich_text_preformatted", "rich_text_quote"):
                yield "\n"


def _day(ts: str) -> str:
    """The UTC day of a message: the unit of the messages outside threads (G7)."""
    return datetime.fromtimestamp(int(ts.split(".", 1)[0]), timezone.utc).strftime("%Y-%m-%d")


def _day_start(day: str) -> int:
    return int(datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())


def _day_end(day: str) -> int:
    return _day_start(day) + 86_400


def _standalone(fingerprint: str) -> bool:
    """Whether a stored fingerprint is of a message without replies (part of its day)."""
    return fingerprint.removeprefix(_EMPTY).split("|", 1)[0] == "0"


def _file_names(message: dict[str, Any]) -> list[str]:
    """Names of the files on a message, in Slack's order. Their content is never read."""
    names = []
    for item in message.get("files") or []:
        name = (item.get("name") or item.get("title")) if isinstance(item, dict) else None
        if isinstance(name, str) and name.strip():
            names.append(name.strip())
    return names


def _workspace_url(url: Any) -> str:
    parsed = urlparse(url if isinstance(url, str) else "")
    host = parsed.hostname or ""
    if parsed.scheme != "https" or not host.endswith(".slack.com"):
        raise SlackSourceError("Slack auth.test returned no slack.com workspace address")
    return f"https://{host}"


class _CutShort(Exception):
    """Internal: stop the run here, keeping the state reached so far."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class _Walker:
    """One run against one channel, writing progress into ``state`` (a plain dict).

    ``state`` is ``{"v", "roots": {root_ts: fingerprint}, "days"?, "dirty_days"?,
    "position"?, "missing"?, "names"?}``: a fingerprint per top-level message read (``~``
    first for a root whose replies were read and rendered nothing), the days that have a
    day row and the days still to render (G7), and while a walk is unfinished, the end of
    the oldest day it reached, the known roots it found gone so far and the authors' names
    it looked up. A walk with no change leaves the state as it found it, so an idle sync
    writes no new dlt state.
    """

    def __init__(
        self,
        client: Any,
        team_id: str,
        channel_id: str,
        state: dict,
        stats: dict[str, int],
        *,
        check_active: Callable[[], None] | None,
        max_requests: int,
        max_seconds: float,
        clock: Callable[[], float],
        changed_roots: set[str] | None = None,
        excluded_messages: set[str] | None = None,
    ):
        self.client = client
        # Messages taken out of memory by their author or the connection's owner.
        self.excluded = frozenset(excluded_messages or ())
        # Roots whose thread changed where their fingerprint cannot see it (an edited or
        # deleted reply, a removed file: G6). Read again and dropped from the set as the
        # walk reaches them; the caller keeps what is left for the next run.
        self.changed = changed_roots if changed_roots is not None else set()
        self.team_id = team_id
        self.channel_id = channel_id
        self.state = state
        self.stats = stats
        self.check_active = check_active
        self.max_requests = max_requests
        self.clock = clock
        self.deadline = clock() + max_seconds
        self.base_url = ""
        self.own_ids: set[str] = set()
        self.names: dict[str, str] = {}
        # The users ``users.info`` says are apps (bot users), kept with the names.
        self.apps: list[str] = []
        self.channel_name: str | None = None
        self.shared = False
        # The messages without replies this run read, by day, and the days to render.
        self.day_messages: dict[str, dict[str, dict[str, Any]]] = {}
        self.dirty: set[str] = set()
        # The read in progress ("newer" or "walk"), where it resumed and the pages it read
        # this run: a budget cut after pages that left its position where it resumed is a
        # day no run can read whole (G56, G73).
        self.reading: tuple[str, str | None] | None = None
        self.reading_pages = 0
        # Rows the run had emitted or deleted when that read started: a read that adds to
        # them made progress, though its position did not move (a day of threads, G83).
        self.reading_rows = 0

    # -- requests ----------------------------------------------------------
    def _call(
        self, method: str, *, answered: frozenset[str] = frozenset(), **params: Any
    ) -> dict[str, Any]:
        """One request within the run's budget. ``answered`` names the Slack error codes
        the caller handles itself (a user or a thread that is gone): they are raised as
        they are, never turned into a cut."""
        if self.stats["requests"] >= self.max_requests or self.clock() >= self.deadline:
            raise _CutShort("budget")
        self.stats["requests"] += 1
        try:
            # The client checks the host before every attempt and wait; what the check
            # raises is not caught here: a revoked connection is not an item failure.
            return self.client.call(
                method,
                check_active=self.check_active,
                max_wait=self.deadline - self.clock(),
                **params,
            )
        except SlackRateLimitedError:
            raise _CutShort("rate_limit") from None
        except SlackAuthError:
            if self.stats["emitted"]:
                raise _CutShort("auth") from None
            raise
        except SlackMissingScopeError:
            raise  # only a reconnect grants it: the whole workspace stops
        except SlackAPIError as error:
            if error.code in answered:
                raise
            self._fail(error)

    def _fail(self, error: SlackAPIError) -> NoReturn:
        """Stop the run on ``error``: an outage, any other code Slack answers with (F-B4)
        or an answer the walk cannot read. Raising would make dlt drop every row this run
        already emitted; after rows the run is cut and the next one resumes, before any row
        the error is the run's."""
        if self.stats["emitted"]:
            raise _CutShort("unavailable") from None
        raise error

    def _malformed(self, method: str, code: str) -> NoReturn:
        self._fail(SlackAPIError(method, code))

    def _history_ts(self, message: Any) -> str:
        try:
            return _ts(message, "conversations.history")
        except SlackAPIError as error:
            self._fail(error)

    def _identify(self) -> None:
        identity = self._call("auth.test")
        if identity.get("team_id") != self.team_id:
            raise SlackSourceError("The Slack token belongs to another workspace than team_id")
        self.base_url = _workspace_url(identity.get("url"))
        # The Cognee app's own messages (answers it posted) are not memory.
        self.own_ids = {
            value for value in (identity.get("user_id"), identity.get("bot_id")) if value
        }

    def _name(self, user_id: str) -> str:
        if user_id not in self.names:
            try:
                user = (
                    self._call("users.info", answered=_UNKNOWN_USER, user=user_id).get("user") or {}
                )
            except SlackAPIError as error:
                if error.code not in _UNKNOWN_USER:
                    raise
                user = {}
            if user.get("is_bot") and user_id not in self.apps:
                self.apps.append(user_id)
            profile = user.get("profile") or {}
            # The full name first: a display name changes more often and is shared
            # more often, so it would split one person and merge two (G8).
            name = (
                profile.get("real_name")
                or user.get("real_name")
                or profile.get("display_name")
                or user.get("name")
            )
            self.names[user_id] = (
                name.strip() if isinstance(name, str) and name.strip() else user_id
            )
        return self.names[user_id]

    def _channel_label(self) -> str:
        if self.channel_name is None:
            info = self._call("conversations.info", channel=self.channel_id).get("channel") or {}
            name = info.get("name")
            self.channel_name = name if isinstance(name, str) else ""
            # Slack Connect: a channel shared with another organization is synced and its
            # documents carry ``slack:shared``, so a search can leave a partner's messages out.
            self.shared = bool(info.get("is_ext_shared"))
        return self.channel_name

    def _thread(self, root_ts: str) -> list[dict[str, Any]]:
        """The root and all of its replies, oldest first; ``[]`` when the thread is gone.
        Raises ``SlackAPIError`` with a ``_MALFORMED`` code when Slack's answer cannot be
        read."""
        messages: dict[str, dict[str, Any]] = {}
        cursor = ""
        cursors: set[str] = set()
        while True:
            params: dict[str, Any] = {"channel": self.channel_id, "ts": root_ts, "limit": PAGE_SIZE}
            if cursor:
                params["cursor"] = cursor
            try:
                page = self._call(
                    "conversations.replies", answered=_THREAD_GONE | _MALFORMED, **params
                )
            except SlackAPIError as error:
                if error.code in _THREAD_GONE:
                    return []
                raise
            batch = page.get("messages")
            if not isinstance(batch, list):
                raise SlackAPIError("conversations.replies", "invalid_response")
            for message in batch:
                messages[_ts(message, "conversations.replies")] = message
            cursor = ((page.get("response_metadata") or {}).get("next_cursor") or "").strip()
            if not cursor:
                if page.get("has_more"):
                    raise SlackAPIError("conversations.replies", "missing_cursor")
                break
            if cursor in cursors:
                raise SlackAPIError("conversations.replies", "repeated_cursor")
            cursors.add(cursor)
        return [messages[ts] for ts in sorted(messages, key=Decimal)]

    # -- rendering ---------------------------------------------------------
    def _is_content(self, message: dict[str, Any]) -> bool:
        if message.get("ts") in self.excluded or self._is_command(message):
            return False
        subtype = message.get("subtype")
        if subtype is not None and subtype not in _CONTENT_SUBTYPES:
            return False
        return message.get("user") not in self.own_ids and message.get("bot_id") not in self.own_ids

    def _is_command(self, message: dict[str, Any]) -> bool:
        """A message that starts by mentioning an app, this one or another: a command or a
        question to that app, not what the channel said (Goran, 2026-11-10 and 2026-11-12)."""
        words = (message.get("text") or "").split()
        mention = _MENTION.fullmatch(words[0]) if words else None
        return mention is not None and (
            mention.group(1) in self.own_ids or self._is_app(mention.group(1))
        )

    def _is_app(self, user_id: str) -> bool:
        self._name(user_id)  # asks users.info once, which says whether it is an app
        return user_id in self.apps

    def _from_app(self, message: dict[str, Any]) -> bool:
        user = message.get("user")
        return bool(message.get("bot_id")) or (
            isinstance(user, str) and bool(user) and self._is_app(user)
        )

    def _author(self, message: dict[str, Any]) -> str:
        user = message.get("user")
        if isinstance(user, str) and user:
            return self._name(user)
        bot = message.get("username") or (message.get("bot_profile") or {}).get("name")
        return bot or message.get("bot_id") or "unknown"

    def _text(self, message: dict[str, Any]) -> str:
        """What a person reads in the message: its text, then the content Slack keeps in
        attachments (a shared message, a bot's alert); blocks only when there is neither,
        since Slack's text is otherwise their fallback (G49). Markup decoded (G48)."""
        parts = [self._plain(message.get("text") or "")]
        parts += [self._plain(text) for text in _attachment_texts(message)]
        if not any(parts):
            parts = [self._plain(text) for text in _block_texts(message.get("blocks"))]
        return "\n".join(part for part in parts if part)

    def _plain(self, text: str) -> str:
        """Slack's markup as a reader sees it: names, channel names, link labels (G48)."""
        return _unescape(_MARKUP.sub(self._markup, text)).strip()

    def _markup(self, match: re.Match) -> str:
        target, _, label = match.group(1).partition("|")
        if target.startswith("@"):
            return f"@{self._name(target[1:])}"
        if target.startswith("#"):
            return f"#{label or target[1:]}"
        if target.startswith("!subteam^"):
            return label or "@group"
        if target.startswith("!date^"):
            return label
        if target.startswith("!"):
            return f"@{target[1:]}"
        if target.startswith("mailto:"):
            return label or target[len("mailto:") :]
        return f"{label} ({target})" if label and label != target else target

    def _render(self, row_id: str, thread: list[dict[str, Any]]) -> dict[str, Any] | None:
        """One document of ``thread``'s messages, oldest first; ``None`` when none has text.
        Its permalink is the first message's."""
        blocks = []
        first = None
        # A thread opened by a command to an app is that app's conversation: what apps reply
        # in it is the app talking, not the channel; what people reply stays (live, 2026-11-12).
        app_thread = bool(thread) and self._is_command(thread[0])
        for message in thread:
            if not self._is_content(message) or (app_thread and self._from_app(message)):
                continue
            text = self._text(message)
            files = _file_names(message)
            if not text and not files:
                continue
            ts = message["ts"]
            author = self._author(message)
            lines = [f"[{_when(ts)}] {author}: {text}".rstrip()]
            if files:
                lines.append(f"Attached: {', '.join(files)}")
            blocks.append("\n".join(lines))
            first = first or (text or f"Attached: {', '.join(files)}", author, ts)
        if first is None:
            return None
        name = self._channel_label()
        channel = {"kind": "slack_channel", "id": self.channel_id}
        if name:
            channel["name"] = f"#{name}"
        return {
            "id": row_id,
            "title": _title(*first),
            "content": "\n\n".join(blocks),
            "url": f"{self.base_url}/archives/{self.channel_id}/p{first[2].replace('.', '')}",
            "_deleted": False,
            NODE_SET_COLUMN: [f"slack:channel:{self.channel_id}"]
            + (["slack:shared"] if self.shared else []),
            STRUCTURE_COLUMN: {"ancestors": [channel]},
        }

    # -- the walk ------------------------------------------------------------
    def _root(self, message: dict[str, Any], roots: dict[str, str]) -> Iterator[dict[str, Any]]:
        """One top-level message. A thread is its own document; a message without replies
        belongs to the document of its day, rendered once the walk has read that whole day
        (G7)."""
        ts = message["ts"]
        self.stats["scanned"] += 1
        fingerprint = _fingerprint(message)
        known = roots.get(ts)
        # A mark is dropped only once its root is read: a run cut before that keeps it (G57).
        changed = ts in self.changed
        replied = bool(message.get("reply_count"))
        if not replied:
            day = _day(ts)
            self.day_messages.setdefault(day, {})[ts] = message
            if known is not None and not _standalone(known) and not known.startswith(_EMPTY):
                # Its replies are gone: the thread's document goes, the message joins its day.
                self.stats["deleted"] += 1
                yield {"id": f"{self.channel_id}:{ts}", "_deleted": True}
            if known != fingerprint or changed:
                self._mark_day(day)
            else:
                self.stats["skipped"] += 1
            roots[ts] = fingerprint
            self.changed.discard(ts)
            return
        # Read without replies earlier in this run (the read of what is newer reads the
        # newest known root's day twice): it leaves that day's messages now, or the day's
        # document would keep it next to its thread (G89).
        if self.day_messages.get(_day(ts), {}).pop(ts, None) is not None:
            self._mark_day(_day(ts))
        if known in (fingerprint, _EMPTY + fingerprint) and not changed:
            self.stats["skipped"] += 1
            return
        if known is not None and _standalone(known):
            self._mark_day(_day(ts))  # it got replies: it leaves its day
        try:
            thread = self._thread(ts)
        except SlackAPIError as error:
            if error.code not in _MALFORMED:
                raise
            # One thread Slack cannot answer readably is a failed item, not a failed run:
            # its stored document and fingerprint stay, so the next run reads it again, and
            # the walk goes on to older messages instead of stopping here on every run.
            self.stats["failed"] = self.stats.get("failed", 0) + 1
            self.stats[UNREADABLE_THREAD_KEY] = self.stats.get(UNREADABLE_THREAD_KEY, 0) + 1
            return
        row = self._render(f"{self.channel_id}:{ts}", thread)
        self.changed.discard(ts)
        if row is None:
            # Every message of the thread was deleted, or none was written by a person. It
            # keeps its fingerprint, so the thread is not read again until it changes.
            self.stats["skipped"] += 1
            roots[ts] = _EMPTY + fingerprint
            if known is not None and not known.startswith(_EMPTY) and not _standalone(known):
                self.stats["deleted"] += 1
                yield {"id": f"{self.channel_id}:{ts}", "_deleted": True}
            return
        roots[ts] = fingerprint
        self.stats["emitted"] += 1
        yield row

    def _mark_day(self, day: str) -> None:
        """Have ``day``'s document rendered again; kept in the state until it is."""
        self.dirty.add(day)
        self.state["dirty_days"] = sorted(self.dirty)

    def _render_days(self, complete: Callable[[str], bool]) -> Iterator[dict[str, Any]]:
        """Render every marked day the walk has read whole, newest first."""
        days = self.state.setdefault("days", {})
        for day in sorted(self.dirty, reverse=True):
            if not complete(day):
                continue
            messages = self.day_messages.get(day, {})
            row = self._render(
                f"{self.channel_id}:day:{day}",
                [messages[ts] for ts in sorted(messages, key=Decimal)],
            )
            if row is not None:
                days[day] = 1
                self.stats["emitted"] += 1
                yield row
            elif days.pop(day, None) is not None:
                self.stats["deleted"] += 1
                yield {"id": f"{self.channel_id}:day:{day}", "_deleted": True}
            self.dirty.discard(day)
            if self.dirty:
                self.state["dirty_days"] = sorted(self.dirty)
            else:
                self.state.pop("dirty_days", None)
        if not days:
            self.state.pop("days", None)

    def _newer_than(self, head: int, roots: dict[str, str]) -> Iterator[dict[str, Any]]:
        """The roots posted since ``head`` (a second: a day start or the newest known root),
        newest first, while a walk is unfinished, so a message posted meanwhile does not
        wait for the walk to end (G47). Finds no deletion: only a complete walk may conclude that a root is gone.

        Each day read whole is rendered after the page that finished it, and a cut keeps
        where this read started and reached (``newer``: ``[head, the end of a day]``), so
        the next run goes on from there instead of reading everything newer again, and
        down to the same ``head``: otherwise, once more was newer than one run could read,
        the channel emitted nothing again and the walk below never moved (G71).
        """
        resumed = self.state.get("newer")
        if resumed:
            head, latest = int(resumed[0]), resumed[1]
        else:
            latest = None
        self.reading, self.reading_pages = ("newer", resumed), 0
        self.reading_rows = self.stats["emitted"] + self.stats["deleted"]
        top = Decimal(latest) if latest is not None else None
        params: dict[str, Any] = {
            "channel": self.channel_id,
            "limit": PAGE_SIZE,
            "oldest": f"{head - 1}.999999",
        }
        if latest is not None:
            params["latest"] = latest

        def read_whole(day: str, reached: Decimal | None = None) -> bool:
            return (
                _day_start(day) >= head
                and (reached is None or _day_start(day) > reached)
                and (top is None or _day_end(day) <= top)
            )

        # What an earlier read of this run gathered at or after ``head`` is read again here:
        # a message deleted in between must not stay in its day's document (G89).
        for messages_of_day in self.day_messages.values():
            for ts in [ts for ts in messages_of_day if Decimal(ts) >= head]:
                del messages_of_day[ts]
        cursors: set[str] = set()
        while True:
            page = self._call("conversations.history", **params)
            self.reading_pages += 1
            messages = page.get("messages")
            if not isinstance(messages, list):
                self._malformed("conversations.history", "invalid_response")
            oldest = None
            for message in messages:
                ts = self._history_ts(message)
                oldest = ts if oldest is None or Decimal(ts) < Decimal(oldest) else oldest
                if message.get("thread_ts", ts) == ts:
                    yield from self._root(message, roots)
            cursor = ((page.get("response_metadata") or {}).get("next_cursor") or "").strip()
            if not cursor:
                if page.get("has_more"):
                    self._malformed("conversations.history", "missing_cursor")
                break
            if cursor in cursors:
                self._malformed("conversations.history", "repeated_cursor")
            cursors.add(cursor)
            params["cursor"] = cursor
            if oldest is not None:
                reached = Decimal(oldest)
                yield from self._render_days(lambda day, reached=reached: read_whole(day, reached))
                self.state["newer"] = [str(head), f"{_day_end(_day(oldest))}.000000"]
        yield from self._render_days(read_whole)
        self.state.pop("newer", None)

    def rows(self) -> Iterator[dict[str, Any]]:
        self._identify()
        state = self.state
        if state.setdefault("v", _STATE_VERSION) != _STATE_VERSION:
            raise SlackSourceError("The Slack sync state was written by an unknown version")
        roots: dict[str, str] = state.setdefault("roots", {})
        # Names live in the state while a walk is unfinished, so a walk over many runs
        # (Tier 1) asks Slack for each author once; a completed walk drops them, so the
        # next one reads them fresh (G50).
        self.names = state.setdefault("names", {})
        self.apps = state.setdefault("apps", [])
        self.dirty = set(state.get("dirty_days", []))
        position = state.get("position")
        # Where this run's walk starts: the end of a day, so every day it reaches is read
        # from its first message (a cut stores its position that way).
        top = Decimal(position) if position is not None else None
        if position is not None and roots:
            # A walk is unfinished: read what is newer first, so a message posted meanwhile
            # does not wait for the walk to end (G47). From the newest known root itself, so
            # a quiet channel spends one request here, not its newest day again on every run
            # (a busy newest day starved the walk below for good, G71), and from the start
            # of any marked day above the walk, which is read whole. A message that arrived
            # in the newest known root's day marks it, and that day is then read whole in
            # this run. Never below where the walk resumes (a day boundary), which the walk
            # itself reads.
            def above_walk() -> list[int]:
                return [_day_start(day) for day in self.dirty if _day_end(day) > top]

            head = max(min([int(Decimal(max(roots, key=Decimal))), *above_walk()]), int(top))
            yield from self._newer_than(head, roots)
            widened = [start for start in above_walk() if start < head]
            if widened:
                yield from self._newer_than(max(min(widened), int(top)), roots)
        self.reading, self.reading_pages = ("position", position), 0
        self.reading_rows = self.stats["emitted"] + self.stats["deleted"]
        # The roots known when the run started, oldest first, to find the ones a page lacks.
        known = sorted(roots, key=Decimal)
        bounds = [Decimal(ts) for ts in known]
        while True:
            params: dict[str, Any] = {"channel": self.channel_id, "limit": PAGE_SIZE}
            if position is not None:
                params["latest"] = position
            page = self._call("conversations.history", **params)
            self.reading_pages += 1
            messages = page.get("messages")
            if not isinstance(messages, list):
                self._malformed("conversations.history", "invalid_response")
            oldest = None
            seen = set()
            for message in messages:
                ts = self._history_ts(message)
                oldest = ts if oldest is None or Decimal(ts) < Decimal(oldest) else oldest
                if message.get("thread_ts", ts) != ts:
                    continue  # a reply also sent to the channel; it belongs to its thread
                seen.add(ts)
                yield from self._root(message, roots)
            more = bool(page.get("has_more"))
            if more and (
                oldest is None or (position is not None and Decimal(oldest) >= Decimal(position))
            ):
                self._malformed("conversations.history", "pagination_stalled")
            # The page holds every message from its oldest one up to ``position``, so a
            # known root in that span that the page lacks is gone. The last page reaches
            # back to the start of the channel: a root older than anything Slack still
            # shows was deleted by a person, removed by a retention policy or hidden by a
            # free plan, and Slack does not say which. Memory follows what Slack shows (the
            # Marketplace guidelines forbid access past a free workspace's 90 days); a root
            # Slack shows again is read again. Roots are forgotten only when the walk
            # completes, so a walk cut short never reaches this.
            low = bisect_left(bounds, Decimal(oldest)) if more else 0
            high = bisect_left(bounds, Decimal(position)) if position is not None else len(known)
            missing = [ts for ts in known[low:high] if ts not in seen and ts in roots]
            if missing:
                state.setdefault("missing", []).extend(missing)
                # A message of a day that is gone: the day is read whole in this run, so its
                # document is rendered again from what Slack shows.
                for ts in missing:
                    if _standalone(roots[ts]):
                        self._mark_day(_day(ts))
            if not more:
                break
            reached = Decimal(oldest)
            yield from self._render_days(
                lambda day, reached=reached: (
                    _day_start(day) > reached and (top is None or _day_end(day) <= top)
                )
            )
            position = oldest
            state["position"] = f"{_day_end(_day(oldest))}.000000"
        # The walk is complete: every day below where it started has been read whole.
        yield from self._render_days(lambda day: top is None or _day_end(day) <= top)
        # Only now may the threads it did not see be forgotten; a day's messages went with
        # its document.
        for ts in sorted(set(state.pop("missing", [])), key=Decimal):
            stored = roots.pop(ts, None)
            if stored is not None and not stored.startswith(_EMPTY) and not _standalone(stored):
                self.stats["deleted"] += 1
                yield {"id": f"{self.channel_id}:{ts}", "_deleted": True}
        state.pop("position", None)
        state.pop("names", None)
        state.pop("apps", None)
        # A mark for a message that is no root of this channel (deleted since) is done; a
        # root above where this walk resumed keeps its mark for the next walk.
        self.changed.intersection_update(roots)


def _iter_rows(
    client: Any,
    team_id: str,
    channel_id: str,
    state: dict,
    stats: dict[str, int],
    *,
    check_active: Callable[[], None] | None = None,
    max_requests: int = DEFAULT_MAX_REQUESTS,
    max_seconds: float = MAX_RUN_SECONDS,
    clock: Callable[[], float] = time.monotonic,
    changed_roots: set[str] | None = None,
    excluded_messages: set[str] | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield the changed rows of a channel. Pure of dlt, so tests drive it with a dict."""
    walker = _Walker(
        client,
        team_id,
        channel_id,
        state,
        stats,
        check_active=check_active,
        max_requests=max_requests,
        max_seconds=max_seconds,
        clock=clock,
        changed_roots=changed_roots,
        excluded_messages=excluded_messages,
    )
    try:
        yield from walker.rows()
    except _CutShort as cut:
        stats["failed"] = max(1, stats.get("failed", 0))
        stats[f"failed_{cut.reason}"] = 1
        key, resumed_from = walker.reading or (None, None)
        if (
            cut.reason == "budget"
            and walker.reading_pages
            and resumed_from is not None
            and state.get(key) == resumed_from
            and stats["emitted"] + stats["deleted"] == walker.reading_rows
        ):
            # A cut keeps its position at the end of a day, so a day bigger than one run
            # (Tier 1: about 140 messages) is read from its end on every run and the walk
            # never moves (G56). Said apart from an ordinary cut, so it is not retried
            # silently forever. Only when the read in progress (the walk, or the read of what
            # is newer, which resumes the same way) read pages and its budget ran out inside
            # that one day: a rate limit or a cut before any page is an ordinary cut (G73).
            stats[STALLED_KEY] = 1
            logger.warning(
                "Slack sync made no progress: one UTC day of a channel has more messages "
                "than a run can read under the current rate limits."
            )
        else:
            logger.warning("Slack sync stopped early (%s); it resumes on the next run.", cut.reason)


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------
def build_slack_service(access_token: str) -> SlackClient:
    """The service the integration layer injects (``_PROVIDERS`` service builder)."""
    return SlackClient(access_token)


def slack_source(
    *,
    access_token: str | None = None,
    service: Any = None,
    team_id: str,
    channel_id: str,
    resource_name: str,
    check_active: Callable[[], None] | None = None,
    max_requests: int = DEFAULT_MAX_REQUESTS,
    changed_roots: set[str] | None = None,
    excluded_messages: set[str] | None = None,
):
    """Return a ``dlt`` resource that yields one Slack channel's documents for ``remember``.

    Args:
        access_token: Bot token, used to build the client when ``service`` is omitted.
            Needs ``channels:history`` (``groups:history`` for a private channel),
            ``channels:read`` (``groups:read``) and ``users:read``.
        service: Pre-built client (see :func:`build_slack_service`), the injection
            point for tests and hosts. Its ``call(method, *, check_active, max_wait,
            **params)`` runs ``check_active`` before every attempt and wait, and raises a
            rate limit asking for longer than ``max_wait`` seconds instead of waiting.
        team_id: The workspace the token must belong to; checked with ``auth.test``.
        channel_id: The channel to read (``C...`` or ``G...``). The app must be a member.
        resource_name: Stable name of this channel's connection. It names the staging
            table and scopes the sync state, so two channels must never share one.
        check_active: Host authorization checkpoint, called before each request attempt
            and each retry wait (the client runs it, ``with_retries``).
        changed_roots: Thread roots to read again although their root looks unchanged
            (a reply edited or deleted, a file removed: what the app's message events
            report). The walk removes each root from the set as it reads it; what is left
            is the host's to keep for the next run.
        excluded_messages: Timestamps of messages taken out of memory: they are left out
            of every document they would be in. Changing the set re-renders nothing by
            itself; the host marks their threads in ``changed_roots``.
        max_requests: Request budget of one run. A bigger channel finishes over several
            runs. A call retried after a rate limit or a 5xx counts once.

    Returns:
        A ``dlt`` resource configured with ``primary_key="id"``,
        ``write_disposition="merge"`` and a ``_deleted`` hard-delete column.
    """
    import dlt

    dlt_utils.require_document_sync_version(
        3, provider="Slack", feature="reads per-row node sets and structure"
    )
    if not isinstance(team_id, str) or not team_id.strip():
        raise ValueError("slack_source needs the team_id of the workspace")
    if not isinstance(channel_id, str) or not _CHANNEL.fullmatch(channel_id):
        raise ValueError("slack_source needs a Slack channel id (C... or G...)")
    if not isinstance(resource_name, str) or not resource_name.strip():
        raise ValueError("slack_source needs a resource_name, one per channel")
    if service is None and not access_token:
        raise ValueError("slack_source needs a service or an access_token")

    # Built here so a malformed token fails at construction, and the closure holds
    # the client, not the raw token.
    client = service if service is not None else SlackClient(access_token or "")
    stats: dict[str, int] = {}

    @dlt.resource(
        name=resource_name,
        primary_key="id",
        write_disposition="merge",
        columns={"_deleted": {"data_type": "bool", "hard_delete": True}},
    )
    def slack_channel():
        stats.clear()
        stats.update(scanned=0, emitted=0, skipped=0, deleted=0, failed=0, requests=0)
        rows = _iter_rows(
            client,
            team_id,
            channel_id,
            dlt.current.resource_state(),
            stats,
            check_active=check_active,
            max_requests=max_requests,
            changed_roots=changed_roots,
            excluded_messages=excluded_messages,
        )
        yield from dlt_utils.guarded_rows(rows, check_active)

    resource = slack_channel()
    setattr(resource, DOCUMENT_SOURCE_ATTR, "slack")
    setattr(resource, dlt_utils.PIPELINE_SCOPE_ATTR, resource_name)
    resource.cognee_sync_stats = stats
    return resource
