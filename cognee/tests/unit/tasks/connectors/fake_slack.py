"""An in-memory Slack Web API, shaped like the methods ``slack_source`` calls.

It follows Slack's documented behaviour where the source depends on it:
``conversations.history`` lists top-level messages (and replies also sent to the
channel) newest first, pages with ``latest`` (exclusive) and says ``has_more``;
a root with replies carries ``reply_count`` and ``latest_reply``;
``conversations.replies`` lists the root and its replies oldest first, paged
with a cursor; a missing thread answers ``thread_not_found`` and an unknown user
``user_not_found``. ``page_limit`` stands in for a smaller page size, so tests
can page without hundreds of messages.
"""

from collections.abc import Callable
from decimal import Decimal
from typing import Any

from cognee.tasks.ingestion.connectors.slack import (
    SlackAPIError,
    SlackMissingScopeError,
    SlackRateLimitedError,
)

TEAM = "T0TEAM"
CHANNEL = "C0GENERAL"
BOT_USER = "U0COGNEE"
BOT_ID = "B0COGNEE"
WORKSPACE_URL = "https://acme.slack.com/"


class FakeSlack:
    def __init__(self, *, page_limit: int = 200):
        self.page_limit = page_limit
        self.url = WORKSPACE_URL
        self.channels: dict[str, dict[str, Any]] = {}
        self.users: dict[str, dict[str, Any]] = {}
        self.members: dict[str, list[str]] = {}
        self.calls: list[tuple[str, dict[str, Any]]] = []
        # method -> [calls still answered, exception raised on every call after them]
        self.failures: dict[str, list] = {}
        self.add_channel(CHANNEL, "general")

    # -- setup ---------------------------------------------------------------
    def add_channel(
        self, channel_id: str, name: str, *, is_member: bool = True, is_private: bool = False
    ) -> None:
        self.channels[channel_id] = {
            "name": name,
            "is_member": is_member,
            "is_private": is_private,
            "messages": {},
        }

    def add_user(self, user_id: str, display_name: str = "", real_name: str = "") -> None:
        self.users[user_id] = {
            "id": user_id,
            "name": user_id.lower(),
            "profile": {"display_name": display_name, "real_name": real_name},
        }

    def post(self, ts: str, user: str | None, text: str, channel: str = CHANNEL, **extra) -> None:
        message = {"type": "message", "ts": ts, "text": text, **extra}
        if user is not None:
            message["user"] = user
        self.channels[channel]["messages"][ts] = message

    def reply(self, root_ts: str, ts: str, user: str, text: str, channel: str = CHANNEL, **extra):
        self.post(ts, user, text, channel, thread_ts=root_ts, **extra)

    def edit(self, ts: str, text: str, edited_ts: str, channel: str = CHANNEL) -> None:
        message = self.channels[channel]["messages"][ts]
        message["text"] = text
        message["edited"] = {"user": message.get("user"), "ts": edited_ts}

    def delete(self, ts: str, channel: str = CHANNEL) -> None:
        del self.channels[channel]["messages"][ts]

    def fail(self, method: str, error: Exception, *, after: int = 0) -> None:
        """Answer ``after`` more calls to ``method``, then raise ``error`` until ``heal()``."""
        self.failures[method] = [after, error]

    def heal(self) -> None:
        self.failures.clear()

    def methods(self) -> list[str]:
        return [method for method, _ in self.calls]

    # -- the API -------------------------------------------------------------
    def call(
        self,
        method: str,
        *,
        check_active: Callable[[], None] | None = None,
        max_wait: float | None = None,
        **params: Any,
    ) -> dict[str, Any]:
        if check_active is not None:
            check_active()  # one attempt, so one check: the client's contract
        self.calls.append((method, params))
        if method in self.failures:
            failure = self.failures[method]
            if failure[0] <= 0:
                raise failure[1]
            failure[0] -= 1
        handler = getattr(self, "_" + method.replace(".", "_"))
        return {"ok": True, **handler(**params)}

    def _auth_test(self) -> dict[str, Any]:
        return {"team_id": TEAM, "url": self.url, "user_id": BOT_USER, "bot_id": BOT_ID}

    def _conversations_info(self, channel: str) -> dict[str, Any]:
        found = self.channels.get(channel)
        if found is None:
            raise SlackAPIError("conversations.info", "channel_not_found")
        return {
            "channel": {
                "id": channel,
                "name": found["name"],
                "is_member": found["is_member"],
                "is_private": found.get("is_private", False),
                "is_im": found.get("is_im", False),
                "is_mpim": found.get("is_mpim", False),
                "is_archived": found.get("is_archived", False),
                "is_ext_shared": found.get("is_ext_shared", False),
            }
        }

    def _conversations_members(self, channel: str, limit: int, cursor: str = "") -> dict:
        members = self.members.get(channel, [])
        start = int(cursor or 0)
        page = members[start : start + self.page_limit]
        following = start + self.page_limit
        next_cursor = str(following) if following < len(members) else ""
        return {"members": page, "response_metadata": {"next_cursor": next_cursor}}

    def _thread_of(self, channel: str, root_ts: str) -> list[dict[str, Any]]:
        messages = self.channels[channel]["messages"].values()
        return [m for m in messages if m["ts"] != root_ts and m.get("thread_ts") == root_ts]

    def _listed(self, channel: str, message: dict[str, Any]) -> dict[str, Any]:
        """A message as conversations.history shows it: a root carries its thread summary."""
        listed = dict(message)
        replies = self._thread_of(channel, message["ts"])
        if replies:
            listed["thread_ts"] = message["ts"]
            listed["reply_count"] = len(replies)
            listed["latest_reply"] = max((r["ts"] for r in replies), key=Decimal)
        return listed

    def _conversations_history(
        self,
        channel: str,
        limit: int,
        latest: str | None = None,
        oldest: str | None = None,
        cursor: str = "",
    ):
        """Newest first; ``latest`` and ``oldest`` exclusive, as Slack's defaults; the
        cursor is an offset into what they select."""
        if channel not in self.channels:
            raise SlackAPIError("conversations.history", "channel_not_found")
        top = [
            self._listed(channel, m)
            for m in self.channels[channel]["messages"].values()
            if m.get("thread_ts") in (None, m["ts"]) or m.get("subtype") == "thread_broadcast"
        ]
        top.sort(key=lambda m: Decimal(m["ts"]), reverse=True)
        if latest is not None:
            top = [m for m in top if Decimal(m["ts"]) < Decimal(latest)]
        if oldest is not None:
            top = [m for m in top if Decimal(m["ts"]) > Decimal(oldest)]
        start = int(cursor or 0)
        size = min(int(limit), self.page_limit)
        page = {"messages": top[start : start + size], "has_more": len(top) > start + size}
        if page["has_more"]:
            page["response_metadata"] = {"next_cursor": str(start + size)}
        return page

    def _conversations_replies(self, channel: str, ts: str, limit: int, cursor: str = ""):
        messages = self.channels[channel]["messages"]
        if ts not in messages:
            raise SlackAPIError("conversations.replies", "thread_not_found")
        thread = [self._listed(channel, messages[ts]), *self._thread_of(channel, ts)]
        thread.sort(key=lambda m: Decimal(m["ts"]))
        start = int(cursor or 0)
        size = min(int(limit), self.page_limit)
        following = start + size
        more = following < len(thread)
        return {
            "messages": thread[start:following],
            "has_more": more,
            "response_metadata": {"next_cursor": str(following) if more else ""},
        }

    def _users_info(self, user: str) -> dict[str, Any]:
        if user not in self.users:
            raise SlackAPIError("users.info", "user_not_found")
        return {"user": self.users[user]}


def missing_scope() -> SlackMissingScopeError:
    return SlackMissingScopeError("users.info", "users:read")


def rate_limited(method: str = "conversations.history") -> SlackRateLimitedError:
    return SlackRateLimitedError(method)
