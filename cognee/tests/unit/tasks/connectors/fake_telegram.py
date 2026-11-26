# Adapted from topoteretes/cognee-community (Apache-2.0), PR #355, commit 76ff1a4.
# Modified to ship in the Cognee SDK with SDK-local imports and test paths.
"""An in-memory Telegram Bot API for tests (no token, no network).

The payloads mirror what a real test bot received: groups, a supergroup upgrade, a channel,
edits and the bot being removed from a chat.
"""

import json
import time

GROUP = {"id": -100, "type": "group", "title": "Team chat"}
SUPERGROUP = {"id": -1009, "type": "supergroup", "title": "Team chat"}
CHANNEL = {"id": -1001, "type": "channel", "title": "News", "username": "news_chan"}
OTHER = {"id": -555, "type": "group", "title": "Unrelated"}
PRIVATE = {"id": 42, "type": "private", "first_name": "Carol", "last_name": "Diaz"}
ALICE = {"id": 1, "is_bot": False, "first_name": "Alice", "username": "alice"}
BOB = {"id": 2, "is_bot": False, "first_name": "Bob"}
# When every fake message was sent, unless a test dates it: an hour ago, so a run on the real
# clock is not six quiet days after it.
DATE = int(time.time()) // 60 * 60 - 3600
TOKEN = "111:AAsecret-token-value-1234567890"


def msg(chat, message_id, text=None, sender=ALICE, date=DATE, **extra):
    message = {"message_id": message_id, "chat": chat, "date": date, **extra}
    if text is not None:
        message["text"] = text
    if sender is not None and chat.get("type") != "channel":
        message["from"] = sender
    return message


def removed(chat, status="left", date=DATE, **member):
    return {
        "chat": chat,
        "date": date,
        "new_chat_member": {"status": status, "user": {"id": 99}, **member},
    }


class FakeBot:
    """In-memory Bot API: a list of updates served like getUpdates does.

    Like the real API, a getUpdates call with an offset confirms (drops) every update below it,
    so a test can see an update get lost for good. ``errors[method]`` is a list of exceptions
    raised by the next calls of that method, one each. ``check_active`` runs once per call, as a
    client runs it before its first attempt.
    """

    def __init__(self, updates=None, bot_id=99, can_read_all=True):
        self.updates = list(updates or [])
        self.bot_id = bot_id
        self.can_read_all = can_read_all
        self.errors: dict[str, list[Exception]] = {}
        self.calls = []
        self.next_id = 1000
        # The public chats getChat resolves by @username: every chat an added update carries
        # under a username, as it is called in the latest one. A test renames or adds by hand.
        self.public: dict[str, dict] = {}
        # The chats getChat answers by id: every chat an added update carries, as it is now.
        # A test renames one by hand (the chat changed at Telegram, nothing posted since).
        self.by_id: dict[int, dict] = {}

    def _publish(self, update):
        for item in update.values():
            chat = item.get("chat") if isinstance(item, dict) else None
            if chat:
                self.by_id[chat["id"]] = chat
            if chat and chat.get("username") and chat.get("type") != "private":
                self.public[chat["username"].lower()] = chat

    def add(self, **update):
        self._publish(update)
        # Ids keep counting after confirmed updates are dropped, like Telegram's.
        self.updates.append({"update_id": self.next_id, **update})
        self.next_id += 1

    def add_with_id(self, update_id, **update):
        self._publish(update)
        self.updates.append({"update_id": update_id, **update})

    def call(self, method, params=None, *, check_active=None, budget=None):
        if check_active is not None:
            check_active()
        params = params or {}
        self.calls.append((method, params))
        if self.errors.get(method):
            raise self.errors[method].pop(0)
        if method == "getMe":
            return {
                "id": self.bot_id,
                "is_bot": True,
                "username": "test_bot",
                "can_read_all_group_messages": self.can_read_all,
            }
        if method == "getChat":
            target = params["chat_id"]
            if isinstance(target, int):
                chat = self.by_id.get(target)
            else:
                chat = self.public.get(target.lstrip("@").lower())
            if chat is None:
                # What the wire answers, turned into an error by the real client's own code.
                from cognee.tasks.ingestion.connectors.telegram import TelegramBotClient

                wire = {
                    "ok": False,
                    "error_code": 400,
                    "description": "Bad Request: chat not found",
                }
                TelegramBotClient(TOKEN)._answer("getChat", 400, json.dumps(wire).encode())
            return chat
        if method == "getUpdates":
            offset = params.get("offset")
            if offset is not None:
                self.updates = [u for u in self.updates if u["update_id"] >= offset]
            return self.updates[: params.get("limit", 100)]
        raise AssertionError(method)

    def get_updates_calls(self):
        return [params for method, params in self.calls if method == "getUpdates"]


class FakeStaging:
    """The staging table's row ids, as a merge load with hard deletes leaves them.

    ``read`` is what the source is handed as ``staged_ids``; ``load`` applies a run's rows the
    way a successful load does. ``reads`` counts how often a run asked.
    """

    def __init__(self, ids=()):
        self.ids = set(ids)
        self.reads = 0

    def read(self):
        self.reads += 1
        return sorted(self.ids)

    def load(self, rows):
        for row in rows:
            if row["_deleted"]:
                self.ids.discard(row["id"])
            else:
                self.ids.add(row["id"])


def never_read():
    raise AssertionError("this run must not read the staging table")
