"""Unit tests for the Gmail connector (cognee/tasks/ingestion/connectors/gmail.py).

The Gmail SaaS API is fully mocked via ``FakeGmailService`` — no Google client
libraries and no live credentials are required, so these run in CI. Coverage:

  - message parsing (headers, base64url body, multipart, label flattening)
  - full backfill yields every message and records the historyId baseline
  - incremental re-sync yields ONLY the delta (added/changed + deletions)
  - deleted/trashed messages become hard-delete markers (forget-on-delete)
  - an expired historyId (404) transparently falls back to a full backfill
  - the dlt resource is wired with merge + id PK + the hard_delete column

The end-to-end "deletion removes from memory" guarantee is provided by the
existing ``orphan_cleanup`` path (see test_dlt_p0_correctness.py); here we
prove the connector emits the markers that drive it.
"""

import base64

import pytest

from cognee.tasks.ingestion.connectors.gmail import (
    full_backfill,
    gmail_source,
    incremental_fetch,
    parse_message,
)


# ---------------------------------------------------------------------------
# Fake Gmail API
# ---------------------------------------------------------------------------
def _b64(text: str) -> str:
    return base64.urlsafe_b64encode(text.encode("utf-8")).decode("utf-8")


def _make_message(msg_id, *, subject="", sender="", body="", labels=None, history_id="100"):
    return {
        "id": msg_id,
        "threadId": f"t_{msg_id}",
        "labelIds": labels or ["INBOX"],
        "snippet": (body or subject)[:50],
        "historyId": history_id,
        "internalDate": "1700000000000",
        "payload": {
            "mimeType": "text/plain",
            "headers": [
                {"name": "Subject", "value": subject},
                {"name": "From", "value": sender},
                {"name": "To", "value": "me@example.com"},
            ],
            "body": {"data": _b64(body)} if body else {},
        },
    }


class _HttpError(Exception):
    """Mimics googleapiclient.errors.HttpError enough for the 404 branch."""

    def __init__(self, status):
        super().__init__(f"HTTP {status}")
        self.resp = type("Resp", (), {"status": status})()


class _Request:
    def __init__(self, result):
        self._result = result

    def execute(self):
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


class _Messages:
    def __init__(self, svc):
        self._svc = svc

    def list(self, **kwargs):
        return _Request({"messages": [{"id": mid} for mid in self._svc.message_ids]})

    def get(self, *, userId, id, format):  # noqa: A002 - mirror Gmail API kwarg name
        if id in self._svc.get_errors:
            return _Request(_HttpError(self._svc.get_errors[id]))
        if id not in self._svc.messages:
            return _Request(_HttpError(404))
        return _Request(self._svc.messages[id])


class _History:
    def __init__(self, svc):
        self._svc = svc

    def list(self, **kwargs):
        return _Request(self._svc.history_response)


class _Users:
    def __init__(self, svc):
        self._svc = svc

    def messages(self):
        return _Messages(self._svc)

    def history(self):
        return _History(self._svc)

    def getProfile(self, **kwargs):  # noqa: N802 - mirror Gmail API name
        return _Request(
            {"emailAddress": "me@example.com", "historyId": self._svc.profile_history_id}
        )


class FakeGmailService:
    """Minimal stand-in for the googleapiclient Gmail service."""

    def __init__(
        self, messages=None, profile_history_id="500", history_response=None, get_errors=None
    ):
        self.messages = {m["id"]: m for m in (messages or [])}
        self.message_ids = list(self.messages.keys())
        self.profile_history_id = profile_history_id
        self.history_response = history_response or {"history": [], "historyId": profile_history_id}
        # Map of message id -> HTTP status to raise from messages().get(), used
        # to simulate transient (non-404) fetch failures.
        self.get_errors = get_errors or {}

    def users(self):
        return _Users(self)


# ---------------------------------------------------------------------------
# parse_message
# ---------------------------------------------------------------------------
def test_parse_message_flattens_headers_body_and_labels():
    msg = _make_message(
        "m1",
        subject="Lunch?",
        sender="alice@example.com",
        body="Want to grab lunch tomorrow?",
        labels=["INBOX", "IMPORTANT"],
    )
    row = parse_message(msg)

    assert row["id"] == "m1"
    assert row["thread_id"] == "t_m1"
    assert row["subject"] == "Lunch?"
    assert row["from"] == "alice@example.com"
    assert row["body"] == "Want to grab lunch tomorrow?"
    assert row["labels"] == "INBOX, IMPORTANT"  # list flattened, no child table
    assert row["internal_date"] == 1700000000000
    assert row["_deleted"] is False


def test_parse_message_handles_multipart_prefers_text_plain():
    msg = {
        "id": "m2",
        "threadId": "t",
        "labelIds": [],
        "payload": {
            "mimeType": "multipart/alternative",
            "headers": [{"name": "Subject", "value": "Hi"}],
            "parts": [
                {"mimeType": "text/html", "body": {"data": _b64("<p>html</p>")}},
                {"mimeType": "text/plain", "body": {"data": _b64("plain wins")}},
            ],
        },
    }
    assert parse_message(msg)["body"] == "plain wins"


def test_parse_message_tolerates_missing_internal_date():
    row = parse_message({"id": "m3", "payload": {}})
    assert row["internal_date"] == 0
    assert row["body"] == ""


# ---------------------------------------------------------------------------
# full_backfill
# ---------------------------------------------------------------------------
def test_full_backfill_yields_all_and_records_baseline_history_id():
    svc = FakeGmailService(
        messages=[_make_message("a"), _make_message("b"), _make_message("c")],
        profile_history_id="900",
    )
    state = {}
    rows = list(full_backfill(svc, state))

    assert {r["id"] for r in rows} == {"a", "b", "c"}
    assert all(r["_deleted"] is False for r in rows)
    # Baseline captured for the next incremental run.
    assert state["last_history_id"] == "900"


def test_full_backfill_respects_max_results():
    svc = FakeGmailService(messages=[_make_message(str(i)) for i in range(10)])
    rows = list(full_backfill(svc, {}, max_results=3))
    assert len(rows) == 3


# ---------------------------------------------------------------------------
# incremental_fetch
# ---------------------------------------------------------------------------
def test_incremental_fetch_yields_only_delta_and_advances_cursor():
    added = _make_message("new1", subject="Fresh", history_id="1010")
    svc = FakeGmailService(
        messages=[added],
        history_response={
            "history": [
                {"id": "1005", "messagesAdded": [{"message": {"id": "new1"}}]},
            ],
            "historyId": "1010",
        },
    )
    state = {"last_history_id": "1000"}
    rows = list(incremental_fetch(svc, state))

    assert len(rows) == 1
    assert rows[0]["id"] == "new1"
    assert rows[0]["_deleted"] is False
    assert state["last_history_id"] == "1010"  # cursor advanced


def test_incremental_fetch_emits_hard_delete_marker_for_deletions():
    svc = FakeGmailService(
        messages=[],
        history_response={
            "history": [
                {"id": "1006", "messagesDeleted": [{"message": {"id": "gone1"}}]},
            ],
            "historyId": "1006",
        },
    )
    state = {"last_history_id": "1000"}
    rows = list(incremental_fetch(svc, state))

    assert rows == [{"id": "gone1", "_deleted": True}]
    assert state["last_history_id"] == "1006"


def test_incremental_fetch_add_then_delete_in_window_is_net_deletion():
    svc = FakeGmailService(
        messages=[_make_message("x")],
        history_response={
            "history": [
                {"id": "1007", "messagesAdded": [{"message": {"id": "x"}}]},
                {"id": "1008", "messagesDeleted": [{"message": {"id": "x"}}]},
            ],
            "historyId": "1008",
        },
    )
    rows = list(incremental_fetch(svc, {"last_history_id": "1000"}))
    # Net effect: a single hard-delete, no fetched body.
    assert rows == [{"id": "x", "_deleted": True}]


def test_incremental_fetch_expired_history_id_falls_back_to_backfill():
    svc = FakeGmailService(
        messages=[_make_message("a"), _make_message("b")],
        profile_history_id="2000",
        history_response=_HttpError(404),
    )
    state = {"last_history_id": "1"}  # too old / expired
    rows = list(incremental_fetch(svc, state))

    assert {r["id"] for r in rows} == {"a", "b"}  # recovered via full backfill
    assert state["last_history_id"] == "2000"


def test_incremental_fetch_missing_message_on_fetch_is_treated_as_deleted():
    # History says "added", but the message 404s on get() (vanished meanwhile).
    svc = FakeGmailService(
        messages=[],  # get("ghost") -> 404
        history_response={
            "history": [{"id": "1009", "messagesAdded": [{"message": {"id": "ghost"}}]}],
            "historyId": "1009",
        },
    )
    rows = list(incremental_fetch(svc, {"last_history_id": "1000"}))
    assert rows == [{"id": "ghost", "_deleted": True}]


def test_incremental_fetch_transient_error_does_not_delete_live_message():
    # History reports a change, but get() hits a transient 500 (NOT a 404).
    # The message is still live — it must not be turned into a hard-delete,
    # and the cursor must not advance so the next run retries the same window.
    svc = FakeGmailService(
        messages=[_make_message("m")],
        history_response={
            "history": [{"id": "1010", "messagesAdded": [{"message": {"id": "m"}}]}],
            "historyId": "1010",
        },
        get_errors={"m": 500},
    )
    state = {"last_history_id": "1000"}
    with pytest.raises(_HttpError):
        list(incremental_fetch(svc, state))
    assert state["last_history_id"] == "1000"  # cursor NOT advanced


# ---------------------------------------------------------------------------
# gmail_source (dlt wiring) — requires dlt
# ---------------------------------------------------------------------------
def test_gmail_source_resource_is_configured_for_merge_and_hard_delete():
    pytest.importorskip("dlt")

    svc = FakeGmailService(messages=[_make_message("a")], profile_history_id="10")
    resource = gmail_source(service=svc, label_ids=["INBOX"])

    assert resource.name == "gmail_messages"

    # write_disposition, primary key, and the hard-delete marker are all
    # declared on the computed table schema.
    schema = resource.compute_table_schema()
    write_disposition = schema.get("write_disposition")
    if isinstance(write_disposition, dict):  # dlt may normalize to a config dict
        write_disposition = write_disposition.get("disposition")
    assert write_disposition == "merge"

    columns = schema["columns"]
    assert columns["id"].get("primary_key") is True
    assert columns["_deleted"].get("hard_delete") is True


def test_gmail_source_requires_dlt(monkeypatch):
    # Simulate dlt being absent: the factory should raise a helpful ImportError.
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "dlt":
            raise ImportError("no dlt")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="cognee\\[dlt\\]"):
        gmail_source(service=object())
