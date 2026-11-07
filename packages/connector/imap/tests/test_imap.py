"""
Unit tests for cognee-community-connector-imap.

All IMAP interactions are mocked — no real mail server needed.
Run:  pytest tests/ -v
"""

from __future__ import annotations

import email
import textwrap
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from unittest.mock import MagicMock, patch, call
import pytest

from cognee_community_connector_imap.imap import (
    DOCUMENT_SOURCE_ATTR,
    DOCUMENT_SOURCE_VALUE,
    _decode_header_value,
    _decode_payload,
    _extract_body_and_attachments,
    _html_to_text,
    _parse_message,
    _search_uids,
    _fetch_raw_batch,
    _get_uidvalidity,
    _sync_folder,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_plain_email(
    subject="Hello",
    from_addr="alice@example.com",
    to_addr="bob@example.com",
    body="Hello world",
    msg_id="<abc@example.com>",
    in_reply_to="",
    date="Mon, 01 Jan 2024 12:00:00 +0000",
) -> bytes:
    msg = email.message.Message()
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to_addr
    msg["Message-ID"] = msg_id
    msg["In-Reply-To"] = in_reply_to
    msg["Date"] = date
    msg.set_payload(body, charset="utf-8")
    msg.set_type("text/plain")
    return msg.as_bytes()


def _make_html_email(
    subject="HTML mail",
    from_addr="alice@example.com",
    body_html="<h1>Hello</h1><p>World</p>",
) -> bytes:
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = "bob@example.com"
    msg["Date"] = "Mon, 01 Jan 2024 12:00:00 +0000"
    msg.attach(MIMEText(body_html, "html", "utf-8"))
    return msg.as_bytes()


def _make_multipart_email_with_attachment() -> bytes:
    msg = MIMEMultipart("mixed")
    msg["Subject"] = "Report"
    msg["From"] = "alice@example.com"
    msg["To"] = "bob@example.com"
    msg["Date"] = "Mon, 01 Jan 2024 12:00:00 +0000"
    msg.attach(MIMEText("See attached report.", "plain", "utf-8"))
    attachment = MIMEBase("application", "octet-stream")
    attachment.set_payload(b"fake pdf bytes")
    encoders.encode_base64(attachment)
    attachment.add_header("Content-Disposition", "attachment", filename="report.pdf")
    msg.attach(attachment)
    return msg.as_bytes()


def _make_encoded_header_email(subject_encoded: str) -> bytes:
    """Email with an RFC 2047 encoded Subject header."""
    msg = email.message.Message()
    msg["Subject"] = subject_encoded
    msg["From"] = "alice@example.com"
    msg["To"] = "bob@example.com"
    msg["Date"] = "Mon, 01 Jan 2024 12:00:00 +0000"
    msg.set_payload("Body text", charset="utf-8")
    msg.set_type("text/plain")
    return msg.as_bytes()


def _make_mock_conn(
    uidvalidity=12345,
    uids=None,
    raw_messages=None,
    caps=frozenset(),
    highestmodseq=0,
):
    """Return a mock imaplib.IMAP4_SSL with sensible defaults."""
    conn = MagicMock()
    uids = uids or [1, 2, 3]
    raw_messages = raw_messages or {u: _make_plain_email(subject=f"Mail {u}") for u in uids}

    # capability()
    cap_str = " ".join(caps | {"IMAP4rev1"})
    conn.capability.return_value = ("OK", [cap_str.encode()])

    # status() — returns UIDVALIDITY and optionally HIGHESTMODSEQ
    def _status(folder, items):
        s = f'"{folder}" (UIDVALIDITY {uidvalidity}'
        if "HIGHESTMODSEQ" in items:
            s += f" HIGHESTMODSEQ {highestmodseq}"
        s += ")"
        return ("OK", [s.encode()])
    conn.status.side_effect = _status

    # select()
    conn.select.return_value = ("OK", [b"3"])

    # uid("search", ...) → UID list
    def _uid(cmd, *args):
        if cmd == "search":
            uid_bytes = b" ".join(str(u).encode() for u in uids)
            return ("OK", [uid_bytes])
        if cmd == "fetch":
            uid_arg = args[0]
            requested = [int(u) for u in uid_arg.split(b",") if u]
            data = []
            for u in requested:
                raw = raw_messages.get(u, b"")
                header = f"{u} (UID {u} RFC822 {{{len(raw)}}}".encode()
                data.append((header, raw))
                data.append(b")")
            return ("OK", data)
        return ("OK", [b""])
    conn.uid.side_effect = _uid

    conn.close.return_value = ("OK", [])
    conn.__enter__ = lambda s: s
    conn.__exit__ = MagicMock(return_value=False)
    return conn


# ── _decode_header_value ──────────────────────────────────────────────────────

class TestDecodeHeaderValue:
    def test_plain_ascii(self):
        assert _decode_header_value("Hello World") == "Hello World"

    def test_none_returns_empty(self):
        assert _decode_header_value(None) == ""

    def test_empty_string(self):
        assert _decode_header_value("") == ""

    def test_rfc2047_utf8_encoded(self):
        # "=?utf-8?b?SGVsbG8gV29ybGQ=?=" decodes to "Hello World"
        encoded = "=?utf-8?b?SGVsbG8gV29ybGQ=?="
        assert _decode_header_value(encoded) == "Hello World"

    def test_rfc2047_latin1_encoded(self):
        # "=?iso-8859-1?q?caf=E9?=" decodes to "café"
        encoded = "=?iso-8859-1?q?caf=E9?="
        result = _decode_header_value(encoded)
        assert "caf" in result  # charset may differ but content is right


# ── _extract_body_and_attachments ─────────────────────────────────────────────

class TestExtractBody:
    def test_plain_text_email(self):
        raw = _make_plain_email(body="Hello world")
        msg = email.message_from_bytes(raw)
        body, attachments = _extract_body_and_attachments(msg)
        assert "Hello world" in body
        assert attachments == []

    def test_html_only_email_converted_to_text(self):
        raw = _make_html_email(body_html="<h1>Title</h1><p>Content here.</p>")
        msg = email.message_from_bytes(raw)
        body, attachments = _extract_body_and_attachments(msg)
        assert "Title" in body or "Content here" in body
        assert "<h1>" not in body  # HTML tags stripped

    def test_attachment_names_extracted(self):
        raw = _make_multipart_email_with_attachment()
        msg = email.message_from_bytes(raw)
        body, attachments = _extract_body_and_attachments(msg)
        assert "See attached report" in body
        assert "report.pdf" in attachments

    def test_empty_email(self):
        msg = email.message.Message()
        msg.set_type("text/plain")
        body, attachments = _extract_body_and_attachments(msg)
        assert body == ""
        assert attachments == []


# ── _parse_message ────────────────────────────────────────────────────────────

class TestParseMessage:
    def test_primary_key_fields(self):
        raw = _make_plain_email()
        row = _parse_message(raw, uid=42, folder="INBOX", uidvalidity=99)
        assert row["uid"] == 42
        assert row["folder"] == "INBOX"
        assert row["uidvalidity"] == 99

    def test_subject_and_sender(self):
        raw = _make_plain_email(subject="Meeting notes", from_addr="alice@example.com")
        row = _parse_message(raw, uid=1, folder="INBOX", uidvalidity=1)
        assert row["subject"] == "Meeting notes"
        assert row["from_address"] == "alice@example.com"

    def test_thread_headers_preserved(self):
        raw = _make_plain_email(
            msg_id="<msg001@example.com>",
            in_reply_to="<msg000@example.com>",
        )
        row = _parse_message(raw, uid=1, folder="INBOX", uidvalidity=1)
        assert row["message_id"] == "<msg001@example.com>"
        assert row["in_reply_to"] == "<msg000@example.com>"

    def test_document_source_attr_set(self):
        raw = _make_plain_email()
        row = _parse_message(raw, uid=1, folder="INBOX", uidvalidity=1)
        assert row[DOCUMENT_SOURCE_ATTR] == DOCUMENT_SOURCE_VALUE

    def test_deleted_flag_false_by_default(self):
        raw = _make_plain_email()
        row = _parse_message(raw, uid=1, folder="INBOX", uidvalidity=1)
        assert row["_deleted"] is False

    def test_malformed_bytes_returns_skeleton(self):
        row = _parse_message(b"not valid email bytes at all !!!!", uid=5, folder="INBOX", uidvalidity=1)
        # Should not raise; returns a usable skeleton row.
        assert row["uid"] == 5
        assert row["folder"] == "INBOX"
        assert row[DOCUMENT_SOURCE_ATTR] == DOCUMENT_SOURCE_VALUE

    def test_date_parsed_to_iso(self):
        raw = _make_plain_email(date="Mon, 01 Jan 2024 12:00:00 +0000")
        row = _parse_message(raw, uid=1, folder="INBOX", uidvalidity=1)
        assert "2024" in row["date"]

    def test_encoded_subject_decoded(self):
        # =?utf-8?b?TGF0ZWJyZWFraW5nIG5ld3M=?= → "Latebreaking news"
        encoded_subj = "=?utf-8?b?TGF0ZWJyZWFraW5nIG5ld3M=?="
        raw = _make_encoded_header_email(encoded_subj)
        row = _parse_message(raw, uid=1, folder="INBOX", uidvalidity=1)
        assert "Latebreaking news" in row["subject"]


# ── _get_uidvalidity ──────────────────────────────────────────────────────────

class TestGetUidvalidity:
    def test_parses_correctly(self):
        conn = MagicMock()
        conn.status.return_value = ("OK", [b'"INBOX" (UIDVALIDITY 99887766)'])
        result = _get_uidvalidity(conn, "INBOX")
        assert result == 99887766

    def test_raises_on_bad_response(self):
        conn = MagicMock()
        conn.status.return_value = ("NO", [b""])
        with pytest.raises(RuntimeError):
            _get_uidvalidity(conn, "INBOX")


# ── _sync_folder — incremental cursor ────────────────────────────────────────

class TestSyncFolderIncremental:
    def test_first_run_fetches_all_messages(self):
        uids = [1, 2, 3]
        conn = _make_mock_conn(uidvalidity=100, uids=uids)
        folder_state: dict = {}
        rows = list(_sync_folder(conn, "INBOX", folder_state, caps=set()))
        fetched = [r for r in rows if not r["_deleted"]]
        assert len(fetched) == 3
        assert folder_state["uidvalidity"] == 100
        assert folder_state["last_uid"] == 3

    def test_incremental_run_only_fetches_new_uids(self):
        all_uids = [1, 2, 3, 4, 5]
        conn = _make_mock_conn(uidvalidity=100, uids=all_uids)

        # Simulate: UIDs 1–3 already ingested on a previous run.
        folder_state: dict = {
            "uidvalidity": 100,
            "last_uid": 3,
            "seen_uids": [1, 2, 3],
        }

        # Override uid("search") to simulate UID > 3 search returning [4, 5].
        original_uid = conn.uid.side_effect
        def patched_uid(cmd, *args):
            if cmd == "search":
                criterion = args[1] if len(args) > 1 else args[0]
                if criterion == "ALL":
                    return ("OK", [b"1 2 3 4 5"])
                # UID 4:* search
                return ("OK", [b"4 5"])
            return original_uid(cmd, *args)
        conn.uid.side_effect = patched_uid

        rows = list(_sync_folder(conn, "INBOX", folder_state, caps=set()))
        new_msgs = [r for r in rows if not r["_deleted"]]
        assert len(new_msgs) == 2
        assert {r["uid"] for r in new_msgs} == {4, 5}


# ── _sync_folder — UIDVALIDITY reset ─────────────────────────────────────────

class TestSyncFolderUidvalidityReset:
    def test_uidvalidity_change_triggers_full_resync(self):
        uids = [10, 11, 12]
        conn = _make_mock_conn(uidvalidity=999, uids=uids)

        # State has OLD uidvalidity.
        folder_state: dict = {
            "uidvalidity": 111,  # differs from current 999
            "last_uid": 5,
            "seen_uids": [1, 2, 3, 4, 5],
        }
        rows = list(_sync_folder(conn, "INBOX", folder_state, caps=set()))
        # All 3 current messages should be fetched (full re-sync).
        new_msgs = [r for r in rows if not r["_deleted"]]
        assert len(new_msgs) == 3
        # State updated to new uidvalidity.
        assert folder_state["uidvalidity"] == 999


# ── _sync_folder — deletion detection ────────────────────────────────────────

class TestSyncFolderDeletion:
    def test_deleted_uids_are_tombstoned(self):
        # Server has UIDs [1, 3] — UID 2 was deleted.
        conn = _make_mock_conn(uidvalidity=100, uids=[1, 3])

        folder_state: dict = {
            "uidvalidity": 100,
            "last_uid": 3,
            "seen_uids": [1, 2, 3],  # we had UID 2 previously
        }

        def patched_uid(cmd, *args):
            if cmd == "search":
                return ("OK", [b"1 3"])  # UID 2 is gone
            # fetch UIDs > 3 (incremental) → empty
            if cmd == "fetch":
                uid_arg = args[0]
                raw_map = {
                    1: _make_plain_email(subject="Mail 1"),
                    3: _make_plain_email(subject="Mail 3"),
                }
                data = []
                for u_str in uid_arg.split(b","):
                    u = int(u_str)
                    raw = raw_map.get(u, b"")
                    header = f"{u} (UID {u} RFC822 {{{len(raw)}}}".encode()
                    data.append((header, raw))
                    data.append(b")")
                return ("OK", data)
            return ("OK", [b""])
        conn.uid.side_effect = patched_uid

        rows = list(_sync_folder(conn, "INBOX", folder_state, caps=set()))
        deleted = [r for r in rows if r["_deleted"]]
        assert len(deleted) == 1
        assert deleted[0]["uid"] == 2

    def test_no_false_deletions_when_nothing_changed(self):
        uids = [1, 2, 3]
        conn = _make_mock_conn(uidvalidity=100, uids=uids)
        folder_state: dict = {
            "uidvalidity": 100,
            "last_uid": 3,
            "seen_uids": [1, 2, 3],
        }

        def patched_uid(cmd, *args):
            if cmd == "search":
                return ("OK", [b"1 2 3"])
            if cmd == "fetch":
                return ("OK", [])  # no new messages
            return ("OK", [b""])
        conn.uid.side_effect = patched_uid

        rows = list(_sync_folder(conn, "INBOX", folder_state, caps=set()))
        deleted = [r for r in rows if r["_deleted"]]
        assert deleted == []
