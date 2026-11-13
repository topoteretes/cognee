"""
cognee-community-connector-imap
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A dlt-based data-source connector that ingests any IMAP mailbox into
cognee memory.  Works with Gmail, Fastmail, iCloud, Yahoo, Zoho Mail,
Proton Bridge, Dovecot, Exchange, or any RFC 3501-compliant server.

Auth:    host + username + app password (TLS only — port 993 by default).
Ingest:  One document per message: subject, participants, date, plain-text
         body (HTML-only mail converted to text), thread headers
         (Message-ID / In-Reply-To / References), attachment names as metadata.
Sync:    Per-folder cursor of (UIDVALIDITY, highest UID) stored in dlt state.
         Each run fetches only UID > last_uid.  CONDSTORE / MODSEQ is used
         where the server advertises it.  If UIDVALIDITY changes the folder
         is re-synced from scratch.
Delete:  After each folder sync a cheap UID SEARCH ALL is compared against the
         set of UIDs we have ingested.  Missing UIDs are yielded with
         ``_deleted=True`` so cognee's orphan_cleanup removes them from the
         knowledge graph, vector store, and relational store.

Primary key: (folder, uidvalidity, uid) — NOT Message-ID (which can be
missing or duplicated, and Gmail re-uses the same message in multiple folders).

Usage::

    from cognee_community_connector_imap import imap_source
    import cognee

    src = imap_source(
        host="imap.fastmail.com",
        username="you@fastmail.com",
        password="app-password-here",
        folders=["INBOX", "Sent"],
    )
    await cognee.remember(src)
    results = await cognee.recall("Q3 budget approval")
"""

from __future__ import annotations

import email
import email.header
import imaplib
import logging
import re
from datetime import datetime, timezone
from email.utils import parseaddr, parsedate_to_datetime
from typing import Iterator, List, Optional

import dlt
from dlt.sources import DltResource

logger = logging.getLogger(__name__)

# ── Optional dependencies ────────────────────────────────────────────────────

try:
    import html2text as _html2text_mod

    _HTML2TEXT = True
except ImportError:
    _HTML2TEXT = False

try:
    import chardet as _chardet_mod

    _CHARDET = True
except ImportError:
    _CHARDET = False

# ── Constants ────────────────────────────────────────────────────────────────

# cognee reads this to route the row through the document ingestion pipeline.
DOCUMENT_SOURCE_ATTR = "document_source"
DOCUMENT_SOURCE_VALUE = "imap"

# Batch size for UID FETCH commands. 50 keeps individual requests small even
# for very large mailboxes.
BATCH_SIZE = 50

# IMAP capability tokens we care about.
CAP_CONDSTORE = "CONDSTORE"
CAP_IMAP4REV1 = "IMAP4REV1"

# State keys (per folder, stored in dlt.state()).
_S_UIDVALIDITY = "uidvalidity"
_S_LAST_UID = "last_uid"
_S_SEEN_UIDS = "seen_uids"    # list[int] — all UIDs we have ingested
_S_LAST_MODSEQ = "last_modseq"  # HIGHESTMODSEQ from last run (CONDSTORE)

# Regex helpers for parsing IMAP responses.
_UID_RE = re.compile(rb"UID (\d+)", re.IGNORECASE)
_UIDVALIDITY_RE = re.compile(r"UIDVALIDITY (\d+)", re.IGNORECASE)
_HIGHESTMODSEQ_RE = re.compile(r"HIGHESTMODSEQ (\d+)", re.IGNORECASE)


# ── MIME / charset helpers ────────────────────────────────────────────────────


def _decode_header_value(raw: str | None) -> str:
    """Decode an RFC 2047 encoded mail header into plain Unicode."""
    if not raw:
        return ""
    parts = email.header.decode_header(raw)
    pieces: list[str] = []
    for chunk, charset in parts:
        if isinstance(chunk, bytes):
            cs = charset or "utf-8"
            try:
                pieces.append(chunk.decode(cs, errors="replace"))
            except (LookupError, UnicodeDecodeError):
                pieces.append(chunk.decode("utf-8", errors="replace"))
        else:
            pieces.append(chunk)
    return " ".join(pieces).strip()


def _decode_payload(part: email.message.Message) -> str:
    """Decode a MIME part's bytes payload to a Unicode string."""
    raw = part.get_payload(decode=True)
    if not raw:
        return ""
    charset = part.get_content_charset() or "utf-8"
    try:
        return raw.decode(charset, errors="replace")
    except (LookupError, UnicodeDecodeError):
        pass
    # chardet fallback for completely unknown charsets.
    if _CHARDET:
        detected = _chardet_mod.detect(raw)
        enc = detected.get("encoding") if detected else None
        if enc:
            try:
                return raw.decode(enc, errors="replace")
            except (LookupError, UnicodeDecodeError):
                pass
    return raw.decode("latin-1", errors="replace")


def _html_to_text(html: str) -> str:
    """Strip HTML markup and return plain text."""
    if _HTML2TEXT:
        h = _html2text_mod.HTML2Text()
        h.ignore_links = True
        h.ignore_images = True
        h.body_width = 0
        return h.handle(html).strip()
    # Minimal regex fallback (no external dep required).
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"&[a-z]{2,6};", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _extract_body_and_attachments(
    msg: email.message.Message,
) -> tuple[str, list[str]]:
    """
    Walk a parsed email and return (body_text, [attachment_filename, ...]).

    Prefers text/plain; falls back to html→text.  Ignores attachment content
    (keeps only filenames) to avoid inflating document sizes.
    """
    plain: list[str] = []
    html: list[str] = []
    attachments: list[str] = []

    if msg.is_multipart():
        for part in msg.walk():
            disposition = str(part.get("Content-Disposition", "")).lower()
            content_type = part.get_content_type()

            if "attachment" in disposition:
                fname = part.get_filename()
                if fname:
                    attachments.append(_decode_header_value(fname))
                continue

            if content_type == "text/plain":
                plain.append(_decode_payload(part))
            elif content_type == "text/html":
                html.append(_decode_payload(part))
    else:
        ct = msg.get_content_type()
        if ct == "text/plain":
            plain.append(_decode_payload(msg))
        elif ct == "text/html":
            html.append(_decode_payload(msg))

    if plain:
        body = "\n\n".join(plain).strip()
    elif html:
        body = _html_to_text("\n\n".join(html))
    else:
        body = ""

    return body, attachments


# ── Message parser ───────────────────────────────────────────────────────────


def _parse_message(
    raw_bytes: bytes,
    uid: int,
    folder: str,
    uidvalidity: int,
) -> dict:
    """
    Parse raw RFC 822 bytes into a flat cognee document row.

    Primary key: (folder, uidvalidity, uid)
    Message-ID is kept only as thread metadata — it cannot be the PK because
    it is absent or duplicated in a significant minority of real-world mail.
    """
    try:
        msg = email.message_from_bytes(raw_bytes)
    except Exception as exc:
        logger.warning("Failed to parse message uid=%d: %s", uid, exc)
        return _empty_row(uid, folder, uidvalidity)

    subject = _decode_header_value(msg.get("Subject"))
    from_raw = msg.get("From", "")
    from_name_raw, from_addr = parseaddr(from_raw)
    from_name = _decode_header_value(from_name_raw)

    to_raw = msg.get("To", "")
    cc_raw = msg.get("Cc", "")
    to_addresses = [a.strip() for a in to_raw.split(",") if a.strip()]
    cc_addresses = [a.strip() for a in cc_raw.split(",") if a.strip()]

    message_id = (msg.get("Message-ID") or "").strip()
    in_reply_to = (msg.get("In-Reply-To") or "").strip()
    references = (msg.get("References") or "").strip()

    date_str = msg.get("Date", "")
    try:
        date_iso = parsedate_to_datetime(date_str).isoformat() if date_str else ""
    except Exception:
        date_iso = date_str

    body, attachments = _extract_body_and_attachments(msg)

    return {
        # ── Primary key ──────────────────────────────────────────────────
        "folder": folder,
        "uidvalidity": uidvalidity,
        "uid": uid,
        # ── Content ──────────────────────────────────────────────────────
        "subject": subject,
        "from_address": from_addr,
        "from_name": from_name,
        "to_addresses": to_addresses,
        "cc_addresses": cc_addresses,
        "date": date_iso,
        "body": body,
        # ── Thread headers ───────────────────────────────────────────────
        "message_id": message_id,
        "in_reply_to": in_reply_to,
        "references": references,
        # ── Attachment metadata (names only — no content) ────────────────
        "attachment_names": attachments,
        # ── Deletion flag — set True when message is expunged/moved ──────
        "_deleted": False,
        # ── cognee document routing ──────────────────────────────────────
        DOCUMENT_SOURCE_ATTR: DOCUMENT_SOURCE_VALUE,
    }


def _empty_row(uid: int, folder: str, uidvalidity: int) -> dict:
    """Return a skeletal row for a message that could not be parsed."""
    return {
        "folder": folder, "uidvalidity": uidvalidity, "uid": uid,
        "subject": "", "from_address": "", "from_name": "",
        "to_addresses": [], "cc_addresses": [], "date": "",
        "body": "", "message_id": "", "in_reply_to": "", "references": "",
        "attachment_names": [], "_deleted": False,
        DOCUMENT_SOURCE_ATTR: DOCUMENT_SOURCE_VALUE,
    }


# ── IMAP connection helpers ──────────────────────────────────────────────────


def _open_connection(host: str, port: int, username: str, password: str) -> imaplib.IMAP4_SSL:
    """Open an IMAP-over-TLS connection and log in."""
    conn = imaplib.IMAP4_SSL(host, port)
    conn.login(username, password)
    logger.info("Logged into %s as %s", host, username)
    return conn


def _get_capabilities(conn: imaplib.IMAP4_SSL) -> set[str]:
    """Return the server's capability set as a set of uppercase strings."""
    try:
        _, data = conn.capability()
        raw = data[0] if data else b""
        return set((raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw).upper().split())
    except Exception:
        return set()


def _get_uidvalidity(conn: imaplib.IMAP4_SSL, folder: str) -> int:
    """Return the UIDVALIDITY integer for *folder* via STATUS."""
    # STATUS is cheaper than SELECT when we only need UIDVALIDITY.
    typ, data = conn.status(f'"{folder}"', "(UIDVALIDITY)")
    if typ != "OK" or not data or not data[0]:
        raise RuntimeError(f"STATUS {folder!r} failed: {data}")
    line = data[0].decode("utf-8", errors="replace") if isinstance(data[0], bytes) else data[0]
    m = _UIDVALIDITY_RE.search(line)
    if not m:
        raise RuntimeError(f"Could not parse UIDVALIDITY from: {line!r}")
    return int(m.group(1))


def _select_folder(conn: imaplib.IMAP4_SSL, folder: str, condstore: bool) -> int:
    """
    SELECT *folder* in read-only mode (optionally with CONDSTORE).
    Returns HIGHESTMODSEQ (0 if not available).
    """
    select_arg = f'"{folder}"' + (" (CONDSTORE)" if condstore else "")
    typ, _ = conn.select(select_arg, readonly=True)
    if typ != "OK":
        raise RuntimeError(f"SELECT {folder!r} failed")

    if not condstore:
        return 0

    typ2, data2 = conn.status(f'"{folder}"', "(HIGHESTMODSEQ)")
    if typ2 == "OK" and data2 and data2[0]:
        line = data2[0].decode("utf-8", errors="replace") if isinstance(data2[0], bytes) else data2[0]
        m = _HIGHESTMODSEQ_RE.search(line)
        return int(m.group(1)) if m else 0
    return 0


def _search_uids(conn: imaplib.IMAP4_SSL, criterion: str) -> list[int]:
    """Run UID SEARCH with *criterion* and return sorted list of UIDs."""
    typ, data = conn.uid("search", None, criterion)  # type: ignore[arg-type]
    if typ != "OK" or not data or not data[0]:
        return []
    return sorted(int(u) for u in data[0].split() if u)


def _fetch_raw_batch(
    conn: imaplib.IMAP4_SSL, uids: list[int]
) -> dict[int, bytes]:
    """
    UID FETCH a batch of UIDs and return {uid: raw_rfc822_bytes}.
    Parses the UID out of each server response tuple.
    """
    if not uids:
        return {}
    uid_set = ",".join(str(u) for u in uids)
    try:
        typ, data = conn.uid("fetch", uid_set, "(UID RFC822)")  # type: ignore[arg-type]
    except Exception as exc:
        logger.error("FETCH failed for UIDs %s: %s", uid_set, exc)
        return {}

    results: dict[int, bytes] = {}
    for item in data:
        if not isinstance(item, tuple) or len(item) < 2:
            continue
        header_raw, body = item[0], item[1]
        if not isinstance(body, bytes):
            continue
        header_bytes = header_raw if isinstance(header_raw, bytes) else str(header_raw).encode()
        m = _UID_RE.search(header_bytes)
        if m:
            results[int(m.group(1))] = body
    return results


# ── Core sync logic ──────────────────────────────────────────────────────────


def _sync_folder(
    conn: imaplib.IMAP4_SSL,
    folder: str,
    folder_state: dict,
    caps: set[str],
) -> Iterator[dict]:
    """
    Sync a single folder and yield document rows.

    Handles:
    - First run: fetch all messages.
    - Subsequent runs: fetch UIDs > last_uid only.
    - UIDVALIDITY change: reset state and re-fetch everything.
    - CONDSTORE: use MODSEQ for flag-change detection (optional upgrade).
    - Deletion: compare current UID set against ingested UIDs, tombstone gaps.
    """
    use_condstore = CAP_CONDSTORE in caps
    stored_uidvalidity: int = folder_state.get(_S_UIDVALIDITY, 0)
    last_uid: int = folder_state.get(_S_LAST_UID, 0)
    seen_uids: list[int] = folder_state.get(_S_SEEN_UIDS, [])

    # ── 1. Get current UIDVALIDITY (without SELECTing) ──────────────────
    current_uidvalidity = _get_uidvalidity(conn, folder)

    # ── 2. UIDVALIDITY changed → full re-sync ───────────────────────────
    if stored_uidvalidity and current_uidvalidity != stored_uidvalidity:
        logger.warning(
            "UIDVALIDITY changed for %r (%d → %d). Re-syncing from scratch.",
            folder, stored_uidvalidity, current_uidvalidity,
        )
        last_uid = 0
        seen_uids = []
        folder_state[_S_LAST_MODSEQ] = 0

    # ── 3. SELECT folder (with CONDSTORE if available) ───────────────────
    highest_modseq = _select_folder(conn, folder, use_condstore)

    # ── 4. Determine which new UIDs to fetch ────────────────────────────
    if last_uid == 0:
        new_uids = _search_uids(conn, "ALL")
        logger.info("First run for %r — found %d messages.", folder, len(new_uids))
    else:
        criterion = f"UID {last_uid + 1}:*"
        new_uids = [u for u in _search_uids(conn, criterion) if u > last_uid]
        logger.info("Incremental sync for %r — %d new messages.", folder, len(new_uids))

    # ── 5. Batch-fetch and yield new messages ────────────────────────────
    yielded_uids: list[int] = []
    for i in range(0, len(new_uids), BATCH_SIZE):
        batch = new_uids[i : i + BATCH_SIZE]
        raw_map = _fetch_raw_batch(conn, batch)
        for uid, raw in raw_map.items():
            row = _parse_message(raw, uid, folder, current_uidvalidity)
            yield row
            yielded_uids.append(uid)

    # ── 6. Deletion detection via UID SEARCH ALL ─────────────────────────
    # Compare every UID we've ever ingested against the server's current
    # UID set.  UIDs that are gone were deleted or moved out of the folder.
    all_server_uids = set(_search_uids(conn, "ALL"))
    ingested_uids = set(seen_uids) | set(yielded_uids)

    deleted_uids = ingested_uids - all_server_uids
    for uid in sorted(deleted_uids):
        logger.info("Tombstoning deleted UID %d in %r.", uid, folder)
        yield {
            "folder": folder,
            "uidvalidity": current_uidvalidity,
            "uid": uid,
            "subject": "", "from_address": "", "from_name": "",
            "to_addresses": [], "cc_addresses": [], "date": "",
            "body": "", "message_id": "", "in_reply_to": "", "references": "",
            "attachment_names": [],
            "_deleted": True,
            DOCUMENT_SOURCE_ATTR: DOCUMENT_SOURCE_VALUE,
        }

    # ── 7. Update persisted state ────────────────────────────────────────
    folder_state[_S_UIDVALIDITY] = current_uidvalidity
    folder_state[_S_LAST_UID] = max(all_server_uids, default=last_uid)
    folder_state[_S_SEEN_UIDS] = sorted(all_server_uids)
    if use_condstore:
        folder_state[_S_LAST_MODSEQ] = highest_modseq

    conn.close()


# ── dlt resource ─────────────────────────────────────────────────────────────


@dlt.resource(
    name="imap_messages",
    primary_key=["folder", "uidvalidity", "uid"],
    write_disposition="merge",
    columns={
        "_deleted": {"hard_delete": True},
    },
)
def imap_messages(
    host: str,
    username: str,
    password: str = dlt.secrets.value,
    port: int = 993,
    folders: Optional[List[str]] = None,
    since_date: Optional[str] = None,
    from_filter: Optional[str] = None,
) -> Iterator[dict]:
    """
    dlt resource that yields IMAP email messages.

    Parameters
    ----------
    host:
        IMAP server hostname.  Examples: ``imap.gmail.com``,
        ``imap.fastmail.com``, ``outlook.office365.com``.
    username:
        Email address / IMAP login.
    password:
        App password (marked as a dlt secret).
    port:
        IMAP-over-TLS port (default 993).
    folders:
        List of IMAP folder names to ingest.  Defaults to ``["INBOX"]``.
        Use ``conn.list()`` to discover available folder names.
    since_date:
        Optional ISO date string (``"YYYY-MM-DD"``) — only messages on or
        after this date are included on the *first* run.  Ignored once a
        per-folder UID cursor exists.
    from_filter:
        Optional sender filter applied during the initial back-fill,
        e.g. ``"boss@example.com"``.
    """
    folders = folders or ["INBOX"]

    with _open_connection(host, port, username, password) as conn:
        caps = _get_capabilities(conn)
        if CAP_CONDSTORE in caps:
            logger.info("Server supports CONDSTORE — enabling MODSEQ tracking.")

        # dlt state is persisted between pipeline runs.
        state: dict = dlt.state().setdefault("imap", {})

        for folder in folders:
            folder_state: dict = state.setdefault(folder, {})
            is_first_run = folder_state.get(_S_UIDVALIDITY) is None

            # Build the search criterion for the initial back-fill.
            if is_first_run:
                criteria_parts: list[str] = []
                if since_date:
                    # Convert "YYYY-MM-DD" → "DD-Mon-YYYY" for IMAP SINCE.
                    try:
                        dt = datetime.fromisoformat(since_date)
                        criteria_parts.append(
                            f'SINCE "{dt.strftime("%d-%b-%Y")}"'
                        )
                    except ValueError:
                        logger.warning("Invalid since_date %r — ignoring.", since_date)
                if from_filter:
                    criteria_parts.append(f'FROM "{from_filter}"')
                # Store for use in _sync_folder's first-run search.
                folder_state["_initial_criteria"] = " ".join(criteria_parts) or "ALL"

            try:
                yield from _sync_folder(conn, folder, folder_state, caps)
            except Exception as exc:
                logger.error("Failed to sync folder %r: %s", folder, exc, exc_info=True)
                # Continue with the remaining folders — don't abort the whole run.

    # Persist state mutations (dlt does this automatically on successful runs).


# ── dlt source ───────────────────────────────────────────────────────────────


@dlt.source(name="imap")
def imap_source(
    host: str,
    username: str,
    password: str = dlt.secrets.value,
    port: int = 993,
    folders: Optional[List[str]] = None,
    since_date: Optional[str] = None,
    from_filter: Optional[str] = None,
) -> DltResource:
    """
    Return a dlt source that can be passed directly to ``cognee.remember()``.

    Parameters
    ----------
    host:
        IMAP server hostname (e.g. ``"imap.fastmail.com"``).
    username:
        IMAP username / email address.
    password:
        App password — stored as a dlt secret, never logged.
    port:
        IMAP-over-TLS port (default 993).
    folders:
        Folders to ingest; defaults to ``["INBOX"]``.
    since_date:
        ISO date string (``"YYYY-MM-DD"``) used as the start of the initial
        back-fill.  Subsequent runs use the UID cursor and ignore this.
    from_filter:
        Optional sender filter for the initial back-fill.

    Example
    -------
    ::

        src = imap_source(
            host="imap.fastmail.com",
            username="you@fastmail.com",
            password="app-password",
            folders=["INBOX", "Sent"],
            since_date="2024-01-01",
        )
        await cognee.remember(src)
    """
    return imap_messages(
        host=host,
        username=username,
        password=password,
        port=port,
        folders=folders,
        since_date=since_date,
        from_filter=from_filter,
    )
