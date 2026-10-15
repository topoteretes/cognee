# cognee-community-connector-imap

A **generic IMAP** data-source connector for [cognee](https://github.com/topoteretes/cognee).
Works with any RFC 3501-compliant mailbox: Gmail, Fastmail, iCloud, Yahoo,
Zoho Mail, Proton via Bridge, a self-hosted Dovecot/Exchange, or a university
IMAP server — with nothing more than an app password.

| Feature | Detail |
|---------|--------|
| Auth | Host + username + app password (TLS / port 993) |
| Ingest | Subject, participants, date, plain-text body (HTML→text), thread headers, attachment names |
| Incremental sync | Per-folder `UIDVALIDITY + UID` cursor in dlt state |
| CONDSTORE | Optional MODSEQ tracking when advertised by the server |
| UIDVALIDITY reset | Detected and handled with automatic full re-sync |
| Forget-on-delete | `UID SEARCH ALL` diff → `_deleted=True` tombstones → cognee orphan_cleanup |
| Primary key | `(folder, uidvalidity, uid)` — not `Message-ID` (unreliable) |

---

## Installation

```bash
# Minimal (uses stdlib imaplib + email only)
pip install "git+https://github.com/topoteretes/cognee-community.git#subdirectory=packages/connector/imap"

# Recommended: better HTML stripping and charset detection
pip install "git+https://github.com/topoteretes/cognee-community.git#subdirectory=packages/connector/imap[full]"
```

Or from a cloned repo:
```bash
cd packages/connector/imap
uv pip install -e ".[full]"
```

---

## Quick start

```python
import asyncio
import cognee
from cognee_community_connector_imap import imap_source

async def main():
    src = imap_source(
        host="imap.fastmail.com",
        username="you@fastmail.com",
        password="your-app-password",    # see below
        folders=["INBOX", "Sent"],
        since_date="2024-01-01",         # optional ISO date
    )
    await cognee.remember(src)
    results = await cognee.recall("budget approval")
    print(results)

asyncio.run(main())
```

See [`examples/example.py`](examples/example.py) for an env-var-driven script.

---

## Getting an app password

### Gmail
1. Enable 2-Step Verification: [myaccount.google.com/security](https://myaccount.google.com/security)
2. Go to **App Passwords** → generate one for "Mail" on "Other device"
3. Host: `imap.gmail.com` · Port: `993`
4. ⚠️ Gmail labels map to IMAP folders. Selecting `INBOX` plus `[Gmail]/All Mail`
   will ingest the same message twice. Recommend: use `INBOX` only (or
   `[Gmail]/All Mail` only) to avoid duplicates.

### Fastmail
1. Go to **Settings → Privacy & Security → App Passwords**
2. Create one with "IMAP / POP" access
3. Host: `imap.fastmail.com` · Port: `993`

### iCloud
1. Enable 2FA on your Apple ID
2. Go to [appleid.apple.com](https://appleid.apple.com) → **Sign-In and Security → App-Specific Passwords**
3. Host: `imap.mail.me.com` · Port: `993`

### Yahoo Mail
1. Go to **Account Security** in your Yahoo account
2. Enable **Allow apps that use less secure sign in** and generate an app password
3. Host: `imap.mail.yahoo.com` · Port: `993`

### Zoho Mail
1. Go to **Settings → Security → App-Specific Passwords**
2. Host: `imap.zoho.com` · Port: `993`

### Proton Mail (via Bridge)
1. Install [Proton Mail Bridge](https://proton.me/mail/bridge) and sign in
2. Host: `127.0.0.1` · Port: `1143` (Bridge runs locally)
3. Password: the Bridge-generated IMAP password (not your Proton account password)

### Self-hosted (Dovecot / Exchange IMAP)
Use your regular username and password (or an app-specific one if your server
supports it). Host and port are whatever your mail admin set up.

---

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `host` | `str` | required | IMAP server hostname |
| `username` | `str` | required | Login / email address |
| `password` | `str` | dlt secret | App password |
| `port` | `int` | `993` | IMAP-over-TLS port |
| `folders` | `list[str]` | `["INBOX"]` | Folders to ingest |
| `since_date` | `str` | `None` | ISO date (`"YYYY-MM-DD"`) for initial back-fill only |
| `from_filter` | `str` | `None` | Sender filter for initial back-fill |

Store secrets in `~/.dlt/secrets.toml`:
```toml
[sources.imap]
password = "your-app-password"
```

---

## How it works

### First run — back-fill
All messages in each selected folder matching `since_date` / `from_filter`
are fetched via batched `UID FETCH` (50 messages per request).

### Subsequent runs — incremental sync
The `(UIDVALIDITY, highest_uid)` cursor is stored per folder in dlt state.
Only `UID > last_uid` messages are fetched. On servers that advertise
`CONDSTORE`, the `HIGHESTMODSEQ` is also stored; flag changes (read/star/move)
can be picked up via `UID FETCH (CHANGEDSINCE modseq)`.

### UIDVALIDITY reset
On every run, the server's current `UIDVALIDITY` is compared with the stored
value. If it changed (server was rebuilt, folder recreated), the cursor is
reset to zero and all messages are re-fetched from scratch. Stale rows from
the previous UID space are tombstoned.

### Forget-on-delete
After each folder sync, `UID SEARCH ALL` (a single cheap command) returns the
current UID set. Any UIDs in our ingested set that are missing from the server
were deleted or moved out of the folder. The connector emits a row with
`_deleted=True` for each of these; cognee's `orphan_cleanup` removes them from
the knowledge graph, vector store, and relational store.

### Primary key
Each message is identified by `(folder, uidvalidity, uid)`. `Message-ID` is
kept as thread metadata only — it is absent or duplicated in many real-world
messages, and Gmail re-exposes the same message under multiple folder paths.

---

## Running the tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

All tests run without a real mail server (imaplib is mocked via `unittest.mock`).

---

## Contributing

This package lives in
[topoteretes/cognee-community](https://github.com/topoteretes/cognee-community).
Open issues and PRs there.
