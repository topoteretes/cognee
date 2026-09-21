"""Sync a connected Google Drive account's files into cognee memory as text.

The textual mirror of the Linear adapter's issue sync rather than of GitHub's
repository sync: files are rendered to a stable plain-text form and handed to
the ordinary ``remember()`` path, so their contents become searchable
knowledge instead of a code graph.

Everything is turned into text here rather than downloaded to disk and left
to the loaders, for two reasons. Google-native documents (Docs, Sheets,
Slides) have no bytes to download at all and must be exported, so an export
step is unavoidable in any case; and the loaders that read PDF and DOCX sit
behind the ``docs`` extra, so a default install would silently skip most of a
real Drive. Exporting what Google can render and skipping the rest is honest
about what was indexed, and the skip count says how much was left out.

One dataset per connected account (``google_drive_<email>``), never one
shared per Workspace domain. Cognee's permissions are dataset-scoped, so a
shared dataset would let one colleague's questions answer from another's
private files. A shared company index is a separate piece of work that starts
with per-document ACLs, not with this connector.
"""

import logging
import re
from hashlib import sha256
from typing import Any

from cognee.modules.integrations.google_drive import client
from cognee.modules.integrations.models.IntegrationCredential import IntegrationCredential

logger = logging.getLogger(__name__)

GOOGLE_DRIVE_DATASET_PREFIX = "google_drive"

# Spelled out rather than imported from the adapter, which imports this module
# for its post-install hook.
GOOGLE_DRIVE_PROVIDER = "google_drive"

# Google-native types carry no bytes; each is exported in the text form that
# keeps the most meaning. Anything not listed here is downloaded as-is when
# it is already text, and skipped otherwise.
_EXPORT_AS = {
    "application/vnd.google-apps.document": "text/plain",
    "application/vnd.google-apps.spreadsheet": "text/csv",
    "application/vnd.google-apps.presentation": "text/plain",
}

# Stored files worth downloading directly. Deliberately narrow: a type that
# needs a parser belongs to the loaders, not to this module.
_DOWNLOADABLE_PREFIXES = ("text/",)
_DOWNLOADABLE_TYPES = {"application/json", "application/xml"}

# A first pass indexes the most recently touched files rather than everything.
# This counts rendered documents only, so on its own it bounds nothing: a Drive
# full of images never reaches it. _MAX_PAGES is what actually ends the walk.
_DEFAULT_FILE_LIMIT = 200

# The real bound on the listing walk. Every page costs one request whatever it
# contains, so capping pages caps the sync regardless of how much of the Drive
# this connector can render, and it also ends a listing whose page token never
# advances. 50 pages is 5000 files at Drive's page size.
_MAX_PAGES = 50

# Guards against one pathological export (a spreadsheet with a hundred
# thousand rows) dominating an account's whole memory.
_MAX_CHARS_PER_FILE = 200_000

SYNC_STATUS_OK = "ok"
SYNC_STATUS_DEGRADED = "degraded"


def dataset_name_for_account(email: str, account_id: str = "") -> str:
    """The one dataset a connected account's files land in.

    The email is slugged for readability but is not unique on its own: Gmail
    treats dots as insignificant, and the slug folds dots, plus signs and
    hyphens into the same underscore, so ``john.doe@`` and ``john_doe@`` are
    two different Google accounts that render identically. A digest of the
    account id (the Google subject, which is what the credential is keyed on)
    disambiguates them. A digest rather than a slice of the id itself, because
    Google subjects are long numbers sharing a prefix, so any truncation is a
    collision waiting for the two accounts that differ outside the window.
    """
    slug = re.sub(r"[^A-Za-z0-9_]+", "_", email or "").strip("_").lower()
    name = f"{GOOGLE_DRIVE_DATASET_PREFIX}_{slug or 'account'}"
    if not account_id:
        return name
    return f"{name}_{sha256(account_id.encode()).hexdigest()[:10]}"


def format_file(file: dict[str, Any], content: str) -> str:
    """A stable plain-text rendering of one file.

    Name, link, type, then contents. Nothing volatile (no modified-at, no
    size), so an unchanged file renders to byte-identical text and a re-sync
    does not churn the graph with cosmetic differences.
    """
    lines = [
        f"Google Drive file: {file.get('name') or 'untitled'}",
        f"URL: {file.get('webViewLink') or ''}",
        f"Type: {file.get('mimeType') or 'unknown'}",
        "",
        content.strip(),
    ]
    return "\n".join(lines)


async def _read_file(access_token: str, file: dict[str, Any]) -> str | None:
    """The file's text, or ``None`` when it is not something we can render."""
    mime_type = file.get("mimeType") or ""

    export_as = _EXPORT_AS.get(mime_type)
    if export_as:
        return await client.export_file(access_token, file["id"], export_as)

    if mime_type.startswith(_DOWNLOADABLE_PREFIXES) or mime_type in _DOWNLOADABLE_TYPES:
        return await client.download_file(access_token, file["id"])

    return None


def _dataset_name(credential: IntegrationCredential) -> str:
    email = (credential.provider_metadata or {}).get("email") or ""
    return dataset_name_for_account(email, str(credential.provider_account_id))


async def sync_drive(
    credential: IntegrationCredential, file_limit: int = _DEFAULT_FILE_LIMIT
) -> None:
    """Index the account's most recently modified files, one batch.

    Runs to completion; callers are already off the request path (the
    post-install hook runs detached), so there is nothing to hand off to.

    One ``remember()`` call for the whole batch rather than one per file: a
    single pipeline run over a list is far cheaper than two hundred, and it
    keeps a slow account from holding the ingestion path open all day.

    Nothing re-runs this sync: Drive registers no webhook verifier and the
    repo has no scheduler, so whatever one pass collects is what the account
    gets. That is why a listing failure partway through keeps the pages that
    already worked instead of raising, and why the outcome is stamped on the
    credential rather than only logged.
    """
    # Imported here, not at module top: this module is reached at API startup
    # through the adapter's registration, and cognee's package root is
    # heavyweight.
    from cognee.api.v1.remember.remember import remember as cognee_remember
    from cognee.modules.integrations.credentials import record_sync_result
    from cognee.modules.integrations.google_drive.adapter import access_token_for
    from cognee.modules.users.methods import get_user

    access_token = await access_token_for(credential)

    documents: list[str] = []
    skipped = 0
    failed = 0
    scanned = 0
    listing_failed = False
    truncated = False
    page_token: str | None = None

    for page_number in range(_MAX_PAGES):
        try:
            page = await client.list_files(access_token, page_token)
        except Exception:
            # Keep what the earlier pages produced. Raising here would throw
            # away every document already collected, and since nothing retries
            # this sync the account would be left permanently empty.
            listing_failed = True
            logger.exception(
                "Google Drive listing for account %s failed on page %d; "
                "keeping the %d documents collected so far",
                credential.provider_account_id,
                page_number + 1,
                len(documents),
            )
            break

        for file in page.get("files") or []:
            scanned += 1
            if len(documents) >= file_limit:
                truncated = True
                break
            try:
                content = await _read_file(access_token, file)
            except Exception:
                # One unreadable file (a permission quirk, an export Google
                # refuses) must not cost the whole sync.
                failed += 1
                logger.exception(
                    "Google Drive sync for account %s could not read a file",
                    credential.provider_account_id,
                )
                continue

            if not content or not content.strip():
                skipped += 1
                continue
            documents.append(format_file(file, content[:_MAX_CHARS_PER_FILE]))

        # Only an absent page token ends the walk. Drive filters by permission
        # after cutting a page, so a page of items this account cannot see
        # comes back empty with a live token, and treating that as the end
        # silently indexes a prefix of the Drive (or nothing at all).
        page_token = page.get("nextPageToken")
        if not page_token:
            break
        if len(documents) >= file_limit:
            truncated = True
            break
    else:
        truncated = True
        logger.info(
            "Google Drive listing for account %s stopped at the %d-page cap",
            credential.provider_account_id,
            _MAX_PAGES,
        )

    partial = listing_failed or truncated or failed > 0
    status = SYNC_STATUS_DEGRADED if partial else SYNC_STATUS_OK

    if not documents:
        logger.info(
            "Google Drive account %s synced no documents (%d scanned, %d unsupported, "
            "%d failed, listing_failed=%s)",
            credential.provider_account_id,
            scanned,
            skipped,
            failed,
            listing_failed,
        )
        await record_sync_result(
            GOOGLE_DRIVE_PROVIDER, credential.provider_account_id, status=status
        )
        return

    owner = await get_user(credential.user_id)
    dataset_name = _dataset_name(credential)

    logger.info(
        "Syncing %d Google Drive files for account %s into dataset %s "
        "(%d scanned, %d unsupported, %d failed, listing_failed=%s)",
        len(documents),
        credential.provider_account_id,
        dataset_name,
        scanned,
        skipped,
        failed,
        listing_failed,
    )
    result = await cognee_remember(
        documents,
        dataset_name=dataset_name,
        user=owner,
        # improve() is a whole-graph enrichment pass with LLM cost attached,
        # far too heavy to fire on every sync. Same stance as the Linear
        # adapter: enrichment stays a human or scheduled decision.
        self_improvement=False,
    )
    if getattr(result, "status", None) == "errored":
        status = SYNC_STATUS_DEGRADED
        logger.warning(
            "Google Drive sync for account %s finished with errors: %s",
            credential.provider_account_id,
            getattr(result, "error", None),
        )

    await record_sync_result(GOOGLE_DRIVE_PROVIDER, credential.provider_account_id, status=status)
