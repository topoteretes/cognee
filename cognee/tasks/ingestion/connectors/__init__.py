"""SDK-bundled data sources for incremental, deletion-aware ingestion.

Install ``cognee[gmail]``, ``cognee[google-drive]`` or ``cognee[google-calendar]`` for the
Google dependencies and ``cognee[evernote]`` for Evernote's Thrift client; the Apollo, Asana,
Dropbox, Granola, Linear, Notion, Obsidian, OneNote, Slack, Telegram and Vercel sources need none.
Importing this package does not load DLT or client libraries, authenticate, or start ingestion.
Other connectors remain available in cognee-community.
"""

from .apollo import apollo_source, build_apollo_service
from .asana import asana_source, build_asana_service
from .dropbox import build_dropbox_service, dropbox_account_id, dropbox_source
from .evernote import build_evernote_service, evernote_source
from .gmail import build_gmail_service_from_access_token, gmail_source
from .google_calendar import build_calendar_service_from_access_token, google_calendar_source
from .google_drive import build_drive_service_from_access_token, google_drive_source
from .granola import build_granola_service, granola_source
from .linear import build_linear_service, linear_source
from .notion import notion_source
from .obsidian import obsidian_source
from .onenote import list_onenote_notebooks, onenote_source
from .slack import build_slack_service, slack_source
from .telegram import build_telegram_service, telegram_account_id, telegram_source
from .vercel import build_vercel_service, vercel_source

__all__ = [
    "apollo_source",
    "asana_source",
    "build_apollo_service",
    "build_asana_service",
    "build_calendar_service_from_access_token",
    "build_drive_service_from_access_token",
    "build_dropbox_service",
    "build_evernote_service",
    "build_gmail_service_from_access_token",
    "build_granola_service",
    "build_linear_service",
    "build_slack_service",
    "build_telegram_service",
    "build_vercel_service",
    "dropbox_account_id",
    "dropbox_source",
    "evernote_source",
    "gmail_source",
    "google_calendar_source",
    "google_drive_source",
    "granola_source",
    "linear_source",
    "list_onenote_notebooks",
    "notion_source",
    "obsidian_source",
    "onenote_source",
    "slack_source",
    "telegram_account_id",
    "telegram_source",
    "vercel_source",
]
