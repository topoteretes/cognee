"""SDK-bundled data sources for incremental, deletion-aware ingestion.

Install ``cognee[gmail]``, ``cognee[google-drive]`` or ``cognee[google-calendar]`` for the
Google dependencies; the Linear and Notion sources need none. Importing this package does
not load DLT or client libraries, authenticate, or start ingestion. Other connectors remain
available in cognee-community.
"""

from .gmail import build_gmail_service_from_access_token, gmail_source
from .google_calendar import build_calendar_service_from_access_token, google_calendar_source
from .google_drive import build_drive_service_from_access_token, google_drive_source
from .linear import build_linear_service, linear_source
from .notion import notion_source

__all__ = [
    "build_calendar_service_from_access_token",
    "build_drive_service_from_access_token",
    "build_gmail_service_from_access_token",
    "build_linear_service",
    "gmail_source",
    "google_calendar_source",
    "google_drive_source",
    "linear_source",
    "notion_source",
]
