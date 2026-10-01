"""SDK-bundled Google data sources for incremental, deletion-aware ingestion.

Install ``cognee[gmail]`` or ``cognee[google-drive]`` for optional dependencies.
Importing this package does not load DLT or Google client libraries, authenticate,
or start ingestion. Other connectors remain available in cognee-community.
"""

from .gmail import build_gmail_service_from_access_token, gmail_source
from .google_drive import build_drive_service_from_access_token, google_drive_source

__all__ = [
    "build_drive_service_from_access_token",
    "build_gmail_service_from_access_token",
    "gmail_source",
    "google_drive_source",
]
