"""Google Drive connector for cognee.

Exports the contract object (``CONNECTOR``), the DLT source factory
(``google_drive_source``) and the client builders cognee's connector package exported.
This module imports no host: cognee is needed only by
:mod:`cognee_connector_google_drive.cognee_host`, which is installed with the ``cognee``
extra and which nothing here imports.
"""

from cognee_connector_google_drive.connector import CONNECTOR, DESCRIPTOR, GoogleDriveConnector
from cognee_connector_google_drive.source import (
    DriveAuthError,
    build_drive_service,
    build_drive_service_from_access_token,
    google_drive_source,
)

__all__ = [
    "CONNECTOR",
    "DESCRIPTOR",
    "DriveAuthError",
    "GoogleDriveConnector",
    "build_drive_service",
    "build_drive_service_from_access_token",
    "google_drive_source",
]
