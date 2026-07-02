"""Google Drive authentication helpers.

All Google client libraries are imported lazily inside these functions so
``import cognee`` never requires them — install with
``pip install cognee[google-drive]`` to use this connector.
"""

import os

from cognee.shared.logging_utils import get_logger
from .config import GoogleDriveConfig
from .exceptions import GoogleDriveConfigError

logger = get_logger("google_drive_auth")

DRIVE_READONLY_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


def _import_google_libs():
    try:
        from googleapiclient.discovery import build
        from google.oauth2 import service_account
        from google.oauth2.credentials import Credentials as UserCredentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from google.auth.transport.requests import Request
    except ImportError as e:
        raise GoogleDriveConfigError(
            message=(
                "Google Drive connector requires the 'google-drive' extra. "
                "Install it with: pip install cognee[google-drive]"
            )
        ) from e
    return build, service_account, UserCredentials, InstalledAppFlow, Request


def _load_service_account_credentials(service_account, credentials_path: str):
    if not credentials_path or not os.path.exists(credentials_path):
        raise GoogleDriveConfigError(
            message=(
                f"Service account credentials file not found: {credentials_path!r}. "
                "Set GOOGLE_DRIVE_CREDENTIALS_PATH to a valid service account JSON key file."
            )
        )
    return service_account.Credentials.from_service_account_file(
        credentials_path, scopes=DRIVE_READONLY_SCOPES
    )


def _load_oauth_credentials(UserCredentials, InstalledAppFlow, Request, config: GoogleDriveConfig):
    token_path = config.google_drive_token_path
    client_secret_path = config.google_drive_credentials_path

    creds = None
    if token_path and os.path.exists(token_path):
        creds = UserCredentials.from_authorized_user_file(token_path, DRIVE_READONLY_SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not client_secret_path or not os.path.exists(client_secret_path):
                raise GoogleDriveConfigError(
                    message=(
                        f"OAuth client secret file not found: {client_secret_path!r}. "
                        "Set GOOGLE_DRIVE_CREDENTIALS_PATH to a valid OAuth client secret "
                        "JSON file downloaded from the Google Cloud Console."
                    )
                )
            flow = InstalledAppFlow.from_client_secrets_file(
                client_secret_path, DRIVE_READONLY_SCOPES
            )
            creds = flow.run_local_server(port=0)

        if token_path:
            with open(token_path, "w") as f:
                f.write(creds.to_json())

    return creds


def build_drive_service(config: GoogleDriveConfig):
    """Build an authenticated Drive v3 API client from connector config."""
    build, service_account, UserCredentials, InstalledAppFlow, Request = _import_google_libs()

    if config.google_drive_auth_mode == "service_account":
        credentials = _load_service_account_credentials(
            service_account, config.google_drive_credentials_path
        )
    elif config.google_drive_auth_mode == "oauth":
        credentials = _load_oauth_credentials(UserCredentials, InstalledAppFlow, Request, config)
    else:
        raise GoogleDriveConfigError(
            message=(
                f"Unsupported google_drive_auth_mode: {config.google_drive_auth_mode!r}. "
                "Must be 'service_account' or 'oauth'."
            )
        )

    return build("drive", "v3", credentials=credentials, cache_discovery=False)
