"""Persist and load the saved connection from ~/.cognee/cloud_credentials.json."""

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from cognee.shared.logging_utils import get_logger

logger = get_logger("serve.credentials")

_CREDENTIALS_DIR = Path.home() / ".cognee"
_CREDENTIALS_FILE = _CREDENTIALS_DIR / "cloud_credentials.json"


@dataclass
class CloudCredentials:
    """The instance ``serve()`` connected to and the API key it accepted."""

    service_url: str = ""
    api_key: str = ""
    # Tenant the instance's /api/v1/users/me reported at connect time (on
    # Cognee Cloud, the control-plane tenants.id). Read by telemetry.
    tenant_id: str = ""


def get_credentials_path() -> Path:
    return _CREDENTIALS_FILE


def load_credentials() -> CloudCredentials | None:
    path = get_credentials_path()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        # Keys the file may still carry from older releases (Auth0 tokens,
        # tenant fields) are dropped here, not rejected.
        return CloudCredentials(
            **{k: v for k, v in data.items() if k in CloudCredentials.__dataclass_fields__}
        )
    except Exception as e:
        logger.debug("Failed to load cloud credentials: %s", e, exc_info=True)
        return None


def save_credentials(creds: CloudCredentials) -> None:
    path = get_credentials_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(creds), indent=2))
    os.chmod(path, 0o600)
    logger.debug("Saved cloud credentials to %s", path)


def clear_credentials() -> None:
    path = get_credentials_path()
    if path.exists():
        path.unlink()
        logger.debug("Cleared cloud credentials at %s", path)
