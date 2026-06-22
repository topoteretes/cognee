"""Persist and load cloud credentials from ~/.cognee/cloud_credentials.json."""

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from cognee.shared.logging_utils import get_logger

logger = get_logger("serve.credentials")

_CREDENTIALS_DIR = Path.home() / ".cognee"
_CREDENTIALS_FILE = _CREDENTIALS_DIR / "cloud_credentials.json"


@dataclass
class CloudCredentials:
    access_token: str
    refresh_token: Optional[str] = None
    expires_at: float = 0.0  # Unix timestamp
    service_url: str = ""
    api_key: str = ""
    management_url: str = ""
    tenant_id: str = ""
    tenant_name: str = ""
    email: str = ""


def get_credentials_path() -> Path:
    return _CREDENTIALS_FILE


def load_credentials() -> Optional[CloudCredentials]:
    path = get_credentials_path()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return CloudCredentials(
            **{k: v for k, v in data.items() if k in CloudCredentials.__dataclass_fields__}
        )
    except Exception as e:
        logger.debug("Failed to load cloud credentials: %s", e)
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


def is_token_expired(creds: CloudCredentials) -> bool:
    if not creds.expires_at:
        return True
    return time.time() > (creds.expires_at - 60)  # 60s buffer
