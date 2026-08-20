"""Persist and load connection credentials from ~/.cognee/cloud_credentials.json.

The file stores one profile per service URL (schema version 2)::

    {
      "version": 2,
      "last_used": "http://localhost:8011",
      "profiles": {
        "http://localhost:8011": { ...CloudCredentials fields... },
        "https://tenant-abc.cloud.cognee.ai": { ... }
      }
    }

Keying by URL means connecting to a local instance never overwrites the
cloud profile (whose Auth0 refresh token would otherwise be lost, forcing
a browser re-auth on the next cloud connect) and vice versa. Files
written by older versions hold a single flat profile; they are read
transparently and upgraded to the profile schema on the next save.
"""

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from cognee.shared.logging_utils import get_logger

logger = get_logger("serve.credentials")

_CREDENTIALS_DIR = Path.home() / ".cognee"
_CREDENTIALS_FILE = _CREDENTIALS_DIR / "cloud_credentials.json"

_SCHEMA_VERSION = 2


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


def _credentials_from_dict(data: dict) -> CloudCredentials:
    return CloudCredentials(
        **{k: v for k, v in data.items() if k in CloudCredentials.__dataclass_fields__}
    )


def _is_cloud_profile(profile: dict) -> bool:
    # Cloud-mode profiles carry the Auth0 session; direct-mode profiles
    # store an empty access_token.
    return bool(profile.get("access_token") or profile.get("management_url"))


def _read_store() -> dict:
    """Read the store, upgrading a legacy single-profile file in memory."""
    path = get_credentials_path()
    if not path.exists():
        return {"version": _SCHEMA_VERSION, "last_used": "", "profiles": {}}
    try:
        data = json.loads(path.read_text())
    except Exception as e:
        logger.debug("Failed to read credentials store: %s", e)
        return {"version": _SCHEMA_VERSION, "last_used": "", "profiles": {}}

    if isinstance(data, dict) and isinstance(data.get("profiles"), dict):
        return {
            "version": _SCHEMA_VERSION,
            "last_used": data.get("last_used", ""),
            "profiles": data["profiles"],
        }

    # Legacy flat schema: the whole file is one profile.
    if isinstance(data, dict) and data.get("service_url"):
        service_url = data["service_url"]
        return {
            "version": _SCHEMA_VERSION,
            "last_used": service_url,
            "profiles": {service_url: data},
        }

    return {"version": _SCHEMA_VERSION, "last_used": "", "profiles": {}}


def _write_store(store: dict) -> None:
    """Write the store atomically (temp file + rename).

    The file now holds every connection's profile and may be touched by
    concurrent processes; a partial write must never destroy it.
    """
    path = get_credentials_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp{os.getpid()}")
    temp_path.write_text(json.dumps(store, indent=2))
    os.chmod(temp_path, 0o600)
    os.replace(temp_path, path)


def load_credentials(
    service_url: Optional[str] = None,
    *,
    cloud: bool = False,
) -> Optional[CloudCredentials]:
    """Load saved credentials.

    Args:
        service_url: Return the profile for this exact URL, or None if
            absent. Without it, the most recently used profile is returned.
        cloud: Only consider cloud-mode profiles (those holding an Auth0
            session). Used by ``serve()``'s cloud mode so a more recent
            local connection cannot shadow the cloud one.
    """
    store = _read_store()
    profiles = store["profiles"]

    if service_url is not None:
        profile = profiles.get(service_url.rstrip("/"))
        if profile and (not cloud or _is_cloud_profile(profile)):
            return _credentials_from_dict(profile)
        return None

    candidates = {
        url: profile for url, profile in profiles.items() if not cloud or _is_cloud_profile(profile)
    }
    if not candidates:
        return None
    last_used = store.get("last_used", "")
    if last_used in candidates:
        return _credentials_from_dict(candidates[last_used])
    # last_used points elsewhere (e.g. a local profile when cloud=True):
    # fall back to any matching profile.
    return _credentials_from_dict(next(iter(candidates.values())))


def save_credentials(creds: CloudCredentials) -> None:
    """Save the profile for ``creds.service_url``, leaving other profiles intact."""
    service_url = creds.service_url.rstrip("/")
    store = _read_store()
    store["profiles"][service_url] = asdict(creds)
    store["last_used"] = service_url
    _write_store(store)
    logger.debug("Saved credentials profile for %s", service_url)


def clear_credentials(service_url: Optional[str] = None) -> None:
    """Clear one profile, or the whole store when no URL is given."""
    path = get_credentials_path()
    if service_url is None:
        if path.exists():
            path.unlink()
            logger.debug("Cleared credentials store at %s", path)
        return

    store = _read_store()
    removed = store["profiles"].pop(service_url.rstrip("/"), None)
    if removed is None:
        return
    if store.get("last_used") == service_url.rstrip("/"):
        store["last_used"] = next(iter(store["profiles"]), "")
    _write_store(store)
    logger.debug("Cleared credentials profile for %s", service_url)


def is_token_expired(creds: CloudCredentials) -> bool:
    if not creds.expires_at:
        return True
    return time.time() > (creds.expires_at - 60)  # 60s buffer
