"""Shared Google OAuth primitives used by Drive and Gmail integrations."""

from .client import (
    GoogleAuthError,
    exchange_code,
    fetch_userinfo,
    refresh_access_token,
    revoke_token,
)
from .google_settings import GoogleSettings, require

__all__ = [
    "GoogleAuthError",
    "GoogleSettings",
    "exchange_code",
    "fetch_userinfo",
    "refresh_access_token",
    "require",
    "revoke_token",
]
