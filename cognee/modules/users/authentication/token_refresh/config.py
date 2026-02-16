"""Configuration for access and refresh token lifetimes."""

import os


def get_access_token_lifetime_seconds() -> int:
    """Short-lived access token lifetime (default 30 minutes)."""
    return int(
        os.getenv("ACCESS_TOKEN_LIFETIME_SECONDS") or os.getenv("JWT_LIFETIME_SECONDS", "1800")
    )


def get_refresh_token_lifetime_seconds() -> int:
    """Long-lived refresh token lifetime (default 7 days)."""
    return int(os.getenv("REFRESH_TOKEN_LIFETIME_SECONDS", "604800"))
