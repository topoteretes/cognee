import os
import secrets

from .create_user import create_user
from cognee.base_config import get_base_config
from cognee.shared.logging_utils import get_logger

logger = get_logger("create_default_user")


def _is_production() -> bool:
    # Mirrors cognee.api.client: ENV defaults to "prod".
    return os.getenv("ENV", "prod").lower() == "prod"


def _resolve_default_password(configured: str | None) -> str:
    """Return the password for the auto-created default superuser.

    Previously this fell back to the static, well-known literal
    "default_password" for an ``is_superuser=True`` account — trivial admin
    access on any deployment that didn't override it. Now:
      * a configured password is used as-is;
      * in production an unset password is a hard error (a superuser must have an
        operator-set credential);
      * otherwise a random password is generated so no known credential exists.
    """
    if configured:
        return configured

    if _is_production():
        raise RuntimeError(
            "DEFAULT_USER_PASSWORD is not set. The default superuser must have an "
            "explicit password in production. Set DEFAULT_USER_PASSWORD (and "
            "DEFAULT_USER_EMAIL) to strong values before starting."
        )

    generated = secrets.token_urlsafe(32)
    logger.warning(
        "DEFAULT_USER_PASSWORD is not set — generated a random password for the "
        "default user. Set DEFAULT_USER_PASSWORD if you need to log in as this user."
    )
    return generated


async def create_default_user():
    base_config = get_base_config()
    default_user_email = base_config.default_user_email or "default_user@example.com"
    default_user_password = _resolve_default_password(base_config.default_user_password)

    user = await create_user(
        email=default_user_email,
        password=default_user_password,
        is_superuser=True,
        is_active=True,
        is_verified=True,
        auto_login=True,
    )

    return user
