"""Posture the API server reports once, at startup.

Separate from ``cognee/api/client.py`` so these checks can be exercised without
importing the FastAPI app and every router behind it.
"""

from cognee.base_config import get_base_config
from cognee.shared.logging_utils import get_logger

logger = get_logger()


def report_default_user_login_posture() -> None:
    """Say whether the default account can be logged into. Server startup only.

    The default user is created lazily by every surface, so this lives here
    rather than in ``create_default_user``: only a running server exposes
    ``POST /api/v1/auth/login`` and the UI, so only here is the login posture
    something an operator needs to act on. Plain SDK and CLI use never
    authenticates as this account and so never sees this line.

    Quiet when a password is configured -- that is the working case.
    """
    if get_base_config().default_user_password:
        return

    logger.warning(
        "DEFAULT_USER_PASSWORD is unset: no default-user password login is configured. "
        "A default user created by the SDK or CLI has no password and cannot be logged "
        "into; set DEFAULT_USER_PASSWORD and restart to give it one (needed for the UI "
        "and any HTTP client). An account that already has a password keeps it."
    )
