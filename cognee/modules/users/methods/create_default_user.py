import secrets

from cognee.base_config import get_base_config

from .create_user import create_user

DEFAULT_USER_EMAIL = "default_user@example.com"


async def create_default_user():
    """Create the default user, lazily, on first use.

    **This account is a superuser** (``is_superuser=True``): it bypasses
    dataset ACLs and can reach every tenant's data. Every cognee surface falls
    back to it when no user is passed, so on a single-user install it is the
    identity all work runs as.

    ``DEFAULT_USER_PASSWORD`` unset means nobody asked for a loginable account,
    so one is not created: the password is random and never recorded, which
    leaves the account usable by every caller that resolves it through
    ``get_default_user()`` (SDK, CLI, and the API's own user resolution all
    read the row directly and never authenticate) while
    ``POST /api/v1/auth/login`` has no credential to accept.

    This replaces the literal ``default_password`` fallback, which gave every
    install a superuser whose credentials were public knowledge -- a
    known-credential admin login on any instance later served over the network
    with authentication enabled.

    Set ``DEFAULT_USER_PASSWORD`` to log in as this user; it is used verbatim,
    exactly as before. The API server reports the resulting login posture at
    startup (``cognee/api/client.py`` lifespan) -- this function stays silent,
    because it also runs under plain SDK and CLI use where an account nobody
    is going to log into is not worth a log line.
    """
    base_config = get_base_config()
    default_user_email = base_config.default_user_email or DEFAULT_USER_EMAIL
    default_user_password = base_config.default_user_password

    if not default_user_password:
        # Never logged and never stored in plaintext: the point is that no
        # known value opens this account. Recovery is setting the env var, not
        # retrieving this string.
        default_user_password = secrets.token_urlsafe(32)

    user = await create_user(
        email=default_user_email,
        password=default_user_password,
        is_superuser=True,
        is_active=True,
        is_verified=True,
        auto_login=True,
    )

    return user
