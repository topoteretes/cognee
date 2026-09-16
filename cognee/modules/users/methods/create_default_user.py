from cognee.base_config import get_base_config

from .create_user import NO_PASSWORD_SENTINEL, create_user

DEFAULT_USER_EMAIL = "default_user@example.com"

__all__ = ["DEFAULT_USER_EMAIL", "NO_PASSWORD_SENTINEL", "create_default_user"]


async def create_default_user():
    """Create the default user, lazily, on first use.

    The account is created as a superuser (``is_superuser=True``). Every cognee
    surface falls back to it when no user is passed.

    ``DEFAULT_USER_PASSWORD`` unset means nobody asked for a loginable account,
    so the user is created with **no password** (``NO_PASSWORD_SENTINEL``):
    usable by every caller that resolves it through ``get_default_user()`` --
    SDK, CLI, and the API's own user resolution all read the row directly and
    never authenticate -- while ``POST /api/v1/auth/login`` has nothing to
    accept. No secret is generated, so there is none to lose.

    This replaces the literal ``default_password`` fallback, which gave every
    install a superuser whose credentials were public knowledge -- a
    known-credential admin login on any instance later served over the network
    with authentication enabled.

    A server started with ``DEFAULT_USER_PASSWORD`` gives a password-less
    default user that password once, at startup
    (``set_default_user_password_if_unset``). It never changes a password the
    account already has. This function stays silent: it also runs under plain
    SDK and CLI use, where an account nobody is going to log into is not worth
    a log line.
    """
    base_config = get_base_config()
    default_user_email = base_config.default_user_email or DEFAULT_USER_EMAIL

    user = await create_user(
        email=default_user_email,
        password=base_config.default_user_password or None,
        is_superuser=True,
        is_active=True,
        is_verified=True,
        auto_login=True,
    )

    return user
