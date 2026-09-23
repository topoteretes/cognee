"""Give the default user its password once -- and never change one it has (SDK-549).

The default user is created lazily by whichever surface touches the database
first, usually an SDK script or the CLI, and those create it with NO password
(``NO_PASSWORD_SENTINEL``): nothing to log in with, nothing to leak. A server
later started with ``DEFAULT_USER_PASSWORD`` -- `cognee-cli -ui`, docker
compose, an integration -- calls this once at startup to set that password.

This is first-time initialization, not reconciliation. A password the account
already has is never rewritten by an environment variable: an env var that can
silently replace a stored password is a takeover primitive. On a mismatch the
server warns and leaves the account alone.
"""

from pwdlib.exceptions import UnknownHashError

from cognee.base_config import get_base_config
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.get_user_db import get_user_db_context
from cognee.modules.users.get_user_manager import get_user_manager_context
from cognee.shared.logging_utils import get_logger

from .create_default_user import DEFAULT_USER_EMAIL, NO_PASSWORD_SENTINEL

logger = get_logger()


async def set_default_user_password_if_unset() -> bool | None:
    """Set ``DEFAULT_USER_PASSWORD`` on the default user only if it has none.

    Returns ``None`` when the variable is unset or the user does not exist yet,
    ``True`` when a password was set on a password-less account, and ``False``
    when the account already had one (matching or not -- it is never changed).
    """
    base_config = get_base_config()
    password = base_config.default_user_password
    if not password:
        return None
    email = base_config.default_user_email or DEFAULT_USER_EMAIL

    async with (
        get_relational_engine().get_async_session() as session,
        get_user_db_context(session) as user_db,
        get_user_manager_context(user_db) as user_manager,
    ):
        user = await user_db.get_by_email(email)
        if user is None:
            return None

        if user.hashed_password == NO_PASSWORD_SENTINEL:
            await user_db.update(
                user, {"hashed_password": user_manager.password_helper.hash(password)}
            )
            logger.info("Set the default user's password from DEFAULT_USER_PASSWORD.")
            return True

        try:
            matches, _ = user_manager.password_helper.verify_and_update(
                password, user.hashed_password
            )
        except UnknownHashError:
            matches = False

        if not matches:
            logger.warning(
                "DEFAULT_USER_PASSWORD does not match the default user's existing password. "
                "Leaving it unchanged: an environment variable never overrides a stored "
                "password. Change it through the API if that is intended."
            )
        return False
