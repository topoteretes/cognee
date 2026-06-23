"""Resolve a CLI --user-id flag into a User object."""

from typing import Optional
from uuid import UUID

import cognee.cli.echo as fmt


async def resolve_cli_user(user_id: Optional[str] = None, strict: bool = False):
    """Return the User for the given --user-id, or the default user when omitted.

    Raises ValueError with a clear message if user_id is not a valid UUID.

    When ``strict`` is False (default), a valid-but-unknown UUID warns and falls
    back to the default user. When ``strict`` is True, an unknown UUID is a hard
    error instead — used by ownership-sensitive commands (e.g. ``agents``) where a
    silent fallback to the default user would break the isolation the flag promises.
    """
    from cognee.modules.users.methods import get_default_user

    if not user_id:
        return await _get_default_user_with_recovery()

    try:
        uid = UUID(user_id)
    except ValueError:
        raise ValueError(
            f"Invalid --user-id: '{user_id}' is not a valid UUID.  "
            f"Example: --user-id 550e8400-e29b-41d4-a716-446655440000"
        )

    from cognee.modules.users.methods import get_user

    try:
        return await get_user(uid)
    except Exception:
        if strict:
            raise ValueError(
                f"--user-id {uid} does not exist.  Refusing to fall back to the default "
                f"user because that would silently break isolation.  Create the user first "
                f"or omit --user-id to act as the default user."
            )
        fmt.warning(
            f"User {uid} not found — falling back to the default user.  "
            f"The --user-id will not provide isolation."
        )
        return await _get_default_user_with_recovery()


async def _get_default_user_with_recovery():
    """Try get_default_user(); on DatabaseNotCreatedError run migrations and retry."""
    from cognee.modules.users.methods import get_default_user
    from cognee.infrastructure.databases.exceptions import DatabaseNotCreatedError

    try:
        return await get_default_user()
    except DatabaseNotCreatedError:
        from cognee.infrastructure.databases.relational import get_relational_engine
        from cognee.modules.migrations.startup import run_migrations

        await run_migrations()

        return await get_default_user()


def scoped_session_id(user_id: UUID, session_id: Optional[str] = None) -> str:
    """Return a session_id scoped to the user so agents don't share history."""
    base = session_id or "default"
    return f"{user_id}:{base}"
