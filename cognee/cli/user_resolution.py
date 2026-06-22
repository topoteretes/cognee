"""Resolve a CLI --user-id flag into a User object."""

from typing import Optional
from uuid import UUID


async def resolve_cli_user(user_id: Optional[str] = None):
    """Return the User for the given --user-id, or the default user when omitted.

    Raises ValueError with a clear message if user_id is not a valid UUID, or if
    the given --user-id does not exist.
    """
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
        raise ValueError(
            f"--user-id {uid} does not exist. Create the user first "
            f"or omit --user-id to act as the default user."
        )


async def _get_default_user_with_recovery():
    """Try get_default_user(); on DatabaseNotCreatedError run migrations and retry."""
    from cognee.modules.users.methods import get_default_user
    from cognee.infrastructure.databases.exceptions import DatabaseNotCreatedError

    try:
        return await get_default_user()
    except DatabaseNotCreatedError:
        from cognee.infrastructure.databases.relational import get_relational_engine
        from cognee.modules.migrations.startup import run_migrations

        try:
            await run_migrations()
        except Exception:
            db_engine = get_relational_engine()
            await db_engine.create_database()
            await run_migrations()

        return await get_default_user()


def scoped_session_id(user_id: UUID, session_id: Optional[str] = None) -> str:
    """Return a session_id scoped to the user so agents don't share history."""
    base = session_id or "default"
    return f"{user_id}:{base}"
