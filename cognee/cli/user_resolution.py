"""Resolve a CLI --user-id flag into a User object."""

from typing import Optional
from uuid import UUID

import cognee.cli.echo as fmt


async def resolve_cli_user(user_id: Optional[str] = None):
    """Return the User for the given --user-id, or the default user when omitted.

    Raises ValueError with a clear message if user_id is not a valid UUID.
    Warns and falls back to the default user if the UUID is valid but unknown.
    """
    from cognee.modules.users.methods import get_default_user

    if not user_id:
        return await get_default_user()

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
        fmt.warning(
            f"User {uid} not found — falling back to the default user.  "
            f"The --user-id will not provide isolation."
        )
        return await get_default_user()


def scoped_session_id(user_id: UUID, session_id: Optional[str] = None) -> str:
    """Return a session_id scoped to the user so agents don't share history."""
    base = session_id or "default"
    return f"{user_id}:{base}"
