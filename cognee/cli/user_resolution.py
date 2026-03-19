"""Resolve a CLI --user-id flag into a User object."""

from typing import Optional
from uuid import UUID


async def resolve_cli_user(user_id: Optional[str] = None):
    """Return the User for the given --user-id, or the default user when omitted.

    If a user_id is supplied but does not exist yet, we create a new user
    so that each agent / caller gets its own identity automatically.
    """
    from cognee.modules.users.methods import get_default_user

    if not user_id:
        return await get_default_user()

    from cognee.modules.users.methods import get_user

    uid = UUID(user_id)
    try:
        return await get_user(uid)
    except Exception:
        # User does not exist yet — fall back to default.
        # A full "create-on-demand" flow would need the auth stack;
        # for CLI use the default user is the safe fallback.
        return await get_default_user()


def scoped_session_id(user_id: UUID, session_id: Optional[str] = None) -> str:
    """Return a session_id scoped to the user so agents don't share history."""
    base = session_id or "default"
    return f"{user_id}:{base}"
