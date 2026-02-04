import types

from cognee.infrastructure.databases.cache.models import SessionQAEntry

from .session import add_feedback, get_session

session = types.SimpleNamespace(
    get_session=get_session,
    add_feedback=add_feedback,
)

__all__ = ["get_session", "add_feedback", "session", "SessionQAEntry"]
