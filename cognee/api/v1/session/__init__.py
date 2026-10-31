import types

from cognee.infrastructure.databases.cache.models import SessionQAEntry
from cognee.modules.session_distillation import DistillationResult, distill_session

from .session import add_feedback, delete_feedback, delete_session, get_session

session = types.SimpleNamespace(
    get_session=get_session,
    add_feedback=add_feedback,
    delete_feedback=delete_feedback,
    delete_session=delete_session,
    distill_session=distill_session,
)

__all__ = [
    "DistillationResult",
    "SessionQAEntry",
    "add_feedback",
    "delete_feedback",
    "delete_session",
    "distill_session",
    "get_session",
    "session",
]
