import types

from cognee.infrastructure.databases.cache.models import SessionQAEntry
from cognee.modules.session_distillation import DistillationResult, distill_session

from .session import add_feedback, add_frequency_weights, delete_feedback, get_session

session = types.SimpleNamespace(
    get_session=get_session,
    add_feedback=add_feedback,
    add_frequency_weights=add_frequency_weights,
    delete_feedback=delete_feedback,
    distill_session=distill_session,
)

__all__ = [
    "get_session",
    "add_feedback",
    "add_frequency_weights",
    "delete_feedback",
    "distill_session",
    "DistillationResult",
    "session",
    "SessionQAEntry",
]
