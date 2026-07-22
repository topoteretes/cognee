"""Chat-memory adapter: the #3608 contract, a fake impl, and the real cognee impl."""

from .interface import (
    Answer,
    ChatMemoryAdapter,
    Citation,
    Conversation,
    Message,
    Scope,
)
from .scope_policy import per_user_scope, resolve_user

__all__ = [
    "Answer",
    "ChatMemoryAdapter",
    "Citation",
    "Conversation",
    "Message",
    "Scope",
    "per_user_scope",
    "resolve_user",
]
