"""Per-user scope policy for the second-brain bot.

This is the memory boundary decision for #3613: the dataset is keyed by the
canonical user, not by the channel, so a note captured on one transport is
recallable from any other. The session stays per-transport for fast recent
context.

    dataset = brain:{canonical_user}
    session = {transport}:{source}

Both the fake adapter (tests) and the real cognee adapter resolve scope
through this one function, so the tested scope logic is the shipped scope
logic. When the #3608 adapter merges with a configurable ScopePolicy, this
maps to its per-user option.
"""

from __future__ import annotations

from .interface import Conversation, Scope


def resolve_user(conversation: Conversation) -> str:
    """The canonical user id, falling back to the raw external identity.

    The bot's identity layer normally fills ``canonical_user`` before the
    conversation reaches the adapter. The fallback keeps the adapter usable
    on its own (each unlinked external identity gets its own brain).
    """
    if conversation.canonical_user:
        return conversation.canonical_user
    return f"{conversation.transport}:{conversation.external_user or conversation.source}"


def per_user_scope(conversation: Conversation) -> Scope:
    user = resolve_user(conversation)
    return Scope(
        dataset=f"brain:{user}",
        session=f"{conversation.transport}:{conversation.source}",
    )
