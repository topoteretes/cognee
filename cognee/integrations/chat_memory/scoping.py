"""Reusable scope strategies, the only place platform choices about memory
boundaries live.

A scope strategy is just ``Callable[[Conversation], Scope]``. Each one answers
"what is shared, and what does 'forget me' wipe", expressed through the two keys
on :class:`Scope`. A bot picks a strategy (or writes its own one-liner) and
never touches cognee dataset/session semantics directly.

Three strategies cover almost everything:

* :func:`per_channel_scope`: collaborative team memory (Slack/Discord/Telegram
  channel bots). One connected graph per channel, so "what did we decide" can
  traverse across users.
* :func:`per_user_scope`: a personal cross-transport "second brain". One graph
  per user, spanning every platform, with a per-transport live session.
* :func:`per_workspace_scope`: memory shared across every channel in a
  workspace. The opt-in wide boundary.

All keys are produced through :func:`sanitize_key`, so the returned
:class:`Scope` is ready to use.
"""

from __future__ import annotations

from .models import Conversation, Scope
from .sanitizer import sanitize_key, sanitize_token


def _require(token: str, field: str) -> str:
    """Return ``token`` if it survives sanitization, else raise.

    The dataset key is the privacy / forget boundary, so a distinguishing
    segment that sanitizes to empty (which :func:`sanitize_key` would silently
    drop) must be rejected rather than aliasing distinct conversations into one
    dataset.
    """
    if not sanitize_token(token):
        raise ValueError(f"Conversation.{field} is required but sanitizes to empty: {token!r}")
    return token


def per_channel_scope(conversation: Conversation) -> Scope:
    """Per-channel memory: the default for team chat bots.

    ``dataset`` is the channel, so all members' messages land in one connected
    graph and a multi-hop question can walk a decision across who proposed it,
    who agreed, and what superseded it. ``session`` narrows to the live thread
    (falling back to the channel when the event is not threaded) for fast,
    recent-context recall.

    * ``dataset  = chat:{platform}:{workspace}:{channel}``
    * ``session  = {platform}:{workspace}:{channel}:{thread}``
    """
    dataset = sanitize_key(
        "chat",
        conversation.platform,
        conversation.workspace,
        _require(conversation.channel, "channel"),
    )
    session = sanitize_key(
        conversation.platform,
        conversation.workspace,
        conversation.channel,
        conversation.thread or "",
    )
    return Scope(dataset=dataset, session=session)


def per_user_scope(conversation: Conversation) -> Scope:
    """Per-user memory: a personal second brain spanning every transport.

    ``dataset`` is keyed by the user alone, so a note captured in Telegram is
    recallable from the web: durable recall targets the whole brain, not one
    conversation. ``session`` stays per-transport for recent context.
    This is the case that only works because ``dataset`` and ``session`` are
    independent fields.

    * ``dataset  = brain:{user}``
    * ``session  = {platform}:{workspace}:{channel}:{thread}``
    """
    dataset = sanitize_key("brain", _require(conversation.user, "user"))
    session = sanitize_key(
        conversation.platform,
        conversation.workspace,
        conversation.channel,
        conversation.thread or "",
    )
    return Scope(dataset=dataset, session=session)


def per_workspace_scope(conversation: Conversation) -> Scope:
    """Per-workspace memory: shared across every channel in a workspace.

    The opt-in wide boundary: memory and "forget this workspace" span all
    channels. ``session`` remains per-thread so recent context stays local.

    * ``dataset  = chat:{platform}:{workspace}``
    * ``session  = {platform}:{workspace}:{channel}:{thread}``
    """
    dataset = sanitize_key("chat", conversation.platform, conversation.workspace)
    session = sanitize_key(
        conversation.platform,
        conversation.workspace,
        conversation.channel,
        conversation.thread or "",
    )
    return Scope(dataset=dataset, session=session)
