"""The chat-memory adapter: three primitives every bot shares.

A bot supplies two things: a scope strategy (what memory boundary this bot
uses) and a translation of platform events into :class:`Conversation` and
:class:`Message`. Everything else (consent gating, the ``external_metadata``
stamp, citation assembly, the forget paths) lives here and is identical across
Slack, Telegram, Discord, and a personal second brain.

    adapter = ChatMemoryAdapter(scope=per_channel_scope)
    await adapter.ingest(conversation, message)      # background remember
    answer = await adapter.answer(conversation, q)   # recall + citations
    await adapter.forget(user="U123")                # privacy: forget me
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from cognee.shared.logging_utils import get_logger

from .backend import CogneeMemoryBackend, MemoryBackend, deterministic_item_id
from .consent import ConsentStore, InMemoryConsentStore
from .models import Answer, Conversation, Message, Scope
from .scoping import per_channel_scope

logger = get_logger("chat_memory.adapter")

ScopeStrategy = Callable[[Conversation], Scope]


def _is_direct(conversation: Conversation) -> bool:
    """Heuristic for a 1:1 / direct conversation.

    A DM on most platforms has no workspace above it and a channel that is the
    peer's id, so ``channel == user`` with an empty workspace is a good default
    signal. A bot that knows better can set consent explicitly instead of
    relying on this.
    """
    return not conversation.workspace and conversation.channel == conversation.user


class ChatMemoryAdapter:
    """Framework-agnostic chat memory over cognee. Keeps every bot ~100 lines.

    Args:
        scope: Strategy mapping a :class:`Conversation` to a :class:`Scope`
            (its two keys). Defaults to :func:`per_channel_scope`. Pick one from
            :mod:`.scoping` or pass any ``Conversation -> Scope`` callable.
        backend: The memory backend. Defaults to :class:`CogneeMemoryBackend`
            (in-process cognee SDK). Inject a fake in tests.
        consent_store: Where per-user opt-in/opt-out lives. Defaults to
            :class:`InMemoryConsentStore`.
        group_default_consent: Consent default for group/channel conversations
            when a user has made no explicit choice. ``False`` (default) means a
            channel bot stays silent for a user until they opt in.
        direct_default_consent: Consent default for 1:1 / direct conversations.
            ``True`` (default), since using the bot in a DM is itself the opt-in.
        top_k: Default recall breadth.
    """

    def __init__(
        self,
        *,
        scope: ScopeStrategy = per_channel_scope,
        backend: Optional[MemoryBackend] = None,
        consent_store: Optional[ConsentStore] = None,
        group_default_consent: bool = False,
        direct_default_consent: bool = True,
        top_k: int = 10,
    ) -> None:
        self._scope = scope
        self.backend: MemoryBackend = backend or CogneeMemoryBackend(top_k=top_k)
        self.consent: ConsentStore = consent_store or InMemoryConsentStore()
        self.group_default_consent = group_default_consent
        self.direct_default_consent = direct_default_consent
        self.top_k = top_k

    # -- scoping -----------------------------------------------------------
    def scope(self, conversation: Conversation) -> Scope:
        """Resolve a conversation to its two keys. Override or inject to customize."""
        return self._scope(conversation)

    # -- consent -----------------------------------------------------------
    def set_consent(self, user: str, on: bool) -> None:
        """Record a user's explicit opt-in (``on=True``) or opt-out."""
        self.consent.set(user, on)

    def has_consent(self, conversation: Conversation, user: str) -> bool:
        """Whether ``user`` may be remembered in this conversation.

        An explicit choice always wins; otherwise the context default applies
        (deny in groups, allow in DMs). "Direct" is inferred from an empty
        workspace and a channel that equals the user id, the shape a 1:1 DM
        takes on most platforms. A bot can always set consent explicitly
        instead of relying on the inference.
        """
        if self.consent.is_set(user):
            return self.consent.get(user)
        return (
            self.direct_default_consent if _is_direct(conversation) else self.group_default_consent
        )

    # -- primitives --------------------------------------------------------
    async def ingest(self, conversation: Conversation, message: Message) -> bool:
        """Remember a message (background), gated on consent. Returns whether it stored.

        Fire-and-forget by design: the backend returns immediately. The stored
        item is stamped with ``external_metadata`` (platform, workspace,
        channel, user, ts, permalink), the single stamp that later powers both
        "forget me" and citations, and given a deterministic id so a replayed
        message dedups instead of duplicating.
        """
        author = message.user or conversation.user
        if not self.has_consent(conversation, author):
            logger.debug("chat_memory: skipping ingest for user=%s (no consent)", author)
            return False

        if not message.text or not message.text.strip():
            return False

        scope = self.scope(conversation)
        external_metadata = self._stamp(conversation, message, author)
        item_id = deterministic_item_id(
            scope.dataset,
            conversation.channel,
            author,
            message.timestamp or message.text,
        )
        await self.backend.remember(
            message.text,
            dataset=scope.dataset,
            session=scope.session,
            external_metadata=external_metadata,
            item_id=item_id,
        )
        return True

    async def answer(
        self, conversation: Conversation, query: str, *, top_k: Optional[int] = None
    ) -> Answer:
        """Recall against this conversation's scope and return text + citations.

        Recall reads the fast session cache first and falls through to the
        dataset graph. The answer body is the best synthesized/graph result when
        present, otherwise the top session snippet; every recalled item is
        returned as a :class:`Citation` resolved back to its source message.
        """
        scope = self.scope(conversation)
        citations = await self.backend.recall(
            query,
            dataset=scope.dataset,
            session=scope.session,
            top_k=top_k or self.top_k,
        )
        return Answer(text=self._compose_answer_text(citations), citations=citations)

    async def forget(
        self,
        *,
        conversation: Optional[Conversation] = None,
        user: Optional[str] = None,
    ) -> dict:
        """Forget memory. The privacy path.

        Two modes:

        * ``forget(conversation=...)`` wipes the conversation's whole dataset
          (everything shared in that scope).
        * ``forget(conversation=..., user=...)`` forgets just that user's items
          inside the scope (the per-user "forget me"). It also clears any stored
          consent so a re-opt-in starts clean.

        Passing ``user`` without ``conversation`` is rejected: which dataset the
        user lives in is scope-dependent and cannot be guessed.
        """
        if user is not None and conversation is None:
            raise ValueError(
                "forget(user=...) needs a conversation to resolve the user's dataset; "
                "pass conversation=... too."
            )
        if conversation is None:
            raise ValueError("forget() requires conversation (and optionally user).")

        scope = self.scope(conversation)
        if user is not None:
            result = await self.backend.forget_user(dataset=scope.dataset, user=user)
            # A forget-me revokes prior consent; the user opts in again to resume.
            self.consent.set(user, False)
            return result
        return await self.backend.forget_scope(dataset=scope.dataset)

    # -- internals ---------------------------------------------------------
    @staticmethod
    def _stamp(conversation: Conversation, message: Message, author: str) -> dict[str, Any]:
        """Build the ``external_metadata`` stamp carried by a stored item."""
        # Free-form message extras first, then the adapter-owned identity keys,
        # so a stray ``metadata`` entry can never overwrite the resolved author
        # (which "forget me" and consent attribution depend on).
        stamp: dict[str, Any] = dict(message.metadata)
        stamp.update(
            {
                "platform": conversation.platform,
                "workspace": conversation.workspace,
                "channel": conversation.channel,
                "thread": conversation.thread,
                "user": author,
                "ts": message.timestamp,
                "permalink": message.permalink,
            }
        )
        # Drop keys with no value so the stamp stays compact and comparable.
        return {key: value for key, value in stamp.items() if value is not None}

    @staticmethod
    def _compose_answer_text(items) -> str:
        """Pick the answer body: the first non-empty graph result, else any snippet."""
        for item in items:
            if item.source in ("graph", "graph_context") and item.text.strip():
                return item.text
        return next((item.text for item in items if item.text.strip()), "")
