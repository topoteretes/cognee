"""In-memory chat-memory adapter for deterministic, no-key tests (#3601).

Stores items in a plain ``{dataset -> [items]}`` dict and does substring
recall. It carries no cognee dependency and needs no API keys, so the bot's
identity, routing, and forget logic can be proven offline in under a second.

It satisfies the exact same ``ChatMemoryAdapter`` contract as the real cognee
adapter and resolves the dataset through the same ``dataset_for`` policy, so a
green test here exercises the real memory boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from .interface import (
    Answer,
    ChatMemoryAdapter,
    Citation,
    Conversation,
    Message,
    dataset_for,
    resolve_user,
)


@dataclass
class _StoredItem:
    text: str
    transport: str
    source: str
    canonical_user: str
    ts: str
    deeplink: str | None


class FakeChatMemoryAdapter(ChatMemoryAdapter):
    """A fake, in-memory implementation of the chat-memory contract."""

    def __init__(self) -> None:
        # dataset name -> list of stored items
        self._store: dict[str, list[_StoredItem]] = {}

    async def ingest(self, conversation: Conversation, message: Message) -> None:
        item = _StoredItem(
            text=message.text,
            transport=conversation.transport,
            source=conversation.source,
            canonical_user=resolve_user(conversation),
            ts=message.ts,
            deeplink=message.deeplink or conversation.msg_ref or "",
        )
        self._store.setdefault(dataset_for(conversation), []).append(item)

    async def answer(self, conversation: Conversation, query: str) -> Answer:
        items = self._store.get(dataset_for(conversation), [])
        matches = [item for item in items if _matches(query, item.text)]

        if not matches:
            return Answer(text="I do not have anything in your memory about that yet.")

        snippets = "; ".join(item.text for item in matches)
        citations = [
            Citation(
                content=item.text,
                source_transport=item.transport,
                source_ref=item.deeplink or f"{item.transport}:{item.source}",
                timestamp=item.ts,
            )
            for item in matches
        ]
        return Answer(text=f"From your memory: {snippets}", citations=citations)

    async def forget(self, target: Union[Conversation, str]) -> None:
        # target is a Conversation (scoped wipe) or a raw canonical user id
        dataset = dataset_for(target) if isinstance(target, Conversation) else f"brain:{target}"
        self._store.pop(dataset, None)


def _matches(query: str, text: str) -> bool:
    """Deterministic token-overlap match. Enough to prove routing, not recall quality.

    A query token matches a note token when either is a substring of the other,
    so a natural question ("where did I park?") still finds its note ("I parked
    the car..."). The real cognee adapter does semantic recall instead.
    """
    q = query.lower().strip()
    t = text.lower()
    if q and q in t:
        return True
    query_words = [w.strip("?.!,;:") for w in q.split()]
    query_words = [w for w in query_words if len(w) > 2]
    text_words = [w.strip("?.!,;:") for w in t.split()]
    return any(qw in tw or tw in qw for qw in query_words for tw in text_words if len(tw) > 2)
