"""Bridge cognee's source references back to tappable Telegram links.

``recall(..., include_references=True)`` grounds an answer in source chunks,
but those references point at cognee documents, not Telegram messages. So the
bot keeps its own ledger: every ingested message is recorded with its
``chat_id`` / ``message_id`` / ``thread_id``, and after a recall we match the
answer's evidence back to the originating message to render a deep link.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_WORD = re.compile(r"[a-z0-9]+")
_MIN_TERM_LEN = 3

# Common words that shouldn't, on their own, link an answer back to a message.
_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "are",
        "was",
        "with",
        "you",
        "this",
        "that",
        "but",
        "not",
        "from",
        "what",
        "when",
        "who",
        "why",
        "how",
        "where",
        "your",
        "our",
        "their",
        "has",
        "have",
        "had",
        "will",
        "would",
        "can",
        "could",
        "should",
        "did",
        "does",
        "about",
        "into",
        "out",
        "they",
        "them",
        "its",
        "his",
        "her",
        "she",
        "him",
        "any",
        "all",
        "some",
    }
)


def _terms(text: str) -> set[str]:
    return {
        t for t in _WORD.findall(text.lower()) if len(t) >= _MIN_TERM_LEN and t not in _STOPWORDS
    }


@dataclass(frozen=True)
class MessageRef:
    """A Telegram message the bot ingested, kept for citation resolution."""

    chat_id: int
    message_id: int
    text: str
    thread_id: int | None = None
    author: str | None = None

    def attributed_text(self) -> str:
        """The text as stored in memory, prefixed with author attribution."""
        return f"{self.author}: {self.text}" if self.author else self.text

    def deep_link(self) -> str | None:
        """A ``t.me/c/...`` link, or None when Telegram exposes no public link.

        Only supergroups/channels (chat ids prefixed ``-100``) have public
        message deep links. DMs and basic groups return None — the caller
        falls back to quoting the source snippet.
        """
        raw = str(self.chat_id)
        if not raw.startswith("-100"):
            return None
        internal = raw[4:]
        if self.thread_id is not None:
            return f"https://t.me/c/{internal}/{self.thread_id}/{self.message_id}"
        return f"https://t.me/c/{internal}/{self.message_id}"


@dataclass
class CitationLedger:
    """Per-dataset record of ingested messages, used to resolve citations."""

    _by_dataset: dict[str, list[MessageRef]] = field(default_factory=dict)

    def record(self, dataset_name: str, ref: MessageRef) -> None:
        self._by_dataset.setdefault(dataset_name, []).append(ref)

    def drop(self, dataset_name: str) -> None:
        """Forget a dataset's ledger (called on ``/forget``)."""
        self._by_dataset.pop(dataset_name, None)

    def refs(self, dataset_name: str) -> list[MessageRef]:
        return list(self._by_dataset.get(dataset_name, []))

    def resolve(self, dataset_name: str, evidence: str, *, limit: int = 3) -> list[MessageRef]:
        """Match recall evidence text back to ingested messages.

        Ranks this dataset's ledger entries by term overlap with the answer /
        evidence text and returns the strongest matches. Returns ``[]`` when
        nothing overlaps (so the bot abstains from citing rather than guess).
        """
        wanted = _terms(evidence)
        if not wanted:
            return []
        scored: list[tuple[int, MessageRef]] = []
        for ref in self._by_dataset.get(dataset_name, []):
            overlap = len(_terms(ref.text) & wanted)
            if overlap:
                scored.append((overlap, ref))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [ref for _, ref in scored[:limit]]
