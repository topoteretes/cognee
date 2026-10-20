"""Split a text into passages a lexical index scores one by one.

A cached document is indexed as passages rather than as one unit: Postgres
caps a ``tsvector`` at 256 positions per lexeme, clamps positions above 16,383
and refuses vectors over 1 MB, so a whole document would undercount term
frequencies or fail outright, and passage-level BM25 ranks long documents
better anyway (the best passage is also the snippet a result shows). The
split is deterministic and chunker-free: paragraphs are packed up to the
token budget, an oversized paragraph is cut at sentence ends, and an
oversized sentence is cut by words.
"""

import re

DEFAULT_PASSAGE_TOKENS = 1000

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _token_count(text: str) -> int:
    return len(text.split())


def _cut_words(text: str, max_tokens: int) -> list[str]:
    words = text.split()
    return [" ".join(words[i : i + max_tokens]) for i in range(0, len(words), max_tokens)]


def _cut_paragraph(paragraph: str, max_tokens: int) -> list[str]:
    pieces: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for sentence in _SENTENCE_END.split(paragraph):
        tokens = _token_count(sentence)
        if tokens > max_tokens:
            if current:
                pieces.append(" ".join(current))
                current, current_tokens = [], 0
            pieces.extend(_cut_words(sentence, max_tokens))
            continue
        if current and current_tokens + tokens > max_tokens:
            pieces.append(" ".join(current))
            current, current_tokens = [], 0
        current.append(sentence)
        current_tokens += tokens
    if current:
        pieces.append(" ".join(current))
    return pieces


def split_passages(
    text: str, max_tokens: int = DEFAULT_PASSAGE_TOKENS, *, pack: bool = True
) -> list[str]:
    """Return the non-empty passages of ``text``, each at most ``max_tokens`` words.

    With ``pack`` (the default, for indexing) consecutive paragraphs are joined
    while they fit the budget; without it every paragraph is its own passage
    (for snippets, where one paragraph is what a reader wants to see).
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    passages: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for paragraph in _PARAGRAPH_BREAK.split(text or ""):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        tokens = _token_count(paragraph)
        if tokens > max_tokens:
            if current:
                passages.append("\n\n".join(current))
                current, current_tokens = [], 0
            passages.extend(_cut_paragraph(paragraph, max_tokens))
            continue
        if current and (not pack or current_tokens + tokens > max_tokens):
            passages.append("\n\n".join(current))
            current, current_tokens = [], 0
        current.append(paragraph)
        current_tokens += tokens
    if current:
        passages.append("\n\n".join(current))
    return passages
