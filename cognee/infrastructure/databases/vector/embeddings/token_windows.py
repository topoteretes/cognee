"""Embed a text longer than the model's input limit as consecutive token windows.

Every engine embeds an over-length text the same way: the text is cut once into
consecutive windows of at most the engine's input limit (the one chunks are cut to,
``resolve_input_limit``), every window of the batch is sent in the engine's normal
request, and each text's vector is the mean of its windows' vectors weighted by
their token counts. The cost is linear in the text's length, every token weighs the
same, and a text that fits is its own single window, so its request is unchanged.

Two cutters, because the engines know their models to different degrees:

- :func:`windows_from_encoding` cuts at the token offsets of the model's own
  ``tokenizers`` encoding. fastembed has that encoding and truncates silently
  instead of rejecting a long input, so its cut must be exact.
- :func:`windows_by_count` needs only ``count_tokens``, the one method every cognee
  tokenizer has. The other engines' tokenizers can be approximate (TikToken stands
  in for a model whose tokenizer cannot be loaded), so a window can still be
  rejected now and then; the provider's rejection is then recovered by the
  engine's own over-length handling, as before, on a text that is already close
  to the limit.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Sequence

import numpy as np

from cognee.infrastructure.databases.vector.embeddings.EmbeddingEngine import (
    EmbeddingResult,
)
from cognee.infrastructure.databases.vector.embeddings.input_limit import resolve_input_limit
from cognee.shared.logging_utils import get_logger

logger = get_logger("token_windows")

# A window: its text and the tokens it holds.
Window = tuple[str, int]

# A word with the white space that follows it; white space that opens the text
# is a piece of its own. The pieces join back to the text exactly.
_PIECE = re.compile(r"\S+\s*|\s+")


def windows_by_count(text: str, count_tokens: Callable[[str], int], limit: int) -> list[Window]:
    """Cut ``text`` into consecutive windows that ``count_tokens`` counts at most
    ``limit`` tokens each.

    Words are packed whole, by their own counts, and each window is counted again
    as a whole and given back words until it fits, so a window never goes over
    the limit by the tokenizer's count. A word longer than the limit is cut inside,
    at the longest prefix that fits. The windows join back to ``text``.
    """
    total = count_tokens(text)
    if total <= limit:
        return [(text, total)]

    # Each word with its own count; a word longer than the limit is cut first.
    pieces: list[Window] = []
    for piece in _PIECE.findall(text):
        size = count_tokens(piece)
        pieces.extend(
            [(piece, size)] if size <= limit else _cut_long_piece(piece, count_tokens, limit)
        )

    windows: list[Window] = []
    start = 0
    while start < len(pieces):
        end, estimate = start, 0
        while end < len(pieces) and (end == start or estimate + pieces[end][1] <= limit):
            estimate += pieces[end][1]
            end += 1
        window = "".join(piece for piece, _ in pieces[start:end])
        tokens = count_tokens(window)
        # Counted alone, words can add up to less than together; give words back.
        while tokens > limit and end - start > 1:
            end -= 1
            window = "".join(piece for piece, _ in pieces[start:end])
            tokens = count_tokens(window)
        windows.append((window, tokens))
        start = end
    return windows


def _fits_unseen(text: str, limit: int) -> bool:
    """Whether ``text`` fits without being counted. Every cognee tokenizer gives a
    text at most one token per byte plus a few: SentencePiece adds a leading "▁"
    token (SFR-Mistral counts "11111111" as 9), and the Mistral tokenizer counts a
    chat template around the text. Half the limit in bytes leaves that room for any
    real limit. Most texts embedded (entity names, relation lines, queries) are far
    shorter than that."""
    return 2 * len(text.encode("utf-8")) <= limit


def _cut_long_piece(piece: str, count_tokens: Callable[[str], int], limit: int) -> list[Window]:
    """``piece``, longer than ``limit`` tokens, cut into consecutive parts that each
    fit: every part is the longest prefix of the rest whose count is within ``limit``.

    The prefix is found by doubling from ``limit`` characters until it no longer
    fits, then bisecting, so each part costs a few counts of about its own length
    and a long unbroken string (a base64 blob) is cut in linear time.
    """
    parts: list[Window] = []
    rest = piece
    while rest:
        fits, over = 0, min(len(rest), limit)
        while count_tokens(rest[:over]) <= limit:
            fits = over
            if over == len(rest):
                break
            over = min(len(rest), over * 2)
        if fits == len(rest):
            parts.append((rest, count_tokens(rest)))
            break
        # rest[:fits] fits (nothing yet when fits is 0) and rest[:over] does not.
        low, high = max(fits, 1), over - 1  # one character always fits
        while low < high:
            middle = (low + high + 1) // 2
            if count_tokens(rest[:middle]) <= limit:
                low = middle
            else:
                high = middle - 1
        parts.append((rest[:low], count_tokens(rest[:low])))
        rest = rest[low:]
    return parts


def windows_from_encoding(text: str, encoding, limit: int) -> list[Window]:
    """Cut ``text`` into consecutive windows of at most ``limit`` tokens, from the
    model's own ``tokenizers`` encoding of it.

    ``encoding`` and its ``overflowing`` pieces together hold every token of the
    text with its character offsets and word index. A cut lands at a word start,
    so each window re-tokenizes to exactly the tokens it was cut from; a single
    word longer than the limit is cut inside, at a token boundary.
    """
    if not encoding.overflowing:
        content = sum(word is not None for word in encoding.word_ids)
        if content <= limit:
            return [(text, content)]
    tokens = [
        (offsets, word)
        for piece in (encoding, *encoding.overflowing)
        for offsets, word in zip(piece.offsets, piece.word_ids)
        if word is not None  # special and padding tokens: offsets (0, 0)
    ]
    windows: list[Window] = []
    start = 0
    while start < len(tokens):
        end = min(start + limit, len(tokens))
        if end < len(tokens) and tokens[end][1] == tokens[end - 1][1]:
            word_start = end
            while word_start > start and tokens[word_start - 1][1] == tokens[end][1]:
                word_start -= 1
            if word_start > start:
                end = word_start
        windows.append((text[tokens[start][0][0] : tokens[end - 1][0][1]], end - start))
        start = end
    return windows


def pool_windows(
    windows: Sequence[Sequence[Window]],
    vectors: Sequence[Sequence[float]],
    unit_length: bool = False,
) -> list[list[float]]:
    """One vector per text: the mean of its windows' vectors, weighted by the tokens
    each window holds, and scaled back to unit length when ``unit_length`` is set
    (an engine whose vectors are all unit length says so). ``vectors`` are in window
    order; a single window is its own vector, unchanged."""
    pooled: list[list[float]] = []
    position = 0
    for text_windows in windows:
        text_vectors = vectors[position : position + len(text_windows)]
        position += len(text_windows)
        if len(text_windows) == 1:
            pooled.append(list(text_vectors[0]))
            continue
        weights = [max(count, 1) for _, count in text_windows]
        mean = np.average(np.asarray(text_vectors, dtype=np.float64), axis=0, weights=weights)
        norm = float(np.linalg.norm(mean))
        if unit_length and norm > 0:
            mean /= norm
        pooled.append(mean.tolist())
    return pooled


async def embed_in_windows(
    engine,
    texts: list[str],
    embed: Callable[[list[str]], Awaitable[EmbeddingResult]],
    unit_length: bool = False,
) -> EmbeddingResult:
    """Embed ``texts`` with ``embed`` (the engine's own request), each over-length
    text as consecutive windows cut by ``engine.tokenizer``, pooled per text.

    A mock engine embeds nothing, so its texts go to ``embed`` as they are. The
    tokens are what ``embed`` reports for the windows, so a long text counts once.
    A text that is not a string is left to ``embed``, which handles it as before.
    """
    texts = texts if isinstance(texts, list) else [texts]
    if getattr(engine, "mock", False) or not texts:
        return await embed(texts)
    limit = await resolve_input_limit(engine)
    if all(not isinstance(text, str) or _fits_unseen(text, limit) for text in texts):
        return await embed(texts)
    count_tokens = engine.tokenizer.count_tokens
    windows = [
        windows_by_count(text, count_tokens, limit) if isinstance(text, str) else [(text, 0)]
        for text in texts
    ]
    log_windows(engine, texts, windows, limit)
    if all(len(text_windows) == 1 for text_windows in windows):
        return await embed(texts)
    # Windows add inputs, so they go in requests of at most the engine's batch
    # size, the cap the gateway sized the call to (TEI refuses more than 32).
    inputs = [window for text_windows in windows for window, _ in text_windows]
    size = max(1, engine.get_batch_size())
    vectors: list[list[float]] = []
    tokens = 0
    for start in range(0, len(inputs), size):
        part = await embed(inputs[start : start + size])
        vectors.extend(part)
        tokens += part.tokens
    return EmbeddingResult(
        vectors=pool_windows(windows, vectors, unit_length=unit_length), tokens=tokens
    )


def log_windows(
    engine, texts: Sequence[str], windows: Sequence[Sequence[Window]], limit: int
) -> None:
    """A debug line for every text embedded as more than one window."""
    for text, text_windows in zip(texts, windows):
        if len(text_windows) > 1:
            logger.debug(
                "Text of %s characters exceeds what %s embeds at once (%s tokens); "
                "embedding it as %s consecutive windows.",
                len(text),
                engine.model,
                limit,
                len(text_windows),
            )
