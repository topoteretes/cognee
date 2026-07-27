"""Chunk-level diff planning for incremental document updates.

Given the old processed text, its stored chunk texts (in document order), and
the new processed text, compute the minimal update plan:

  - which old chunks are affected by the edit,
  - the replacement region of the new text (edit span expanded to the affected
    chunks' boundaries),
  - a balanced re-split of that region into chunks whose token size stays
    under ``max_chunk_size``.

The plan carries a hard no-loss invariant: untouched prefix chunks + new
chunks + untouched suffix chunks must reassemble into exactly the new text.
"""

import re
from dataclasses import dataclass, field
from math import ceil
from typing import Callable, List

from cognee.tasks.chunks.chunk_by_sentence import get_word_size


class IncrementalPlanError(Exception):
    """The stored chunks cannot be diffed against the old text (e.g. no tiling)."""


@dataclass
class IncrementalPlan:
    """Result of the chunk-level diff between old and new document text."""

    affected_indices: List[int]  # indices into the old chunk list to delete
    new_chunk_texts: List[str]  # balanced replacement chunks for the region
    unchanged_prefix_count: int
    unchanged_suffix_count: int
    replacement_region: str = field(repr=False, default="")


def _common_prefix_len(a: str, b: str) -> int:
    limit = min(len(a), len(b))
    i = 0
    while i < limit and a[i] == b[i]:
        i += 1
    return i


def _common_suffix_len(a: str, b: str, prefix_len: int) -> int:
    limit = min(len(a), len(b)) - prefix_len  # the suffix may not overlap the prefix
    i = 0
    while i < limit and a[len(a) - 1 - i] == b[len(b) - 1 - i]:
        i += 1
    return i


def chunk_offsets(old_text: str, old_chunks: List[str]) -> List[tuple]:
    """[(start, end)] of each chunk inside old_text; the chunks must tile it."""
    offsets = []
    cursor = 0
    for index, chunk in enumerate(old_chunks):
        if not old_text.startswith(chunk, cursor):
            raise IncrementalPlanError(
                f"stored chunk {index} does not tile the stored document text"
            )
        offsets.append((cursor, cursor + len(chunk)))
        cursor += len(chunk)
    if cursor != len(old_text):
        raise IncrementalPlanError("stored chunks do not cover the stored document text")
    return offsets


def _words_with_separators(text: str) -> List[str]:
    """Split into units that concatenate back to the exact text (word + its whitespace)."""
    pieces = re.split(r"(\s+)", text)
    units: List[str] = []
    for piece in pieces:
        if not piece:
            continue
        if units and piece.isspace():
            units[-1] += piece
        else:
            units.append(piece)
    return units


def balanced_token_split(
    text: str, max_chunk_size: int, word_size: Callable[[str], int] = get_word_size
) -> List[str]:
    """Split text into near-equal pieces, each within the token budget.

    Piece count starts at ceil(total_tokens / max_chunk_size) and cut points sit
    at the word boundary nearest each even token target, so sizes stay balanced
    (the "split between the number of characters there" behaviour). If boundary
    nudging pushes a piece over budget, the split retries one piece finer.
    """
    if not text:
        return []
    units = _words_with_separators(text)
    sizes = [word_size(unit.rstrip()) for unit in units]
    total = sum(sizes)

    count = max(1, ceil(total / max_chunk_size))
    while count <= len(units):
        target = total / count
        pieces, piece, piece_tokens, budget_ok = [], [], 0, True
        boundary = 1
        for unit, size in zip(units, sizes):
            if piece and piece_tokens + size > min(max_chunk_size, ceil(boundary * target)):
                pieces.append("".join(piece))
                piece, piece_tokens = [], 0
                boundary += 1
            piece.append(unit)
            piece_tokens += size
            if piece_tokens > max_chunk_size and len(piece) > 1:
                budget_ok = False
        if piece:
            pieces.append("".join(piece))
        if budget_ok:
            return pieces
        count += 1
    # Every unit on its own still busts the budget only if a single word does;
    # the standard chunker cannot split words either, so neither do we.
    return units


def compute_incremental_plan(
    old_text: str,
    old_chunks: List[str],
    new_text: str,
    max_chunk_size: int,
    word_size: Callable[[str], int] = get_word_size,
) -> IncrementalPlan:
    """Diff old vs new text and plan the minimal chunk replacement."""
    offsets = chunk_offsets(old_text, old_chunks)

    prefix = _common_prefix_len(old_text, new_text)
    suffix = _common_suffix_len(old_text, new_text, prefix)
    changed_start, changed_end = prefix, len(old_text) - suffix

    if changed_start >= changed_end and len(old_text) == len(new_text):
        return IncrementalPlan([], [], len(old_chunks), 0, "")

    affected = [
        i for i, (start, end) in enumerate(offsets) if start < changed_end and end > changed_start
    ]
    if not affected:
        # Pure insertion exactly at a chunk boundary: attach to the chunk ending there.
        anchor = next(
            (i for i, (_, end) in enumerate(offsets) if end >= changed_start),
            len(offsets) - 1,
        )
        affected = [anchor]

    region_start = offsets[affected[0]][0]
    region_end_from_tail = len(old_text) - offsets[affected[-1]][1]
    replacement_region = new_text[region_start : len(new_text) - region_end_from_tail]

    plan = IncrementalPlan(
        affected_indices=affected,
        new_chunk_texts=balanced_token_split(replacement_region, max_chunk_size, word_size),
        unchanged_prefix_count=affected[0],
        unchanged_suffix_count=len(old_chunks) - affected[-1] - 1,
        replacement_region=replacement_region,
    )

    reassembled = (
        "".join(old_chunks[: plan.unchanged_prefix_count])
        + "".join(plan.new_chunk_texts)
        + "".join(old_chunks[len(old_chunks) - plan.unchanged_suffix_count :])
    )
    if reassembled != new_text:
        raise IncrementalPlanError("incremental plan would lose or corrupt content")
    return plan
