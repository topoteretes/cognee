"""Chunk-level diff planning for incremental document updates.

Given the old processed text, its stored chunk texts (in document order), and
the new processed text, compute the minimal update plan:

  - which old chunks are affected by the edit,
  - the replacement region of the new text (edit span expanded to the affected
    chunks' boundaries).

Splitting the region into chunks is the caller's job (the update flow runs the
standard TextChunker over it, so replacement chunks keep the same sentence and
paragraph boundary semantics as pipeline chunks). ``validate_no_loss`` then
enforces the hard invariant: untouched prefix chunks + the new chunk texts +
untouched suffix chunks must reassemble into exactly the new text.
"""

from dataclasses import dataclass, field
from typing import List


class IncrementalPlanError(Exception):
    """The stored chunks cannot be diffed against the old text (e.g. no tiling)."""


@dataclass
class IncrementalPlan:
    """Result of the chunk-level diff between old and new document text."""

    affected_indices: List[int]  # indices into the old chunk list to delete
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


def compute_incremental_plan(
    old_text: str, old_chunks: List[str], new_text: str
) -> IncrementalPlan:
    """Diff old vs new text and plan the minimal chunk replacement."""
    offsets = chunk_offsets(old_text, old_chunks)

    prefix = _common_prefix_len(old_text, new_text)
    suffix = _common_suffix_len(old_text, new_text, prefix)
    changed_start, changed_end = prefix, len(old_text) - suffix

    if changed_start >= changed_end and len(old_text) == len(new_text):
        return IncrementalPlan([], len(old_chunks), 0, "")

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

    return IncrementalPlan(
        affected_indices=affected,
        unchanged_prefix_count=affected[0],
        unchanged_suffix_count=len(old_chunks) - affected[-1] - 1,
        replacement_region=replacement_region,
    )


def validate_no_loss(
    old_chunks: List[str], plan: IncrementalPlan, new_chunk_texts: List[str], new_text: str
) -> None:
    """Refuse any plan whose reassembly is not byte-identical to the new text.

    With the standard TextChunker producing the replacement chunks, this also
    verifies the external contract that the chunker's output tiles its input.
    """
    reassembled = (
        "".join(old_chunks[: plan.unchanged_prefix_count])
        + "".join(new_chunk_texts)
        + "".join(old_chunks[len(old_chunks) - plan.unchanged_suffix_count :])
    )
    if reassembled != new_text:
        raise IncrementalPlanError("incremental plan would lose or corrupt content")
