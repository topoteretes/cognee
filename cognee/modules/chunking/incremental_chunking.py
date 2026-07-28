"""Chunk-level diff planning for incremental document updates.

Given the old processed text, its stored chunk texts (in document order), and
the new processed text, compute the minimal update plan as a list of DISJOINT
replacement regions:

  - a line-anchored diff (SequenceMatcher) yields every changed hunk,
  - each hunk is expanded to the boundaries of the old chunks it overlaps,
  - hunks whose chunk ranges touch are merged into one region.

Chunks between regions are KEPT — never re-chunked, so their boundaries (and
therefore their ids, entities, and summaries) cannot drift. Splitting each
region into chunks is the caller's job (the update flow runs the standard
TextChunker over it). ``validate_no_loss`` enforces the hard invariant: kept
chunks and region chunk texts, interleaved in document order, must reassemble
into exactly the new text.
"""

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import List


class IncrementalPlanError(Exception):
    """The stored chunks cannot be diffed against the old text (e.g. no tiling)."""


@dataclass
class ReplacementRegion:
    """One contiguous run of old chunks replaced by one stretch of new text."""

    affected_indices: List[int]  # contiguous positions in the old chunk list
    replacement_text: str = field(repr=False, default="")


@dataclass
class IncrementalPlan:
    """Result of the chunk-level diff between old and new document text."""

    regions: List[ReplacementRegion]
    total_old_chunks: int

    @property
    def affected_indices(self) -> List[int]:
        indices: List[int] = []
        for region in self.regions:
            indices.extend(region.affected_indices)
        return indices


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


def _diff_spans(old_text: str, new_text: str) -> List[tuple]:
    """Disjoint changed spans as (old_start, old_end, new_start, new_end) chars.

    Line-anchored: hunks come from a SequenceMatcher over lines (keepends, so
    offsets stay byte-exact); autojunk is off because prose is full of popular
    lines (blank lines) that must still anchor the alignment.
    """
    old_lines = old_text.splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)
    matcher = SequenceMatcher(None, old_lines, new_lines, autojunk=False)

    def prefix_offsets(lines: List[str]) -> List[int]:
        offsets = [0]
        for line in lines:
            offsets.append(offsets[-1] + len(line))
        return offsets

    old_offsets = prefix_offsets(old_lines)
    new_offsets = prefix_offsets(new_lines)

    spans = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        spans.append(
            _shrink_span(
                old_text,
                new_text,
                old_offsets[i1],
                old_offsets[i2],
                new_offsets[j1],
                new_offsets[j2],
            )
        )
    return spans


def _shrink_span(
    old_text: str, new_text: str, old_start: int, old_end: int, new_start: int, new_end: int
) -> tuple:
    """Trim a line-rounded hunk back to character precision.

    Line anchoring makes a mid-line edit claim its whole line(s); shaving the
    hunk's common prefix and suffix restores the minimal changed span, which
    keeps the affected-chunk set as small as the old prefix/suffix diff did.
    """
    while (
        old_start < old_end and new_start < new_end and old_text[old_start] == new_text[new_start]
    ):
        old_start += 1
        new_start += 1
    while (
        old_end > old_start
        and new_end > new_start
        and old_text[old_end - 1] == new_text[new_end - 1]
    ):
        old_end -= 1
        new_end -= 1
    return (old_start, old_end, new_start, new_end)


def compute_incremental_plan(
    old_text: str, old_chunks: List[str], new_text: str
) -> IncrementalPlan:
    """Diff old vs new text and plan the minimal set of chunk replacements."""
    offsets = chunk_offsets(old_text, old_chunks)
    spans = _diff_spans(old_text, new_text)
    if not spans:
        if old_text != new_text:
            raise IncrementalPlanError("diff found no changes but the texts differ")
        return IncrementalPlan([], len(old_chunks))

    # Map each hunk to the contiguous run of old chunks it overlaps.
    # Each entry: [first_chunk, last_chunk, o_start, o_end, n_start, n_end]
    # where (o_start, n_start) belong to the run's EARLIEST hunk and
    # (o_end, n_end) to its LATEST — the equal text around them is what maps
    # old chunk boundaries onto new-text positions.
    runs = []
    for old_start, old_end, new_start, new_end in spans:
        overlapping = [
            i for i, (start, end) in enumerate(offsets) if start < old_end and end > old_start
        ]
        if not overlapping:
            # Pure insertion exactly at a chunk boundary: attach to the chunk
            # ending there (or the first chunk for an insertion at position 0).
            anchor = next(
                (i for i, (_, end) in enumerate(offsets) if end >= old_start),
                len(offsets) - 1,
            )
            overlapping = [anchor]
        runs.append([overlapping[0], overlapping[-1], old_start, old_end, new_start, new_end])

    runs.sort(key=lambda run: run[0])
    merged = [runs[0]]
    for run in runs[1:]:
        last = merged[-1]
        if run[0] <= last[1] + 1:  # chunk ranges overlap or touch: one region
            last[1] = max(last[1], run[1])
            if run[2] < last[2]:
                last[2], last[4] = run[2], run[4]
            if run[3] > last[3]:
                last[3], last[5] = run[3], run[5]
        else:
            merged.append(run)

    regions = []
    for first, last, old_start, old_end, new_start, new_end in merged:
        region_old_start = offsets[first][0]
        region_old_end = offsets[last][1]
        # The stretches between the region's chunk boundaries and its outermost
        # hunks are equal text, so their lengths map 1:1 onto the new text.
        region_new_start = new_start - (old_start - region_old_start)
        region_new_end = new_end + (region_old_end - old_end)
        regions.append(
            ReplacementRegion(
                affected_indices=list(range(first, last + 1)),
                replacement_text=new_text[region_new_start:region_new_end],
            )
        )

    return IncrementalPlan(regions=regions, total_old_chunks=len(old_chunks))


def interleave_texts(
    old_chunks: List[str], plan: IncrementalPlan, region_texts: List[List[str]]
) -> List[str]:
    """Final document chunk texts: kept chunks and region chunks in order."""
    region_by_start = {
        region.affected_indices[0]: index for index, region in enumerate(plan.regions)
    }
    result: List[str] = []
    position = 0
    while position < len(old_chunks):
        if position in region_by_start:
            region_index = region_by_start[position]
            result.extend(region_texts[region_index])
            position = plan.regions[region_index].affected_indices[-1] + 1
        else:
            result.append(old_chunks[position])
            position += 1
    return result


def validate_no_loss(
    old_chunks: List[str],
    plan: IncrementalPlan,
    region_texts: List[List[str]],
    new_text: str,
) -> None:
    """Refuse any plan whose reassembly is not byte-identical to the new text.

    With the standard TextChunker producing the region chunks, this also
    verifies the external contract that the chunker's output tiles its input.
    """
    if "".join(interleave_texts(old_chunks, plan, region_texts)) != new_text:
        raise IncrementalPlanError("incremental plan would lose or corrupt content")
