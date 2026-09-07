"""The chunk-planning seam: text in, a fully-decided plan out.

A policy answers one question — given a document's old text, its stored chunks,
and its new text, which chunks should exist afterwards and what happens to the
old ones. It runs no models and touches no store, so it is testable with
strings.

The plan it returns is COMPLETE: the writer executes it without deciding
anything. That completeness is what makes the seam real. While the writer had
to work out fresh-vs-reused itself, any replacement policy would have had to
reproduce that private logic, and "swap the policy without touching storage or
orchestration" could not hold.

Two boundaries are deliberate:

- **Surviving chunks are named, not built.** ``reused`` and ``kept_moves`` carry
  ids and target positions rather than DocumentChunks, because rebuilding a
  stored chunk means carrying ~15 fields across from the stored node — adapters
  replace a node's whole property set on MERGE, so a field the rebuild forgets
  is erased, not merely reset. That is storage knowledge; keeping it out of the
  policy means a third-party policy cannot destroy data by not knowing the rule.
- **The policy does not validate itself.** ``validate_no_loss`` here is the
  default policy checking its own region arithmetic. The orchestrator
  independently re-derives the final document from the returned plan and
  compares it to the new text, trusting nothing.
"""

from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, List, Optional

from cognee.modules.chunking.chunk_id import chunk_content_hash, content_chunk_id
from cognee.modules.chunking.incremental_chunking import (
    IncrementalPlan,
    IncrementalPlanError,
    compute_incremental_plan,
    validate_no_loss,
)
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk


@dataclass
class ChunkPlanRequest:
    """Everything a policy needs, and nothing that ties it to a store."""

    old_text: str
    new_text: str
    stored_chunks: List[dict]  # full chunk nodes, in document order
    document: object  # the Document new chunks attach to
    chunker_cls: type
    fallback_budget: int  # token budget for chunks with none recorded


@dataclass
class ChunkPlan:
    """A complete decision about the document's chunks after the edit."""

    # Genuinely new content: needs LLM extraction and a fresh write.
    fresh: List[DocumentChunk] = field(default_factory=list)
    # Replacement chunks byte-identical to one being replaced, so they keep
    # their node id and subgraph: stored id -> final chunk_index.
    reused: Dict[str, int] = field(default_factory=dict)
    # Untouched chunks whose position moved: stored id -> final chunk_index.
    kept_moves: Dict[str, int] = field(default_factory=dict)
    # Stored chunk ids that no longer exist after the edit.
    deleted_ids: List[str] = field(default_factory=list)
    # How many disjoint regions the edit touched (reporting only).
    regions: int = 0


ChunkPolicy = Callable[[ChunkPlanRequest], Awaitable[ChunkPlan]]


async def _chunk_region(
    document, region_text: str, max_chunk_size: int, chunker_cls: type
) -> List[DocumentChunk]:
    """Run the configured chunker over one replacement region.

    Replacement chunks get the same boundary semantics as pipeline chunks; only
    the region's last chunk may come out under-filled, like the tail of any
    normally-cognified document. The chunker's ids and indexes are region-local
    and discarded — the assembly below reassigns both.
    """
    if not region_text:
        return []

    async def get_text():
        yield region_text

    chunker = chunker_cls(document, get_text, max_chunk_size)
    return [chunk async for chunk in chunker.read()]


def _region_chunk_budget(stored_chunks: List[dict], region, fallback: int) -> int:
    """Token budget for re-chunking one region.

    The budget recorded on the chunks the region replaces wins when the current
    providers can still accept it, so an edit keeps the granularity of the text
    around it across safe configuration changes. A larger recorded budget is
    refused into a full rebuild; legacy chunks fall back to the current limit.
    """
    for position in region.affected_indices:
        recorded = stored_chunks[position].get("max_chunk_tokens")
        if recorded:
            recorded_budget = int(recorded)
            if recorded_budget > fallback:
                raise IncrementalPlanError(
                    f"stored chunk budget {recorded_budget} exceeds current provider "
                    f"limit {fallback}; run a full update"
                )
            return recorded_budget
    return fallback


def _assemble_final_chunks(
    document,
    stored_chunks: List[dict],
    plan: IncrementalPlan,
    region_chunk_lists: List[List[DocumentChunk]],
) -> tuple:
    """Walk the final document order once: kept chunks and region chunks interleaved.

    Returns (region_chunks, kept_final_index). Region chunks carry
    document-scoped content-hash ids (occurrence counted over the FINAL order,
    so two identical texts stay distinct; surviving legacy ids are dodged by
    bumping the occurrence). kept_final_index maps each kept chunk's old
    position to its final chunk_index — kept chunks are never rebuilt here,
    which is what stops their boundaries and ids drifting.
    """
    affected = set(plan.affected_indices)
    surviving_ids = {
        str(node["id"]) for position, node in enumerate(stored_chunks) if position not in affected
    }
    region_by_start = {
        region.affected_indices[0]: index for index, region in enumerate(plan.regions)
    }

    occurrences: dict = {}
    region_chunks: List[DocumentChunk] = []
    kept_final_index: dict = {}
    final_index = 0
    position = 0
    while position < len(stored_chunks):
        if position in region_by_start:
            region_index = region_by_start[position]
            for region_chunk in region_chunk_lists[region_index]:
                text = region_chunk.text
                content_hash = chunk_content_hash(text)
                occurrence = occurrences.get(content_hash, 0)
                chunk_id = content_chunk_id(document.id, content_hash, occurrence)
                while str(chunk_id) in surviving_ids:
                    occurrence += 1
                    chunk_id = content_chunk_id(document.id, content_hash, occurrence)
                occurrences[content_hash] = occurrence + 1
                region_chunks.append(
                    DocumentChunk(
                        id=chunk_id,
                        text=text,
                        chunk_size=region_chunk.chunk_size,
                        content_hash=content_hash,
                        max_chunk_tokens=region_chunk.max_chunk_tokens,
                        chunker_id=region_chunk.chunker_id,
                        chunk_index=final_index,
                        cut_type=region_chunk.cut_type,
                        is_part_of=document,
                        contains=[],
                        importance_weight=region_chunk.importance_weight,
                        document_id=str(document.id),
                        document_name=document.name,
                    )
                )
                final_index += 1
            position = plan.regions[region_index].affected_indices[-1] + 1
        else:
            text_hash = chunk_content_hash(stored_chunks[position]["text"])
            occurrences[text_hash] = occurrences.get(text_hash, 0) + 1
            kept_final_index[position] = final_index
            final_index += 1
            position += 1
    return region_chunks, kept_final_index


def stored_chunker_id(stored_chunks: List[dict]) -> Optional[str]:
    """The chunker every stored chunk agrees on, or None when unknown.

    None means "cannot tell" — legacy chunks predate the field, and a document
    whose chunks disagree has already been through something this path should
    not reason about. Both fall through to the tiling check, so v1 graphs are
    unaffected.
    """
    ids = {node.get("chunker_id") for node in stored_chunks}
    if len(ids) != 1:
        return None
    only = ids.pop()
    return only or None


async def diff_region_policy(request: ChunkPlanRequest) -> ChunkPlan:
    """The default policy: replace only the regions the edit touched.

    A paragraph-anchored diff yields the disjoint changed spans, each is
    expanded to the boundaries of the chunks it overlaps and re-chunked at the
    budget those chunks recorded, and everything between regions is kept as-is.
    """
    stored_texts = [node["text"] for node in request.stored_chunks]
    plan = compute_incremental_plan(request.old_text, stored_texts, request.new_text)

    region_chunk_lists = [
        await _chunk_region(
            request.document,
            region.replacement_text,
            _region_chunk_budget(request.stored_chunks, region, request.fallback_budget),
            request.chunker_cls,
        )
        for region in plan.regions
    ]
    region_chunks, kept_final_index = _assemble_final_chunks(
        request.document, request.stored_chunks, plan, region_chunk_lists
    )

    # Self-check on this policy's own region arithmetic. The orchestrator
    # re-validates the assembled result independently.
    validate_no_loss(
        stored_texts,
        plan,
        [[chunk.text for chunk in chunks] for chunks in region_chunk_lists],
        request.new_text,
    )

    # A replacement chunk byte-identical to one being replaced hashes to the
    # SAME id: keep its subgraph instead of re-extracting it, and exclude it
    # from deletion.
    affected = set(plan.affected_indices)
    replaced_ids = {str(request.stored_chunks[i]["id"]) for i in affected}
    reused = {
        str(chunk.id): chunk.chunk_index for chunk in region_chunks if str(chunk.id) in replaced_ids
    }
    fresh = [chunk for chunk in region_chunks if str(chunk.id) not in replaced_ids]

    kept_moves = {}
    for position, final_index in kept_final_index.items():
        node = request.stored_chunks[position]
        if int(node.get("chunk_index", -1)) != final_index:
            kept_moves[str(node["id"])] = final_index

    return ChunkPlan(
        fresh=fresh,
        reused=reused,
        kept_moves=kept_moves,
        deleted_ids=sorted(replaced_ids - set(reused)),
        regions=len(plan.regions),
    )


DEFAULT_CHUNK_POLICY: ChunkPolicy = diff_region_policy


__all__ = [
    "ChunkPlan",
    "ChunkPlanRequest",
    "ChunkPolicy",
    "DEFAULT_CHUNK_POLICY",
    "IncrementalPlanError",
    "diff_region_policy",
    "stored_chunker_id",
]
