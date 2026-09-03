"""The chunk-planning seam.

The proof obligation from the scoping documents is that "a different policy can
replace the first policy without changing storage or update orchestration".
These tests exercise the seam directly: a replacement policy is written here,
in the test file, and the writer executes its plan without knowing anything
about it.
"""

import asyncio
from uuid import NAMESPACE_OID, uuid4, uuid5

from cognee.modules.chunking.chunk_id import chunk_content_hash, content_chunk_id

import pytest

from cognee.modules.chunking.chunk_policy import (
    ChunkPlan,
    ChunkPlanRequest,
    diff_region_policy,
    stored_chunker_id,
)
from cognee.modules.chunking.TextChunker import TextChunker
from cognee.modules.data.processing.document_types import TextDocument


def _document():
    return TextDocument(
        id=uuid5(NAMESPACE_OID, "policy-doc"),
        title="doc.txt",
        name="doc",
        raw_data_location="/tmp/doc.txt",
        mime_type="text/plain",
        external_metadata="{}",
    )


def _stored(texts, chunker_id="text_chunker_v1", budget=512):
    """Stored chunk nodes as _get_stored_chunks returns them.

    Ids are content-derived, exactly as ingestion mints them — that is what
    makes a replacement chunk with identical text hash to the SAME id and be
    reused rather than deleted and re-extracted. Faking the ids here would
    make reuse untestable.
    """
    document_id = _document().id
    occurrences: dict = {}
    nodes = []
    for index, text in enumerate(texts):
        content_hash = chunk_content_hash(text)
        occurrence = occurrences.get(content_hash, 0)
        occurrences[content_hash] = occurrence + 1
        nodes.append(
            {
                "id": str(content_chunk_id(document_id, content_hash, occurrence)),
                "text": text,
                "chunk_index": index,
                "chunk_size": len(text.split()),
                "content_hash": content_hash,
                "max_chunk_tokens": budget,
                "chunker_id": chunker_id,
            }
        )
    return nodes


def _request(old_text, stored, new_text, chunker=TextChunker):
    return ChunkPlanRequest(
        old_text=old_text,
        new_text=new_text,
        stored_chunks=stored,
        document=_document(),
        chunker_cls=chunker,
        fallback_budget=512,
    )


def _plan(old_text, stored, new_text):
    return asyncio.run(diff_region_policy(_request(old_text, stored, new_text)))


def _reassemble(plan: ChunkPlan, stored):
    """Rebuild the final document from the plan alone — the writer's view."""
    stored_text_by_id = {str(node["id"]): node["text"] for node in stored}
    placed = {chunk.chunk_index: chunk.text for chunk in plan.fresh}
    for chunk_id, index in list(plan.reused.items()) + list(plan.kept_moves.items()):
        placed[index] = stored_text_by_id[chunk_id]
    moved = set(plan.reused) | set(plan.kept_moves) | set(plan.deleted_ids)
    for node in stored:
        if str(node["id"]) not in moved:
            placed.setdefault(int(node["chunk_index"]), node["text"])
    return "".join(placed[index] for index in sorted(placed))


# ---------------------------------------------------------------- default policy


def test_unchanged_text_plans_nothing():
    texts = ["First para.\n\n", "Second para.\n\n", "Third para.\n"]
    old = "".join(texts)
    plan = _plan(old, _stored(texts), old)

    assert plan.fresh == []
    assert plan.deleted_ids == []
    assert plan.regions == 0


def test_single_mid_document_edit_touches_one_region():
    texts = ["First para.\n\n", "Second para.\n\n", "Third para.\n"]
    old = "".join(texts)
    new = "First para.\n\nSecond para EDITED.\n\nThird para.\n"
    stored = _stored(texts)
    plan = _plan(old, stored, new)

    assert plan.regions == 1
    assert _reassemble(plan, stored) == new
    # The untouched first and last chunks are neither deleted nor rebuilt.
    assert len(plan.deleted_ids) == 1


def test_three_disjoint_edits_are_three_regions():
    texts = ["Alpha.\n\n", "Beta.\n\n", "Gamma.\n\n", "Delta.\n\n", "Epsilon.\n"]
    old = "".join(texts)
    new = "Alpha EDIT.\n\nBeta.\n\nGamma EDIT.\n\nDelta.\n\nEpsilon EDIT.\n"
    stored = _stored(texts)
    plan = _plan(old, stored, new)

    assert plan.regions == 3, "disjoint edits must not collapse into one giant region"
    assert _reassemble(plan, stored) == new


def test_a_surviving_chunk_is_never_also_deleted():
    """The plan's four outcomes must not overlap on any edit shape.

    A chunk that survives — kept in place, moved, or reused because a replaced
    region regenerated it byte-identically — must not also appear in
    deleted_ids. Overlap would have the writer delete a chunk it just decided
    to keep, and the order (add, then delete) means the deletion would win.

    Note reuse itself is a narrow path: the diff is precise enough that
    unchanged content is normally classified as *kept* rather than replaced-
    and-regenerated. This asserts the invariant across shapes rather than
    contriving the rare case.
    """
    texts = ["Alpha one here.\n\n", "Beta two here.\n\n", "Gamma three here.\n"]
    old = "".join(texts)
    shapes = {
        "no-op": old,
        "edit first": "Alpha EDIT here.\n\nBeta two here.\n\nGamma three here.\n",
        "edit middle": "Alpha one here.\n\nBeta EDIT here.\n\nGamma three here.\n",
        "delete middle": "Alpha one here.\n\nGamma three here.\n",
        "insert": "Alpha one here.\n\nNEW para.\n\nBeta two here.\n\nGamma three here.\n",
        "merge separator": "Alpha one here.\nBeta two here.\n\nGamma three here.\n",
        "duplicate para": "Alpha one here.\n\nBeta two here.\n\nBeta two here.\n\nGamma three here.\n",
        "reorder": "Gamma three here.\n\nAlpha one here.\n\nBeta two here.\n",
        "whole rewrite": "Nothing.\n\nIn common.\n",
    }

    for label, new in shapes.items():
        stored = _stored(texts, budget=5)
        plan = asyncio.run(
            diff_region_policy(
                ChunkPlanRequest(
                    old_text=old,
                    new_text=new,
                    stored_chunks=stored,
                    document=_document(),
                    chunker_cls=TextChunker,
                    fallback_budget=5,
                )
            )
        )
        survivors = set(plan.reused) | set(plan.kept_moves)
        assert survivors.isdisjoint(plan.deleted_ids), f"{label}: survivor also deleted"
        assert _reassemble(plan, stored) == new, f"{label}: does not reassemble"


def test_whole_document_rewrite_deletes_everything():
    texts = ["Alpha.\n\n", "Beta.\n\n", "Gamma.\n"]
    old = "".join(texts)
    new = "Totally different content.\n\nNothing in common.\n"
    stored = _stored(texts)
    plan = _plan(old, stored, new)

    assert len(plan.deleted_ids) == len(texts)
    assert plan.kept_moves == {}
    assert _reassemble(plan, stored) == new


def test_plan_always_reassembles_into_the_new_text():
    """The no-loss invariant, checked from the plan the writer receives."""
    texts = ["One.\n\n", "Two.\n\n", "Three.\n\n", "Four.\n"]
    old = "".join(texts)
    for new in (
        "One EDIT.\n\nTwo.\n\nThree.\n\nFour.\n",
        "One.\n\nTwo.\n\nThree.\n\nFour EDIT.\n",
        "One.\n\nInserted.\n\nTwo.\n\nThree.\n\nFour.\n",
        "One.\n\nFour.\n",
    ):
        stored = _stored(texts)
        assert _reassemble(_plan(old, stored, new), stored) == new, new


# ------------------------------------------------------------- chunker identity


def test_stored_chunker_id_reports_agreement():
    assert stored_chunker_id(_stored(["a\n", "b\n"])) == "text_chunker_v1"


def test_stored_chunker_id_is_unknown_for_legacy_chunks():
    """Chunks written before the field exists must not be refused."""
    assert stored_chunker_id(_stored(["a\n", "b\n"], chunker_id=None)) is None


def test_stored_chunker_id_is_unknown_when_chunks_disagree():
    mixed = _stored(["a\n", "b\n"])
    mixed[1]["chunker_id"] = "langchain_chunker_v1"
    assert stored_chunker_id(mixed) is None


def test_every_chunker_declares_a_distinct_id():
    """Identity is useless if two strategies share it, or if one omits it."""
    from cognee.modules.chunking.Chunker import Chunker
    from cognee.modules.chunking.CsvChunker import CsvChunker
    from cognee.modules.chunking.JsonListChunker import JsonListChunker
    from cognee.modules.chunking.text_chunker_with_overlap import TextChunkerWithOverlap

    chunkers = [TextChunker, CsvChunker, JsonListChunker, TextChunkerWithOverlap]
    ids = [c.chunker_id for c in chunkers]
    assert all(ids), f"a chunker declares no id: {dict(zip([c.__name__ for c in chunkers], ids))}"
    assert len(set(ids)) == len(ids), f"chunkers share an id: {ids}"
    assert Chunker.chunker_id == "", "the base must not claim an identity"


def test_langchain_chunker_declares_an_id():
    pytest.importorskip("langchain_text_splitters")
    from cognee.modules.chunking.LangchainChunker import LangchainChunker

    assert LangchainChunker.chunker_id == "langchain_chunker_v1"


def test_text_chunker_stamps_its_id_on_every_chunk():
    async def scenario():
        document = _document()

        async def get_text():
            yield "First para.\n\nSecond para.\n\nThird para.\n"

        chunker = TextChunker(document, get_text, 512)
        return [chunk async for chunk in chunker.read()]

    chunks = asyncio.run(scenario())
    assert chunks
    assert all(chunk.chunker_id == "text_chunker_v1" for chunk in chunks)


# ------------------------------------------------------------ the seam itself


async def whole_document_policy(request: ChunkPlanRequest) -> ChunkPlan:
    """A replacement policy: reuse nothing, re-chunk the whole document.

    Written entirely here, using only the public request/plan types. If this
    needs any change under api/v1/update/ or in the storage layer to work, the
    seam is not real.
    """
    from cognee.modules.chunking.models.DocumentChunk import DocumentChunk

    async def get_text():
        yield request.new_text

    chunker = request.chunker_cls(request.document, get_text, request.fallback_budget)
    fresh = []
    async for index, chunk in _enumerate_async(chunker.read()):
        fresh.append(
            DocumentChunk(
                id=uuid4(),
                text=chunk.text,
                chunk_size=chunk.chunk_size,
                chunk_index=index,
                cut_type=chunk.cut_type,
                chunker_id=chunk.chunker_id,
                is_part_of=request.document,
                contains=[],
            )
        )
    return ChunkPlan(
        fresh=fresh,
        reused={},
        kept_moves={},
        deleted_ids=[str(node["id"]) for node in request.stored_chunks],
        regions=1,
    )


async def _enumerate_async(iterator):
    index = 0
    async for item in iterator:
        yield index, item
        index += 1


def test_a_replacement_policy_produces_a_complete_plan():
    """The proof obligation: a foreign policy's plan is directly executable.

    The writer reads only fresh / reused / kept_moves / deleted_ids, so a plan
    built by a policy that has never heard of regions is executed the same way.
    """
    texts = ["Alpha.\n\n", "Beta.\n\n", "Gamma.\n"]
    old = "".join(texts)
    new = "Rewritten alpha.\n\nRewritten beta.\n"
    stored = _stored(texts)

    plan = asyncio.run(whole_document_policy(_request(old, stored, new)))

    assert plan.reused == {}
    assert set(plan.deleted_ids) == {str(node["id"]) for node in stored}
    assert _reassemble(plan, stored) == new
    assert plan.fresh, "every chunk is fresh under this policy"


def test_the_policy_is_injectable_without_touching_the_orchestrator():
    """update() and incremental_update() accept a policy, defaulting to the diff one."""
    import inspect

    from cognee.api.v1.update.incremental import incremental_update
    from cognee.api.v1.update.update import update
    from cognee.modules.chunking.chunk_policy import DEFAULT_CHUNK_POLICY

    for function in (update, incremental_update):
        parameter = inspect.signature(function).parameters["policy"]
        assert parameter.default is DEFAULT_CHUNK_POLICY, function.__name__

    assert DEFAULT_CHUNK_POLICY is diff_region_policy


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
