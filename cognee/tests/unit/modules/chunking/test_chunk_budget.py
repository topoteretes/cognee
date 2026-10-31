"""Per-chunk token-budget recording and the region-budget selection rule.

Every chunk records the budget it was cut against (``max_chunk_tokens``);
incremental updates re-chunk a region with the budget recorded on the chunks
it replaces, so a document stays self-consistent across global config changes.
Legacy chunks carry no recorded budget and fall back to the current config.
"""

import asyncio

import pytest

from cognee.modules.chunking.chunk_policy import IncrementalPlanError, _region_chunk_budget
from cognee.modules.chunking.incremental_chunking import ReplacementRegion
from cognee.modules.chunking.TextChunker import TextChunker
from cognee.modules.data.processing.document_types.Document import Document


def _region(indices):
    return ReplacementRegion(affected_indices=indices, replacement_text="x")


def test_region_budget_uses_replaced_chunks_record():
    stored = [{"max_chunk_tokens": 60}, {"max_chunk_tokens": 60}, {"max_chunk_tokens": 60}]
    assert _region_chunk_budget(stored, _region([1, 2]), fallback=400) == 60


def test_region_budget_skips_unrecorded_and_uses_first_recorded():
    stored = [{}, {"max_chunk_tokens": None}, {"max_chunk_tokens": 128}]
    assert _region_chunk_budget(stored, _region([0, 1, 2]), fallback=400) == 128


def test_region_budget_falls_back_for_legacy_chunks():
    stored = [{}, {}, {}]
    assert _region_chunk_budget(stored, _region([0, 1]), fallback=400) == 400


def test_region_budget_refuses_when_recorded_budget_exceeds_current_limit():
    stored = [{"max_chunk_tokens": 400}]

    with pytest.raises(
        IncrementalPlanError,
        match="stored chunk budget 400 exceeds current provider limit 60",
    ):
        _region_chunk_budget(stored, _region([0]), fallback=60)


def test_text_chunker_stamps_the_budget_it_cut_against():
    class FakeDoc(Document):
        type: str = "text"

        async def read(self, *args, **kwargs): ...

    async def scenario():
        text = ("word " * 30 + ".\n\n") * 6

        async def get_text():
            yield text

        document = FakeDoc(name="d", raw_data_location="loc", external_metadata="", mime_type="")
        chunker = TextChunker(document, get_text, max_chunk_size=50)
        chunks = [chunk async for chunk in chunker.read()]
        assert len(chunks) > 1
        assert all(chunk.max_chunk_tokens == 50 for chunk in chunks)

    asyncio.run(scenario())
