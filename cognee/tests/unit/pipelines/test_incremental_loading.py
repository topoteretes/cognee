"""
Integration test for incremental loading in simplified pipeline.

Tests that skip_processed=True correctly skips already-processed items
and marks new items as completed.
"""

import pytest

import cognee
from cognee.pipelines import run_steps, step, cognee_pipeline
from cognee.tasks.documents import classify_documents, extract_chunks_from_documents
from cognee.tasks.graph import extract_graph_from_data
from cognee.tasks.summarization import summarize_text
from cognee.tasks.storage import add_data_points
from cognee.modules.data.methods.get_dataset_data import get_dataset_data
from cognee.shared.data_models import KnowledgeGraph
from cognee.infrastructure.llm import get_max_chunk_tokens
from cognee.modules.chunking.TextChunker import TextChunker


@pytest.mark.asyncio
async def test_incremental_loading_skips_processed():
    """Run cognify twice — second run should skip already-processed items."""
    # Reset
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    text = "Natural language processing is a subfield of computer science."
    await cognee.add(text, "test_incremental")

    call_count = {"classify": 0}
    original_classify = classify_documents

    async def counting_classify(data):
        call_count["classify"] += 1
        return await original_classify(data)

    cognify_steps = [
        counting_classify,
        step(
            extract_chunks_from_documents,
            max_chunk_size=get_max_chunk_tokens(),
            chunker=TextChunker,
        ),
        step(extract_graph_from_data, graph_model=KnowledgeGraph, batch_size=100),
        step(summarize_text, batch_size=100),
        step(add_data_points, batch_size=100),
    ]

    async with cognee_pipeline(dataset="test_incremental") as dataset:
        data = await get_dataset_data(dataset_id=dataset.id)
        assert len(data) == 1

        # First run — process each data item through the chain
        for data_item in data:
            await run_steps(
                *cognify_steps,
                input=[data_item],
                context={"dataset": dataset, "data": data_item},
                pipeline_name="test_cognify",
                skip_processed=True,
            )

        assert call_count["classify"] == 1

        # Second run — same data, should be skipped by incremental loading
        data2 = await get_dataset_data(dataset_id=dataset.id)
        for data_item in data2:
            result = await run_steps(
                *cognify_steps,
                input=[data_item],
                context={"dataset": dataset, "data": data_item},
                pipeline_name="test_cognify",
                skip_processed=True,
            )
            # Result should be empty since item was skipped
            assert result == []

        # classify should NOT have been called again
        assert call_count["classify"] == 1
