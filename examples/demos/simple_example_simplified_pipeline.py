"""
Simplified pipeline rewrite of the custom pipeline example.

This demonstrates the same workflow as run_custom_pipeline_example.py
but using the new simplified pipeline API (run_steps, @step, cognee_pipeline).
"""

import asyncio
from pprint import pprint

import cognee
from cognee.shared.logging_utils import setup_logging, INFO
from cognee.api.v1.search import SearchType

# New simplified pipeline imports
from cognee.pipelines import run_steps, step, cognee_pipeline

# Task functions (same ones used by the original pipeline)
from cognee.tasks.documents import classify_documents, extract_chunks_from_documents
from cognee.tasks.graph import extract_graph_from_data
from cognee.tasks.summarization import summarize_text
from cognee.tasks.storage import add_data_points

from cognee.modules.data.methods.get_dataset_data import get_dataset_data
from cognee.shared.data_models import KnowledgeGraph
from cognee.infrastructure.llm import get_max_chunk_tokens
from cognee.modules.chunking.TextChunker import TextChunker


async def main():
    # Reset
    print("Resetting cognee data...")
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    print("Data reset complete.\n")

    text = """
    Natural language processing (NLP) is an interdisciplinary
    subfield of computer science and information retrieval.
    """

    print("Adding text to cognee:")
    print(text.strip())

    # --- ADD (using cognee.add — it handles setup internally) ---
    await cognee.add(text, "main_dataset")
    print("Text added successfully.\n")

    # --- COGNIFY PIPELINE (using run_steps with @step config) ---
    # cognee_pipeline() sets up DB isolation, permissions, ContextVars
    print("Running cognify with simplified pipeline...\n")

    async with cognee_pipeline(dataset="main_dataset") as dataset:
        # Fetch dataset data explicitly
        data = await get_dataset_data(dataset_id=dataset.id)

        # Process each data item through the full pipeline (matches original per-item behavior)
        for data_item in data:
            await run_steps(
                classify_documents,
                step(extract_chunks_from_documents,
                     max_chunk_size=get_max_chunk_tokens(), chunker=TextChunker),
                step(extract_graph_from_data,
                     graph_model=KnowledgeGraph, batch_size=100),
                step(summarize_text, batch_size=100),
                step(add_data_points, batch_size=100),
                input=[data_item],
                context={"dataset": dataset, "data": data_item},
            )

    print("Cognify process complete.\n")

    # --- SEARCH ---
    query_text = "Tell me about NLP"
    print(f"Searching cognee for insights with query: '{query_text}'")
    search_results = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION, query_text=query_text
    )

    print("Search results:")
    for result_text in search_results:
        pprint(result_text)


if __name__ == "__main__":
    logger = setup_logging(log_level=INFO)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
