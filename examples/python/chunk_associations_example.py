import asyncio
import pathlib
import os

import cognee
from cognee import memify
from cognee.api.v1.visualize.visualize import visualize_graph
from cognee.shared.logging_utils import setup_logging, ERROR
from cognee.modules.pipelines.tasks.task import Task
from cognee.tasks.memify.extract_subgraph_chunks import extract_subgraph_chunks
from cognee.tasks.chunks.create_chunk_associations import create_chunk_associations


async def main():
    print("Resetting cognee data...")
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    print("Data reset complete.\n")

    text_chunks = [
        "River dolphins are freshwater mammals found in South America and Asia.",
        "Scientists study dolphin behavior in Amazon rivers to understand their communication.",
        "Python is a high-level programming language widely used for data science.",
        "Dolphins use echolocation to navigate and hunt in murky river waters.",
        "Machine learning models require large datasets for training.",
        "The Amazon river dolphin, also known as boto, is pink in color.",
    ]

    print("Adding text chunks to cognee:\n")
    for idx, text in enumerate(text_chunks, 1):
        print(f"{idx}. {text}")
        await cognee.add(text)

    print("\nText added successfully.\n")

    print("Running cognify to create knowledge graph...")
    await cognee.cognify()
    print("Cognify process complete.\n")

    file_path = os.path.join(
        pathlib.Path(__file__).parent, ".artifacts", "graph_visualization_before_associations.html"
    )
    await visualize_graph(file_path)
    print(f"Graph visualization before associations: {file_path}\n")

    print("Running memify to create chunk associations...")

    subgraph_extraction_tasks = [Task(extract_subgraph_chunks)]

    chunk_association_tasks = [
        Task(
            create_chunk_associations,
            similarity_threshold=0.7,
            min_chunk_length=10,
            top_k_candidates=10,
            task_config={"batch_size": 10},
        ),
    ]

    await memify(
        extraction_tasks=subgraph_extraction_tasks,
        enrichment_tasks=chunk_association_tasks,
    )

    print("\nMemify process complete. Querying for chunk associations...\n")

    search_results = await cognee.search(
        query_text="""
            MATCH (c1:Node)-[a:EDGE]->(c2:Node)
            WHERE c1.type = 'DocumentChunk' AND c2.type = 'DocumentChunk' AND a.relationship_name = 'associated_with'
            RETURN c1.properties, c2.properties, a.properties
            LIMIT 20
        """,
        query_type=cognee.SearchType.CYPHER,
    )

    if search_results and len(search_results) > 0:
        import json

        all_rows = []
        for result_wrapper in search_results:
            for dataset_results in result_wrapper.get("search_result", []):
                for row in dataset_results:
                    all_rows.append(row)

        print(f"Found {len(all_rows)} chunk associations:\n")
        for idx, row in enumerate(all_rows, 1):
            props1 = json.loads(row[0]) if isinstance(row[0], str) else (row[0] or {})
            props2 = json.loads(row[1]) if isinstance(row[1], str) else (row[1] or {})
            edge_props = json.loads(row[2]) if isinstance(row[2], str) else (row[2] or {})
            print(f"{idx}. Similarity: {edge_props.get('weight', 0):.2f}")
            print(f"   Type: {edge_props.get('association_type', 'N/A')}")
            print(f"   Chunk 1: {props1.get('text', '')[:80]}")
            print(f"   Chunk 2: {props2.get('text', '')[:80]}")
            print(f"   Reasoning: {edge_props.get('reasoning', '')}")
            print()
    else:
        print("No associations found. Possible reasons:")
        print("- Similarity threshold too high")
        print("- Chunks not semantically related")
        print("- LLM API issues")

    file_path = os.path.join(
        pathlib.Path(__file__).parent, ".artifacts", "graph_visualization_with_associations.html"
    )
    await visualize_graph(file_path)
    print(f"\nGraph visualization with associations: {file_path}")


if __name__ == "__main__":
    logger = setup_logging(log_level=ERROR)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
