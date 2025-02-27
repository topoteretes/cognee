# TODO: delete after merging COG-1365, see COG-1403
import asyncio
import json
import logging
import os
from typing import Any, Callable, Dict, Type

from cognee.modules.retrieval.chunks_retriever import ChunksRetriever
from cognee.modules.retrieval.code_retriever import CodeRetriever
from cognee.modules.retrieval.completion_retriever import CompletionRetriever
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from cognee.modules.retrieval.graph_summary_completion_retriever import (
    GraphSummaryCompletionRetriever,
)
from cognee.modules.retrieval.insights_retriever import InsightsRetriever
from cognee.modules.retrieval.summaries_retriever import SummariesRetriever
from cognee.modules.retrieval.utils.code_graph_retrieval import code_graph_retrieval
from cognee.tasks.chunks import query_chunks
from cognee.tasks.completion import (
    query_completion,
    graph_query_completion,
    graph_query_summary_completion,
)
from cognee.tasks.graph import query_graph_connections
from cognee.tasks.summarization import query_summaries
from examples.python.dynamic_steps_example import main as setup_main


CONTEXT_DUMP_DIR = "context_dumps"

# Define retriever configurations
COMPLETION_RETRIEVERS = [
    {
        "name": "completion",
        "old_implementation": query_completion,
        "retriever_class": CompletionRetriever,
        "type": "completion",
    },
    {
        "name": "graph completion",
        "old_implementation": graph_query_completion,
        "retriever_class": GraphCompletionRetriever,
        "type": "graph_completion",
    },
    {
        "name": "graph summary completion",
        "old_implementation": graph_query_summary_completion,
        "retriever_class": GraphSummaryCompletionRetriever,
        "type": "graph_summary_completion",
    },
]

BASIC_RETRIEVERS = [
    {
        "name": "summaries search",
        "old_implementation": query_summaries,
        "retriever_class": SummariesRetriever,
    },
    {
        "name": "chunks search",
        "old_implementation": query_chunks,
        "retriever_class": ChunksRetriever,
    },
    {
        "name": "insights search",
        "old_implementation": query_graph_connections,
        "retriever_class": InsightsRetriever,
    },
    {
        "name": "code search",
        "old_implementation": code_graph_retrieval,
        "retriever_class": CodeRetriever,
    },
]


async def compare_completion(old_results: list, new_results: list) -> Dict:
    """Compare two lists of completion results and print differences."""
    lengths_match = len(old_results) == len(new_results)
    matches = []

    if lengths_match:
        print("Results length match")
        matches = [old == new for old, new in zip(old_results, new_results)]
        if all(matches):
            print("All entries match")
        else:
            print(f"Differences found at indices: {[i for i, m in enumerate(matches) if not m]}")
            print("\nDifferences:")
            for i, (old, new) in enumerate(zip(old_results, new_results)):
                if old != new:
                    print(f"\nIndex {i}:")
                    print("Old:", json.dumps(old, indent=2))
                    print("New:", json.dumps(new, indent=2))
    else:
        print(f"Results length mismatch: {len(old_results)} vs {len(new_results)}")
        print("\nOld results:", json.dumps(old_results, indent=2))
        print("\nNew results:", json.dumps(new_results, indent=2))

    return {
        "old_results": old_results,
        "new_results": new_results,
        "lengths_match": lengths_match,
        "element_matches": matches,
    }


async def compare_retriever(
    query: str, old_implementation: Callable, new_retriever: Any, name: str
) -> Dict:
    """Compare old and new retriever implementations."""
    print(f"\nComparing {name}...")

    # Get results from both implementations
    old_results = await old_implementation(query)
    new_results = await new_retriever.get_completion(query)

    return await compare_completion(old_results, new_results)


async def compare_completion_context(
    query: str, old_implementation: Callable, retriever_class: Type, name: str, retriever_type: str
) -> Dict:
    """Compare context between old completion implementation and new retriever."""
    print(f"\nComparing {name} contexts...")

    # Get context from old implementation with dumping
    context_path = f"{CONTEXT_DUMP_DIR}/{retriever_type}_{hash(query)}_context.json"
    os.makedirs(CONTEXT_DUMP_DIR, exist_ok=True)
    await old_implementation(query, save_context_path=context_path)

    # Get context from new implementation
    retriever = retriever_class()
    new_context = await retriever.get_context(query)

    # Read dumped context
    with open(context_path, "r") as f:
        old_context = json.load(f)

    # Compare contexts
    contexts_match = old_context == new_context
    if contexts_match:
        print("Contexts match exactly")
    else:
        print("Contexts differ:")
        print("\nOld context:", json.dumps(old_context, indent=2))
        print("\nNew context:", json.dumps(new_context, indent=2))

    return {
        "old_context": old_context,
        "new_context": new_context,
        "contexts_match": contexts_match,
    }


async def main(query: str, comparisons: Dict[str, bool], setup_steps: Dict[str, bool]):
    """Run comparison tests for selected retrievers with the given setup configuration."""
    # Ensure retriever is always False in setup steps
    setup_steps["retriever"] = False
    await setup_main(setup_steps)

    # Compare contexts for completion-based retrievers
    for retriever in COMPLETION_RETRIEVERS:
        context_key = f"{retriever['type']}_context"
        if comparisons.get(context_key, False):
            await compare_completion_context(
                query=query,
                old_implementation=retriever["old_implementation"],
                retriever_class=retriever["retriever_class"],
                name=retriever["name"],
                retriever_type=retriever["type"],
            )

    # Run completion comparisons
    for retriever in COMPLETION_RETRIEVERS:
        if comparisons.get(retriever["type"], False):
            await compare_retriever(
                query=query,
                old_implementation=retriever["old_implementation"],
                new_retriever=retriever["retriever_class"](),
                name=retriever["name"],
            )

    # Run basic retriever comparisons
    for retriever in BASIC_RETRIEVERS:
        retriever_type = retriever["name"].split()[0]
        if comparisons.get(retriever_type, False):
            await compare_retriever(
                query=query,
                old_implementation=retriever["old_implementation"],
                new_retriever=retriever["retriever_class"](),
                name=retriever["name"],
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)

    test_query = "Who has experience in data science?"
    comparisons = {
        # Context comparisons
        "completion_context": True,
        "graph_completion_context": True,
        "graph_summary_completion_context": True,
        # Result comparisons
        "summaries": True,
        "chunks": True,
        "insights": True,
        "code": False,
        "completion": True,
        "graph_completion": True,
        "graph_summary_completion": True,
    }
    setup_steps = {
        "prune_data": True,
        "prune_system": True,
        "add_text": True,
        "cognify": True,
    }

    asyncio.run(main(test_query, comparisons, setup_steps))
