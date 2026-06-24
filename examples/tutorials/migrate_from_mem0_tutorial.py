"""
Tutorial: migrate memories from mem0 to Cognee.

Import a mem0 export with Mem0Source, then query the migrated memories with
cognee.recall().

Usage:
    uv run python examples/tutorials/migrate_from_mem0_tutorial.py

This example uses "preserve" mode to migrate mem0 memories into Cognee without
re-running extraction. After import, cognify() processes the migrated memories
so they can be queried with recall().
"""

import asyncio
import importlib
import json
import os
from pathlib import Path
from typing import Any

# Configure mock flags before loading core components
USE_SAMPLE_MOCKS = os.getenv("COGNEE_TUTORIAL_USE_MOCKS", "true").lower() in ("true", "1", "yes")
if USE_SAMPLE_MOCKS:
    os.environ.setdefault("COGNEE_SKIP_CONNECTION_TEST", "true")
    os.environ.setdefault("LLM_API_KEY", "mock-key")
    os.environ.setdefault("MOCK_EMBEDDING", "true")

DATASET_NAME = "mem0_migration_tutorial"
MODE = "preserve"
SAMPLE_DUMP_PATH = Path(__file__).parent / "data" / "mem0_sample_dump.json"


def print_step(number: int, title: str) -> None:
    print(f"\n--- Step {number}: {title} ---")


def load_sample_dump() -> dict[str, Any]:
    return json.loads(SAMPLE_DUMP_PATH.read_text(encoding="utf-8"))


def install_sample_mocks() -> None:
    if not USE_SAMPLE_MOCKS:
        return

    from cognee.infrastructure.llm.LLMGateway import LLMGateway
    from cognee.shared.data_models import Edge, KnowledgeGraph, Node, SummarizedContent

    @staticmethod
    async def sample_structured_output(text_input, system_prompt, response_model, **kwargs):
        if response_model is KnowledgeGraph or (isinstance(response_model, type) and issubclass(response_model, KnowledgeGraph)):
            if "Blue Bottle" not in text_input:
                return KnowledgeGraph(nodes=[], edges=[])
            return KnowledgeGraph(
                nodes=[
                    Node(id="Ava", name="Ava", type="Person", description="Ava is a person with imported memories."),
                    Node(id="Blue Bottle", name="Blue Bottle", type="CoffeeShop", description="Preferred coffee shop."),
                ],
                edges=[
                    Edge(source_node_id="Ava", target_node_id="Blue Bottle", relationship_name="prefers_coffee_from")
                ],
            )

        if response_model is SummarizedContent or (isinstance(response_model, type) and issubclass(response_model, SummarizedContent)):
            return SummarizedContent(summary=text_input, description="")

        return response_model()

    LLMGateway.acreate_structured_output = sample_structured_output

    # Clear explicit cache pools to apply configurations safely
    embedding_module = importlib.import_module("cognee.infrastructure.databases.vector.embeddings.get_embedding_engine")
    vector_module = importlib.import_module("cognee.infrastructure.databases.vector.create_vector_engine")

    embedding_module.create_embedding_engine.cache_clear()
    vector_module._create_vector_engine.cache_clear()


def explain_import_modes() -> None:
    print(
        """
Mem0Source accepts mem0 exports as lists, {"results": [...]},
{"memories": [...]}, or pre-fetched API response dictionaries.
- preserve: import source memories as COGXMemory records without re-running Cognee extraction.
- re-derive: ingest raw memory text and let Cognee extract its own graph representation.
- hybrid: preserve source graph records when available and cognify raw text.
""".strip()
    )


async def close_cached_engines() -> None:
    from cognee.infrastructure.databases.cache.get_cache_engine import close_cache_engine
    from cognee.infrastructure.databases.graph.get_graph_engine import _create_graph_engine
    from cognee.infrastructure.databases.vector.create_vector_engine import _create_vector_engine

    await close_cache_engine()
    _create_graph_engine.cache_clear()
    _create_vector_engine.cache_clear()
    await asyncio.sleep(0.25)


async def main() -> None:
    import cognee
    from cognee import SearchType
    from cognee.migration import Mem0Source

    print("=== Migrate from mem0 to Cognee ===")
    install_sample_mocks()

    print_step(1, "Start from a clean Cognee workspace")
    await cognee.forget(everything=True)
    print("Cleared existing data.")

    print_step(2, "Load a short mem0 dump")
    mem0_dump = load_sample_dump()
    memories = mem0_dump["results"]
    if not memories:
        raise ValueError("Sample mem0 dump contains no memories.")

    print(f"Loaded {len(memories)} mem0 memories.")
    print(f"First memory: {memories[0]['memory']}")

    print_step(3, "Choose the migration mode")
    explain_import_modes()
    print(f"\nThis run uses mode={MODE!r}.")

    print_step(4, "Import mem0 memories with Mem0Source")
    import_result = await cognee.remember(
        Mem0Source(mem0_dump, mode=MODE),
        dataset_name=DATASET_NAME,
        self_improvement=False,
    )
    print(import_result)

    print("\nProcessing imported data into the vector graph database...")
    await cognee.cognify()

    print_step(5, "Recall a migrated memory")
    query = "What coffee does Ava prefer before architecture reviews?"
    results = await cognee.recall(
        query,
        query_type=SearchType.CHUNKS_LEXICAL,
        datasets=[DATASET_NAME],
        top_k=3,
        auto_route=False,
    )
    
    print(f"Query: {query}")
    for index, item in enumerate(results, start=1):
        print(f"Result {index}: {item}")
        
    results_str = "\n".join(str(item) for item in results)
    if "Blue Bottle" not in results_str:
        raise RuntimeError("Recall did not return the migrated mem0 content.")

    print_step(6, "COGX note")
    print("Mem0Source translates each mem0 memory into a COGXMemory record in the COGX standard.")
    print("Reference: https://github.com/topoteretes/cognee/blob/dev/cognee/modules/migration/cogx.py")
    
    print("\nDone.")
    await close_cached_engines()


if __name__ == "__main__":
    asyncio.run(main())