"""Example: Running Cognee fully locally using Ollama.

Demonstrates local graph extraction and search using a recommended Ollama setup:
- LLM Provider: Ollama (Llama 3.1 8B)
- Embeddings: Ollama (nomic-embed-text)
- Local embedded database stack (Ladybug, LanceDB, SQLite)

Requires `ollama serve` running and the following models pulled locally:
- `ollama pull llama3.1:8b`
- `ollama pull nomic-embed-text`
"""

import os
import asyncio
import tempfile
from pathlib import Path

# Setup temp directory to keep this example self-contained
_DATA_DIR = tempfile.mkdtemp(prefix="cognee_ollama_example_")
os.environ["ENABLE_BACKEND_ACCESS_CONTROL"] = "false"
os.environ["CACHING"] = "false"

# Configure Ollama environment settings
os.environ["LLM_PROVIDER"] = "ollama"
os.environ["LLM_MODEL"] = "llama3.1:8b"
os.environ["LLM_ENDPOINT"] = "http://localhost:11434/v1"
os.environ["LLM_API_KEY"] = "ollama"
os.environ["LLM_TEMPERATURE"] = "0.0"

os.environ["EMBEDDING_PROVIDER"] = "ollama"
os.environ["EMBEDDING_MODEL"] = "nomic-embed-text"
os.environ["EMBEDDING_ENDPOINT"] = "http://localhost:11434/api/embed"
os.environ["EMBEDDING_DIMENSIONS"] = "768"

import cognee  # noqa: E402
from cognee.modules.search.types import SearchType  # noqa: E402
from cognee.infrastructure.llm.config import get_llm_config  # noqa: E402

# Force local embedded stack configuration
cognee.config.set_graph_database_provider("kuzu")
cognee.config.set_vector_db_provider("lancedb")
cognee.config.data_root_directory(str(Path(_DATA_DIR) / "data"))
cognee.config.system_root_directory(str(Path(_DATA_DIR) / "system"))


SAMPLE_TEXT = """\
Cognee is an open-source library that helps developers turn documents into AI memory.
It builds semantic graphs, indexes entities, and stores vectors to enable structured retrieval.
Cognee supports local execution via Ollama as well as hosted cloud providers.
"""


def banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


async def main() -> None:
    # Start from a clean slate in isolated directory
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    banner("LOCAL PIPELINE: ADD & COGNIFY USING OLLAMA")
    llm_config = get_llm_config()
    print(f"Using LLM: {llm_config.llm_model}")
    print(f"Using Embeddings: {os.environ.get('EMBEDDING_MODEL')}")

    # Add sample text to dataset
    await cognee.add(SAMPLE_TEXT, dataset_name="ollama_local_demo")

    # Process dataset (this will trigger warning if an unvalidated model is used)
    await cognee.cognify(datasets=["ollama_local_demo"])
    print("Local knowledge graph built successfully.")

    banner("LOCAL SEARCH")
    query = "What does Cognee help developers do?"
    results = await cognee.search(
        query_text=query,
        query_type=SearchType.GRAPH_COMPLETION,
        datasets=["ollama_local_demo"],
    )
    print(f"Query: {query}")
    print("Search Results:")
    print(results[0] if results else "<no results>")


if __name__ == "__main__":
    asyncio.run(main())
