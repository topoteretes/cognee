"""Example: run cognee with any OpenAI-compatible LLM endpoint + local embeddings.

Most examples assume an OpenAI key. This one shows how to run the full
`add -> cognify -> search` pipeline against **any OpenAI-compatible endpoint**
(NVIDIA API catalog, Together, Groq, vLLM, Ollama, LM Studio, ...) while running
embeddings **locally** with fastembed — so no OpenAI key is required.

Two things this makes explicit, which otherwise take a source dive to find:

1. Point the LLM at a custom base URL with `cognee.config.set_llm_endpoint(...)`
   and provider `"custom"`.
2. Providers like Anthropic / NVIDIA do not expose an embeddings API, so pair the
   hosted LLM with a local fastembed embedding model. `pip install fastembed`.

Runs fully self-contained on the default local stack (Ladybug graph, LanceDB
vector, SQLite relational) in an isolated temp directory, so it does not touch
your configured databases.

Set before running:
    LLM_MODEL     e.g. openai/z-ai/glm-5.2   (the openai/ prefix routes via the
                  OpenAI-compatible client)
    LLM_ENDPOINT  e.g. https://integrate.api.nvidia.com/v1
    LLM_API_KEY   your endpoint's key

Tip: free-tier endpoints sometimes rate-limit the startup connection test.
Set COGNEE_SKIP_CONNECTION_TEST=true to bypass it (real calls still retry).
"""

import asyncio
import os
import tempfile
from pathlib import Path

_DATA_DIR = tempfile.mkdtemp(prefix="cognee_byo_llm_example_")
os.environ["ENABLE_BACKEND_ACCESS_CONTROL"] = "false"
os.environ["CACHING"] = "false"

import cognee  # noqa: E402
from cognee.modules.search.types import SearchType  # noqa: E402

# --- LLM: any OpenAI-compatible endpoint (config setters win over ambient .env) ---
cognee.config.set_llm_provider("custom")
cognee.config.set_llm_model(os.environ.get("LLM_MODEL", "openai/z-ai/glm-5.2"))
cognee.config.set_llm_endpoint(
    os.environ.get("LLM_ENDPOINT", "https://integrate.api.nvidia.com/v1")
)
cognee.config.set_llm_api_key(os.environ["LLM_API_KEY"])

# --- Embeddings: local, no API key (works with any LLM provider) ---
cognee.config.set_embedding_provider("fastembed")
cognee.config.set_embedding_model("sentence-transformers/all-MiniLM-L6-v2")
cognee.config.set_embedding_dimensions(384)

# --- Keep everything in an isolated local stack ---
cognee.config.set_graph_database_provider("kuzu")  # Ladybug (local, embedded)
cognee.config.set_vector_db_provider("lancedb")
cognee.config.data_root_directory(str(Path(_DATA_DIR) / "data"))
cognee.config.system_root_directory(str(Path(_DATA_DIR) / "system"))


SAMPLE_TEXT = """\
Cognee turns documents, code, and application data into persistent AI memory.
It builds a knowledge graph alongside a vector store, so agents can recall facts
and the relationships between them. The memory improves over time as new data is
added and feedback is applied.
"""


async def main() -> None:
    await cognee.add(SAMPLE_TEXT)
    await cognee.cognify()

    answer = await cognee.search(
        query_text="What does cognee build alongside the vector store?",
        query_type=SearchType.GRAPH_COMPLETION,
    )
    print("\nAnswer:")
    for item in answer:
        print(item)


if __name__ == "__main__":
    asyncio.run(main())
