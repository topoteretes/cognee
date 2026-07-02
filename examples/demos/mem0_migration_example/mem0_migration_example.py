import asyncio
import os

# This example imports memories from a mem0 export using the local-first stack:
# Ollama for embeddings (free, runs locally) and a free OpenRouter model for
# LLM calls. Swap these for your own provider if you have one configured.
os.environ.setdefault("EMBEDDING_PROVIDER", "ollama")
os.environ.setdefault("EMBEDDING_MODEL", "nomic-embed-text")  # bare model name, no "ollama/" prefix
os.environ.setdefault("EMBEDDING_ENDPOINT", "http://localhost:11434/api/embed")
os.environ.setdefault("EMBEDDING_DIMENSIONS", "768")
os.environ.setdefault("HUGGINGFACE_TOKENIZER", "nomic-ai/nomic-embed-text-v1.5")

import cognee
from cognee.modules.migration.sources.mem0 import Mem0Source

# Shape matches both the mem0 platform export API and the OSS get_all() call.
# Swap this for your own export, or use mem0's client directly:
#   from mem0 import Memory
#   raw = Memory().get_all(user_id="your_user_id")
sample_export = {
    "results": [
        {
            "id": "1",
            "memory": "User prefers dark mode",
            "user_id": "test_user",
            "categories": ["preferences"],
        },
        {
            "id": "2",
            "memory": "User is learning Python",
            "user_id": "test_user",
            "categories": ["skills"],
        },
    ]
}


async def main():
    source = Mem0Source(data=sample_export, mode="preserve")
    await cognee.remember(source)

    # preserve mode stores the raw memories but does not build the graph;
    # cognify() is required before recall() can answer questions about them.
    await cognee.cognify()

    results = await cognee.recall("what does the user prefer?")
    for r in results:
        print(r)


if __name__ == "__main__":
    asyncio.run(main())