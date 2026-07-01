"""Local-only Cognee example using Ollama (no cloud API needed).

Prerequisites
-------------
1. Install Ollama: https://ollama.com/download
2. Pull a supported model and embedding model::

       ollama pull llama3.1:8b
       ollama pull nomic-embed-text

3. Make sure Ollama is running (it starts automatically on most installs,
   or run ``ollama serve`` in a terminal).

4. Install Cognee with the Ollama extra::

       pip install "cognee[ollama]"

5. Set environment variables (or create a .env file)::

       LLM_PROVIDER=ollama
       LLM_MODEL=llama3.1:8b
       LLM_ENDPOINT=http://localhost:11434/v1
       LLM_API_KEY=ollama

       EMBEDDING_PROVIDER=ollama
       EMBEDDING_MODEL=nomic-embed-text:latest
       EMBEDDING_ENDPOINT=http://localhost:11434/api/embed
       HUGGINGFACE_TOKENIZER=nomic-ai/nomic-embed-text-v1.5

Run
---
    python examples/python/local_ollama_example.py

Troubleshooting
---------------
* "connection refused" → Ollama is not running; start it with ``ollama serve``.
* Extraction produces empty results → the model may not support JSON-schema mode.
  Try llama3.1:8b, hermes3, or qwen2.5:14b.  Avoid mistral:7b and llama2.
  See the full list: docs/ollama_models.md
* "model not found" → run ``ollama pull <model>`` first.
"""

from __future__ import annotations

import asyncio
import os

# ---------------------------------------------------------------------------
# Configure for local Ollama — fall back to env vars if already set
# ---------------------------------------------------------------------------

os.environ.setdefault("LLM_PROVIDER", "ollama")
os.environ.setdefault("LLM_MODEL", "llama3.1:8b")
os.environ.setdefault("LLM_ENDPOINT", "http://localhost:11434/v1")
os.environ.setdefault("LLM_API_KEY", "ollama")

os.environ.setdefault("EMBEDDING_PROVIDER", "ollama")
os.environ.setdefault("EMBEDDING_MODEL", "nomic-embed-text:latest")
os.environ.setdefault("EMBEDDING_ENDPOINT", "http://localhost:11434/api/embed")
os.environ.setdefault("HUGGINGFACE_TOKENIZER", "nomic-ai/nomic-embed-text-v1.5")

# Disable cloud connection test (Ollama is local)
os.environ.setdefault("COGNEE_SKIP_CONNECTION_TEST", "true")
os.environ.setdefault("ENABLE_BACKEND_ACCESS_CONTROL", "false")

import cognee  # noqa: E402  (import after env vars are set)
from cognee.infrastructure.llm.ollama_models import (  # noqa: E402
    SUPPORTED_MODELS,
    check_ollama_model,
)


TEXTS = [
    "The hippocampus is a region of the brain involved in the formation of new memories "
    "and is associated with learning and spatial navigation.",
    "Long-term potentiation (LTP) is a persistent strengthening of synapses based on "
    "recent patterns of activity. It is considered a major cellular mechanism for learning.",
    "Dopamine is a neurotransmitter that plays an important role in reward-motivated "
    "behaviour. It is released during pleasurable activities and reinforces behaviour.",
    "The prefrontal cortex is involved in planning, decision-making, and moderating "
    "social behaviour. It is one of the last brain regions to fully mature.",
]


async def main() -> None:
    model = os.environ["LLM_MODEL"]
    print(f"\n{'='*60}")
    print(f"  Cognee local Ollama example")
    print(f"  Model: {model}")
    print(f"{'='*60}\n")

    # Advisory check — warns but does not block
    check_ollama_model(model)

    print("Supported models for graph extraction:")
    for tag, note in list(SUPPORTED_MODELS.items())[:5]:
        print(f"  • {tag:30s}  {note}")
    print("  … (see docs/ollama_models.md for the full list)\n")

    # -----------------------------------------------------------------------
    # Store texts in Cognee memory
    # -----------------------------------------------------------------------
    print("Storing texts in memory…")
    for i, text in enumerate(TEXTS, 1):
        print(f"  [{i}/{len(TEXTS)}] {text[:60]}…")
        await cognee.remember(text, dataset_name="neuroscience_local")

    # -----------------------------------------------------------------------
    # Search
    # -----------------------------------------------------------------------
    query = "What role does dopamine play in the brain?"
    print(f"\nSearching: {query!r}\n")
    results = await cognee.recall(query_text=query)

    if results:
        print("Results:")
        for r in results[:3]:
            text = getattr(r, "text", None) or str(r)
            print(f"  • {text[:120]}")
    else:
        print("No results found. Check that cognify ran successfully.")

    # -----------------------------------------------------------------------
    # Clean up
    # -----------------------------------------------------------------------
    await cognee.forget(dataset="neuroscience_local")
    print("\nDone — local memory cleared.")


if __name__ == "__main__":
    asyncio.run(main())
