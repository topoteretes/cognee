"""Mock-replay ingestion: build a real cognee graph/vector store with zero
LLM or embedding API calls.

Extracted from the performance test system
(``cognee/tests/performance/statistics_percentile/bench_cognee.py``) so any
suite can build large deterministic systems the same way: the perf benchmarks,
and the large-scale migration release test, which seeds a 100k-node system on
a LEGACY cognee version and migrates it on the current branch.

How the replay works
--------------------
* ``install_mocks`` replaces ``LLMGateway.acreate_structured_output`` with a
  replay function: each chunk is matched against the mock entries' titles by
  substring, and the pre-captured ``KnowledgeGraph`` / ``SummarizedContent``
  for that entry is returned. Mock files are produced by ``capture_mock.py``
  (one real LLM run) and optionally multiplied by ``generate_large_mock.py``.
* Embeddings run through cognee's built-in ``MOCK_EMBEDDING`` switch: the real
  engine is constructed (its real tokenizer decides chunk boundaries, keeping
  the title-substring matching aligned with the capture), but ``embed_text``
  returns zero-vectors.

Everything else in the pipeline is real: chunking, graph writes, vector-store
writes, relational records. Only the AI calls are replayed.

VERSION PORTABILITY: this module is also executed against old cognee releases
(the release migration test seeds data on the pinned legacy tag), so every
cognee import stays inside the function that needs it and only touches
interfaces that exist since v1.2.0 (``LLMGateway``, ``shared.data_models``,
the embedding/vector engine caches, ``add``/``cognify``).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


def load_memories(path: Path) -> list[dict]:
    """Load a memories JSON array: objects with "title" and "content" keys."""
    with open(path) as f:
        memories = json.load(f)
    if not isinstance(memories, list) or not memories:
        sys.exit(f"Error: {path} must contain a non-empty JSON array")
    for i, m in enumerate(memories):
        if "content" not in m:
            sys.exit(f"Error: memory {i} is missing a 'content' key")
    return memories


def memory_to_text(mem: dict) -> str:
    """The exact document text a memory becomes — MUST stay in sync with the
    transform used at capture time, or chunk boundaries shift and the
    title-substring matching breaks for multi-chunk documents."""
    title = mem.get("title", "Untitled")
    content = mem["content"]
    refs = mem.get("references", "none")
    if isinstance(refs, list):
        refs = ", ".join(refs) if refs else "none"
    return f"Title: {title}\n\n{content}\n\nReferences: {refs}"


def load_mock_data(path: Path) -> dict:
    """Mock responses file ({"memories": [{title, knowledge_graph, summary}]})
    -> {title: entry} replay map."""
    with open(path) as f:
        raw = json.load(f)
    by_title: dict[str, dict] = {}
    for entry in raw["memories"]:
        by_title[entry["title"]] = entry
    return by_title


def install_mocks(mock_data: dict[str, dict], mock_embeddings: bool = True) -> None:
    """Mock the LLM (structured-output replay) and, by default, embeddings via
    cognee's built-in MOCK_EMBEDDING switch.

    Pass ``mock_embeddings=False`` when a document-embedding replay manages
    embeddings instead.
    """
    import importlib

    from cognee.infrastructure.llm.LLMGateway import LLMGateway
    from cognee.shared.data_models import KnowledgeGraph, SummarizedContent

    emb_mod = importlib.import_module(
        "cognee.infrastructure.databases.vector.embeddings.get_embedding_engine"
    )
    vec_mod = importlib.import_module("cognee.infrastructure.databases.vector.create_vector_engine")

    def _match_memory(text_input: str) -> dict | None:
        for title, entry in mock_data.items():
            if title in text_input:
                return entry
        return None

    @staticmethod
    async def _mock_acreate(text_input, system_prompt, response_model, **kwargs):
        entry = _match_memory(text_input)

        if response_model is KnowledgeGraph or (
            isinstance(response_model, type) and issubclass(response_model, KnowledgeGraph)
        ):
            if entry:
                return KnowledgeGraph(**entry["knowledge_graph"])
            return KnowledgeGraph(nodes=[], edges=[])

        if response_model is SummarizedContent or (
            isinstance(response_model, type) and issubclass(response_model, SummarizedContent)
        ):
            if entry:
                return SummarizedContent(**entry["summary"])
            return SummarizedContent(summary="Mock summary.", description="")

        return response_model()

    LLMGateway.acreate_structured_output = _mock_acreate

    if mock_embeddings:
        # cognee's built-in switch: the real embedding engine is still
        # constructed (its real tokenizer decides chunk boundaries), but
        # embed_text skips the API and returns zero vectors. Clear cached
        # engines so the flag takes effect.
        os.environ["MOCK_EMBEDDING"] = "true"
    emb_mod.create_embedding_engine.cache_clear()
    vec_mod._create_vector_engine.cache_clear()


async def ingest_mock(
    memories_path: Path,
    mock_path: Path,
    dataset_names: list[str],
    prune_first: bool = False,
) -> dict:
    """Ingest the mock corpus into each dataset with a REAL add + cognify.

    Installs the replay mocks, optionally prunes to a clean slate, then runs
    ``add`` + ``cognify`` per dataset sequentially. Returns per-dataset
    timings. All AI calls are replayed; the stored system is genuine for the
    running cognee version.
    """
    import cognee

    mock_data = load_mock_data(mock_path)
    install_mocks(mock_data)
    memories = load_memories(memories_path)
    texts = [memory_to_text(m) for m in memories]
    print(f"mock_ingestion: {len(memories)} memories, {len(mock_data)} mock entries", flush=True)

    if prune_first:
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)

    timings: dict = {}
    for dataset_name in dataset_names:
        t = time.time()
        await cognee.add(texts, dataset_name=dataset_name)
        add_s = time.time() - t
        t = time.time()
        await cognee.cognify(datasets=[dataset_name])
        cognify_s = time.time() - t
        timings[dataset_name] = {"add_s": round(add_s, 1), "cognify_s": round(cognify_s, 1)}
        print(
            f"mock_ingestion: [{dataset_name}] add={add_s:.1f}s cognify={cognify_s:.1f}s",
            flush=True,
        )

    return timings
