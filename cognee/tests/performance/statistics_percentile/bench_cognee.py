"""
Standalone Cognee ingest benchmark.

Inserts text memories into Cognee (add + cognify) and measures wall-clock
time for each phase.

Usage:
    python bench_cognee.py                     # default settings
    python bench_cognee.py --memories data.json # custom memories file
    python bench_cognee.py --llm-model gpt-4o  # override LLM model

The memories file should be a JSON array of objects with "title" and "content"
keys (see the bundled memories.json).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from dotenv import dotenv_values

# ── Defaults ─────────────────────────────────────────────────────────────────

DEFAULT_MEMORIES_FILE = Path(__file__).with_name("memories.json")
DEFAULT_MOCK_MEMORIES_FILE = Path(__file__).with_name("mock_memories.json")
DEFAULT_LLM_PROVIDER = "openai"
DEFAULT_LLM_MODEL = "gpt-4.1-mini"
DEFAULT_EMBEDDING_PROVIDER = "openai"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_EMBEDDING_DIMS = 1536
DATASET_NAME = "bench_memories"

ENV_FILE = Path(__file__).resolve().parents[4] / ".env"

os.environ.setdefault("COGNEE_SKIP_CONNECTION_TEST", "true")


def _resolve_config(args: argparse.Namespace) -> dict:
    """Resolve config values: CLI arg → .env file → script defaults."""
    mock_llm = getattr(args, "mock_llm", False)
    env = dotenv_values(ENV_FILE) if ENV_FILE.exists() else {}

    def pick(cli_val, env_key: str, default):
        if cli_val is not None:
            return cli_val
        env_val = env.get(env_key) or os.environ.get(env_key)
        if env_val is not None:
            return type(default)(env_val) if not isinstance(default, str) else env_val
        return default

    api_key = pick(None, "LLM_API_KEY", "") or pick(None, "OPENAI_API_KEY", "")
    if not api_key and not mock_llm:
        sys.exit("Error: LLM_API_KEY is not set (CLI, .env, or environment)")

    # Embeddings may use a different provider/key than the LLM (e.g.
    # + OpenAI embeddings). Resolve the embedding key independently, falling back
    # to the LLM/OpenAI key when LLM and embeddings share a provider.
    embedding_api_key = pick(None, "EMBEDDING_API_KEY", "") or api_key

    return {
        "api_key": api_key or "mock-key",
        "embedding_api_key": embedding_api_key or "mock-key",
        "llm_provider": pick(args.llm_provider, "LLM_PROVIDER", DEFAULT_LLM_PROVIDER),
        "llm_model": pick(args.llm_model, "LLM_MODEL", DEFAULT_LLM_MODEL),
        "embedding_provider": pick(
            args.embedding_provider, "EMBEDDING_PROVIDER", DEFAULT_EMBEDDING_PROVIDER
        ),
        "embedding_model": pick(args.embedding_model, "EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        "embedding_dims": pick(args.embedding_dims, "EMBEDDING_DIMENSIONS", DEFAULT_EMBEDDING_DIMS),
        "mock_llm": mock_llm,
    }


# ── Mock LLM / Embedding ────────────────────────────────────────────────────


def _load_mock_data(path: Path) -> dict:
    with open(path) as f:
        raw = json.load(f)
    by_title: dict[str, dict] = {}
    for entry in raw["memories"]:
        by_title[entry["title"]] = entry
    return by_title


def _install_mocks(mock_data: dict[str, dict]) -> None:
    """Mock the LLM (structured-output replay) and embeddings (cognee MOCK_EMBEDDING)."""
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

    # Mock embeddings via cognee's built-in MOCK_EMBEDDING switch instead of
    # monkey-patching the engine. The real embedding engine is still constructed,
    # so it keeps its real tokenizer — chunk boundaries are decided by
    # embedding_engine.tokenizer.count_tokens() in chunk_by_sentence, and a stub
    # without a tokenizer would silently re-chunk the text (one-token-per-word),
    # shifting boundaries and breaking title-substring matching for multi-chunk
    # documents. With the flag set, embed_text skips the API and returns zero
    # vectors. Clear cached engines so the flag takes effect.
    os.environ["MOCK_EMBEDDING"] = "true"
    emb_mod.create_embedding_engine.cache_clear()
    vec_mod._create_vector_engine.cache_clear()


# ── Helpers ──────────────────────────────────────────────────────────────────


def load_memories(path: Path) -> list[dict]:
    with open(path) as f:
        memories = json.load(f)
    if not isinstance(memories, list) or not memories:
        sys.exit(f"Error: {path} must contain a non-empty JSON array")
    for i, m in enumerate(memories):
        if "content" not in m:
            sys.exit(f"Error: memory {i} is missing a 'content' key")
    return memories


def memory_to_text(mem: dict) -> str:
    title = mem.get("title", "Untitled")
    content = mem["content"]
    refs = mem.get("references", "none")
    if isinstance(refs, list):
        refs = ", ".join(refs) if refs else "none"
    return f"Title: {title}\n\n{content}\n\nReferences: {refs}"


# ── Benchmark ────────────────────────────────────────────────────────────────


async def run_benchmark(
    memories: list[dict],
    *,
    config: dict,
) -> dict:
    import cognee

    # Register community adapters before any engine is created. Comma-separated
    # module names; a module-level register() is called if present (some
    # adapters register on import alone).
    import importlib

    for module_name in filter(None, os.environ.get("COGNEE_REGISTER_ADAPTERS", "").split(",")):
        module = importlib.import_module(module_name)
        register = getattr(module, "register", None)
        if callable(register):
            register()
        print(f"Registered adapter module: {module_name}")

    llm_model = config["llm_model"]
    llm_provider = config["llm_provider"]
    embedding_model = config["embedding_model"]
    embedding_dims = config["embedding_dims"]

    cognee.config.set_llm_api_key(config["api_key"])
    cognee.config.set_llm_provider(llm_provider)
    cognee.config.set_llm_model(llm_model)
    cognee.config.set_embedding_provider(config["embedding_provider"])
    cognee.config.set_embedding_model(embedding_model)
    cognee.config.set_embedding_dimensions(embedding_dims)
    cognee.config.set_embedding_api_key(config["embedding_api_key"])

    if config.get("mock_llm"):
        mock_data = _load_mock_data(config["mock_memories_file"])
        _install_mocks(mock_data)
        print("Mock LLM/embedding mode enabled")

    n = len(memories)
    status = {
        "prune": "success",
        "db_setup": "success",
        "add": "success",
        "cognify": "success",
        "search": "success",
    }
    t_prune = 0.0
    t_db_setup = 0.0
    t_add = 0.0
    t_cognify = 0.0
    t_search = 0.0

    # ── Prune (clean slate) ──────────────────────────────────────────────
    print("Pruning previous data...")
    try:
        t_prune_start = time.time()
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
        t_prune = time.time() - t_prune_start
        print(f"  Prune completed in {t_prune:.2f}s")
    except Exception as e:
        t_prune = time.time() - t_prune_start
        status["prune"] = f"failed: {e}"
        print(f"  Prune FAILED: {e}")

    # ── DB Setup ─────────────────────────────────────────────────────────
    try:
        from cognee.modules.engine.operations.setup import setup

        t_db_setup_start = time.time()
        await setup()
        t_db_setup = time.time() - t_db_setup_start
    except Exception as e:
        t_db_setup = time.time() - t_db_setup_start
        status["db_setup"] = f"failed: {e}"
        print(f"  DB setup FAILED: {e}")

    # ── Phase 1: cognee.add() ────────────────────────────────────────────
    print(f"\nPhase 1: Adding {n} memories via cognee.add()...")
    text_list = [memory_to_text(mem) for mem in memories]

    try:
        t_add_start = time.time()
        await cognee.add(text_list, dataset_name=DATASET_NAME)
        t_add = time.time() - t_add_start
    except Exception as e:
        t_add = time.time() - t_add_start
        status["add"] = f"failed: {e}"
        print(f"  Add FAILED: {e}")

    # ── Phase 2: cognee.cognify() ────────────────────────────────────────
    print("\nPhase 2: Running cognee.cognify() (knowledge graph build)...")
    try:
        t_cognify_start = time.time()
        await cognee.cognify(data_per_batch=n, chunks_per_batch=10000)
        t_cognify = time.time() - t_cognify_start
    except Exception as e:
        t_cognify = time.time() - t_cognify_start
        status["cognify"] = f"failed: {e}"
        print(f"  Cognify FAILED: {e}")

    t_total = t_add + t_cognify

    # ── Phase 3: cognee.search() ─────────────────────────────────────────
    print("\nPhase 3: Running search queries...")
    try:
        t_q_start = time.time()
        await cognee.search(query_text="What is in the document", only_context=True)
        t_search = time.time() - t_q_start
    except Exception as e:
        t_search = time.time() - t_q_start
        status["search"] = f"failed: {e}"
        print(f"  Search FAILED: {e}")

    all_ok = all(v == "success" for v in status.values())

    # ── Report ───────────────────────────────────────────────────────────
    results = {
        "memories_count": n,
        "add_time_s": round(t_add, 3),
        "cognify_time_s": round(t_cognify, 3),
        "total_ingest_time_s": round(t_total, 3),
        "prune_time_s": round(t_prune, 3),
        "db_setup_time_s": round(t_db_setup, 3),
        "search_time": t_search,
        "status": status,
        "success": all_ok,
        "config": {
            "llm_model": llm_model,
            "embedding_model": embedding_model,
            "embedding_dimensions": embedding_dims,
            "dataset_name": DATASET_NAME,
            "mock_llm": config.get("mock_llm", False),
        },
    }

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Memories inserted : {n}")
    print(f"  cognee.add() time : {t_add:.2f}s  ({t_add / n:.2f}s per memory)  [{status['add']}]")
    print(f"  cognify() time    : {t_cognify:.2f}s  [{status['cognify']}]")
    print(f"  Total ingest time : {t_total:.2f}s  ({t_total / n:.2f}s per memory)")
    print(f"  Search total      : {t_search:.2f}s  [{status['search']}]")
    print(f"  DB setup time     : {t_db_setup:.2f}s  [{status['db_setup']}]")
    print(f"  Prune time        : {t_prune:.2f}s  [{status['prune']}]")
    print(f"  LLM model         : {llm_model}")
    print(f"  Embedding model   : {embedding_model} ({embedding_dims}d)")
    if config.get("mock_llm"):
        print("  Mock mode         : ON")
    print(f"  Overall           : {'ALL OK' if all_ok else 'SOME FAILURES'}")
    print("=" * 60)

    return results


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Cognee memory ingestion (add + cognify).",
    )
    parser.add_argument(
        "--memories",
        type=Path,
        default=DEFAULT_MEMORIES_FILE,
        help=f"JSON file with memories array (default: {DEFAULT_MEMORIES_FILE.name})",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help=f"LLM model (default: .env LLM_MODEL or {DEFAULT_LLM_MODEL})",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help=f"Embedding model (default: .env EMBEDDING_MODEL or {DEFAULT_EMBEDDING_MODEL})",
    )
    parser.add_argument(
        "--llm-provider",
        default=None,
        help=f"LLM provider (default: .env LLM_PROVIDER or {DEFAULT_LLM_PROVIDER})",
    )
    parser.add_argument(
        "--embedding-provider",
        default=None,
        help=f"Embedding provider (default: .env EMBEDDING_PROVIDER or {DEFAULT_EMBEDDING_PROVIDER})",
    )
    parser.add_argument(
        "--embedding-dims",
        type=int,
        default=None,
        help=f"Embedding dimensions (default: .env EMBEDDING_DIMENSIONS or {DEFAULT_EMBEDDING_DIMS})",
    )
    parser.add_argument(
        "--num-memories",
        type=int,
        default=None,
        help="Limit the number of memories to load (default: all)",
    )
    parser.add_argument(
        "--mock-llm",
        action="store_true",
        default=False,
        help="Use mock LLM/embedding responses from mock_memories.json instead of real API calls",
    )
    parser.add_argument(
        "--mock-memories",
        type=Path,
        default=DEFAULT_MOCK_MEMORIES_FILE,
        help=f"Mock responses JSON file (default: {DEFAULT_MOCK_MEMORIES_FILE.name})",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Write JSON results to this file",
    )
    args = parser.parse_args()

    config = _resolve_config(args)
    if config["mock_llm"]:
        config["mock_memories_file"] = args.mock_memories

    memories = load_memories(args.memories)
    if args.num_memories is not None:
        memories = memories[: args.num_memories]
    print(f"Loaded {len(memories)} memories from {args.memories}")
    mock_label = " [MOCK]" if config["mock_llm"] else ""
    print(
        f"Config: llm={config['llm_model']}, embeddings={config['embedding_model']} ({config['embedding_dims']}d){mock_label}\n"
    )

    results = asyncio.run(run_benchmark(memories, config=config))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
