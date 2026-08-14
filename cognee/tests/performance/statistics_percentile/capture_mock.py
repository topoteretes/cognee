"""
Capture real LLM cognify outputs and store them as a mock-responses file.

Runs cognee.add() + cognee.cognify() ONCE with a real LLM, records every
KnowledgeGraph / SummarizedContent the LLM produces (keyed by the exact chunk
text the LLM saw), and writes a mock file in the mock_memories.json format.
bench_cognee.py --mock-llm then replays those outputs with zero API calls,
matching each chunk by substring.

Works from either an existing benchmark memories file or a raw text file:

    # Mock for an existing benchmark input (memories.json format)
    python capture_mock.py --memories memories.json --output mock_memories.json

    # Mock for a raw text document (also writes the paired memories file,
    # so input + mock stay in sync — the war_and_peace.json pattern)
    python capture_mock.py --text book.txt --title "War and Peace" \
        --memories-output war_and_peace.json --output mock_war_and_peace.json

    # Smoke-test on a slice before burning tokens on the full input
    python capture_mock.py --memories memories.json --num-memories 2 \
        --output /tmp/mock_probe.json

Embeddings can be captured too (--capture-embeddings): the run then embeds
for real and every (text -> vector) pair is stored in a SEPARATE file (large!),
enabling bench_cognee.py --mock-document-embeddings — replay of the real
document vectors with live query embedding at search time:

    python capture_mock.py --memories memories.json --output mock_memories.json \
        --capture-embeddings   # also writes mock_memories_embeddings.json

Details that keep the mock replayable:
  - The input passes through bench_cognee.memory_to_text() and cognify runs
    with the same batching as the benchmark, so chunk boundaries — and
    therefore the substring match keys — are identical in capture and replay.
  - Without --capture-embeddings, embeddings are mocked during capture
    (MOCK_EMBEDDING): they never affect the captured LLM outputs, but the
    real embedding engine keeps its real tokenizer, which DOES decide chunk
    boundaries.
  - A substring-collision check verifies no chunk would wrongly match another
    entry's response at replay time.

LLM configuration comes from the environment / repo .env exactly like a real
benchmark run (LLM_PROVIDER / LLM_MODEL / LLM_API_KEY / LLM_ENDPOINT ...).

WARNING: like bench_cognee.py, this prunes the configured local cognee
instance before ingesting.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("COGNEE_SKIP_CONNECTION_TEST", "true")

# Reuse the benchmark's own input transforms so chunking stays in sync.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_cognee import DATASET_NAME, load_memories, memory_to_text  # noqa: E402


def _build_memories(args: argparse.Namespace) -> list[dict]:
    if args.memories:
        memories = load_memories(args.memories)
    else:
        text = Path(args.text).read_text(encoding="utf-8", errors="replace")
        memories = [{"title": args.title, "content": text}]
        if args.memories_output:
            Path(args.memories_output).write_text(
                json.dumps(memories, ensure_ascii=False, indent=2)
            )
            print(f"Wrote paired input file: {args.memories_output} ({len(text):,} chars)")
    if args.num_memories is not None:
        memories = memories[: args.num_memories]
    return memories


def _check_collisions(titles: list[str]) -> int:
    """bench_cognee._match_memory returns the FIRST stored title that is a
    substring of the LLM input; make sure every chunk resolves to itself."""
    collisions = 0
    for i, chunk in enumerate(titles):
        for j, title in enumerate(titles):
            if title in chunk:
                if j != i:
                    collisions += 1
                    print(f"  COLLISION: chunk #{i} would wrongly match entry #{j}")
                break
    return collisions


def _install_embedding_recorder(store: dict[str, list[float]]) -> dict:
    """Record every (text -> vector) pair the real embedding engine produces.

    Patches the engine CLASS so every instance (engines are re-created when
    caches clear) records through the same store. Returns engine metadata for
    the embeddings file header.
    """
    from cognee.infrastructure.databases.vector.embeddings.get_embedding_engine import (
        get_embedding_engine,
    )

    engine = get_embedding_engine()
    engine_cls = type(engine)
    original = engine_cls.embed_text

    async def recording_embed(self, text):
        vectors = await original(self, text)
        for item, vector in zip(text, vectors):
            store[item] = vector
        return vectors

    engine_cls.embed_text = recording_embed
    return {
        "model": getattr(engine, "model", None),
        "dimensions": getattr(engine, "dimensions", None),
    }


async def run(args: argparse.Namespace) -> None:
    import cognee
    from cognee.infrastructure.llm.LLMGateway import LLMGateway
    from cognee.shared.data_models import KnowledgeGraph, SummarizedContent

    embedding_store: dict[str, list[float]] = {}
    embedding_meta: dict = {}
    if args.capture_embeddings:
        # Real embeddings, recorded per text for --mock-document-embeddings.
        embedding_meta = _install_embedding_recorder(embedding_store)
    else:
        # Embeddings never influence the captured LLM outputs — mock them so a
        # capture run only spends LLM tokens. The engine keeps its real
        # tokenizer, which is what decides chunk boundaries.
        os.environ["MOCK_EMBEDDING"] = "true"

    # litellm's native `baseten/` provider authenticates via BASETEN_API_KEY
    # rather than the api_key cognee passes through; mirror it (process-env
    # only) so such configs work out of the box.
    if os.environ.get("LLM_API_KEY") and not os.environ.get("BASETEN_API_KEY"):
        os.environ["BASETEN_API_KEY"] = os.environ["LLM_API_KEY"]

    memories = _build_memories(args)
    text_list = [memory_to_text(m) for m in memories]
    n = len(text_list)
    print(f"Capturing LLM outputs for {n} memory(ies) -> {args.output}")

    # ── Record every structured-output call, keyed by exact prompt text ──
    records: dict[str, dict] = {}
    order: list[str] = []
    original = LLMGateway.acreate_structured_output

    @staticmethod
    async def recording(text_input, system_prompt, response_model, **kwargs):
        result = await original(text_input, system_prompt, response_model, **kwargs)
        if isinstance(result, KnowledgeGraph):
            key = "knowledge_graph"
        elif isinstance(result, SummarizedContent):
            key = "summary"
        else:
            return result
        if text_input not in records:
            records[text_input] = {}
            order.append(text_input)
        records[text_input][key] = result.model_dump()
        return result

    LLMGateway.acreate_structured_output = recording

    # ── Run add + cognify exactly like bench_cognee.run_benchmark ────────
    print("Pruning previous data...")
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    print(f"Adding {n} document(s) via cognee.add()...")
    await cognee.add(text_list, dataset_name=args.dataset_name)

    print("Running cognee.cognify() with a REAL LLM (this calls the API)...")
    await cognee.cognify(data_per_batch=n, chunks_per_batch=10000)

    # ── Build the mock file ───────────────────────────────────────────────
    entries = []
    skipped = 0
    for chunk_text in order:
        record = records[chunk_text]
        if "knowledge_graph" not in record or "summary" not in record:
            skipped += 1
            continue
        entries.append(
            {
                "title": chunk_text,
                "knowledge_graph": record["knowledge_graph"],
                "summary": record["summary"],
            }
        )

    collisions = _check_collisions([e["title"] for e in entries])

    output = Path(args.output)
    output.write_text(json.dumps({"memories": entries}, ensure_ascii=False, indent=2))

    embeddings_line = "  Embeddings                    : not captured"
    if args.capture_embeddings:
        embeddings_output = args.embeddings_output or output.with_name(
            f"{output.stem}_embeddings.json"
        )
        embeddings_output.write_text(
            json.dumps({**embedding_meta, "vectors": embedding_store}, ensure_ascii=False)
        )
        embeddings_line = (
            f"  Embeddings captured           : {len(embedding_store)} texts -> "
            f"{embeddings_output} ({embeddings_output.stat().st_size:,} bytes)"
        )

    print("\n" + "=" * 60)
    print(f"  Chunks captured (KG + summary): {len(entries)}")
    print(f"  Incomplete chunks skipped     : {skipped}")
    print(f"  Substring-match collisions    : {collisions}")
    print(f"  Mock file written             : {output}")
    print(f"  Mock file size                : {output.stat().st_size:,} bytes")
    print(embeddings_line)
    print("=" * 60)
    if collisions:
        raise SystemExit("Collisions detected — mock would mis-match in mock mode.")
    if not entries:
        raise SystemExit("No chunks captured — nothing to replay; mock file is unusable.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture real cognify LLM outputs into a bench_cognee.py mock file.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--memories",
        type=Path,
        help="Benchmark input file (JSON array of {title, content}) to capture from",
    )
    source.add_argument(
        "--text",
        type=Path,
        help="Raw text file to capture from (wrapped as a single memory)",
    )
    parser.add_argument(
        "--title",
        default="Untitled",
        help="Memory title when using --text (default: %(default)s)",
    )
    parser.add_argument(
        "--memories-output",
        type=Path,
        default=None,
        help="With --text: also write the paired benchmark input file here",
    )
    parser.add_argument(
        "--num-memories",
        type=int,
        default=None,
        help="Limit the number of memories to capture (smoke tests)",
    )
    parser.add_argument(
        "--dataset-name",
        default=DATASET_NAME,
        help="Dataset name for the capture run (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Mock-responses JSON file to write",
    )
    parser.add_argument(
        "--capture-embeddings",
        action="store_true",
        default=False,
        help=(
            "Embed for real and store every (text -> vector) pair for "
            "bench_cognee.py --mock-document-embeddings. Needs a working "
            "embedding API key; the embeddings file can be LARGE."
        ),
    )
    parser.add_argument(
        "--embeddings-output",
        type=Path,
        default=None,
        help="Embeddings file path (default: <output stem>_embeddings.json)",
    )
    args = parser.parse_args()

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
