"""Benchmark: ingest a whole book (PDF/txt) through GLiNER-only cognify.

Usage:
    python benchmark_book.py /path/to/book.pdf [workers]

With workers > 0 (default 3), extraction is packed and sharded across that
many GLiNER worker processes, and storage overlaps extraction.

Reference numbers (M-series laptop, CPU only, War & Peace, 2,043-page PDF):
    add() 27s; cognify 1,290s; total ~22 min
    3,140 chunks -> 6,933 entities, 48,159 edges, 3,140 extractive summaries
    ~54% of cognify time = GLiNER, ~46% = OpenAI embeddings + graph writes
"""

import asyncio
import os
import pathlib
import sys
import time

BOOK = sys.argv[1] if len(sys.argv) > 1 else None
if not BOOK or not pathlib.Path(BOOK).is_file():
    sys.exit("usage: python benchmark_book.py /path/to/book.pdf")

DEMO_DIR = pathlib.Path(__file__).parent / ".cognee_gliner_benchmark"
os.environ.setdefault("DATA_ROOT_DIRECTORY", str(DEMO_DIR / "data"))
os.environ.setdefault("SYSTEM_ROOT_DIRECTORY", str(DEMO_DIR / "system"))
os.environ.setdefault("TELEMETRY_DISABLED", "1")
os.environ["COGNEE_SKIP_CONNECTION_TEST"] = "true"
os.environ["AUTO_FEEDBACK"] = "false"
# Allow reading the book from wherever it lives.
os.environ["COGNEE_ALLOWED_LOCAL_FILE_ROOTS"] = str(pathlib.Path(BOOK).parent)

_real_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY") or ""
os.environ["EMBEDDING_API_KEY"] = _real_key
os.environ["LLM_API_KEY"] = "sk-FAKE-proof-that-no-llm-is-called"
os.environ["OPENAI_API_KEY"] = "sk-FAKE-proof-that-no-llm-is-called"

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import cognee  # noqa: E402

from gliner_cognify import gliner_cognify  # noqa: E402

DATASET = "gliner_benchmark"
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 3


def stamp(label, t0, timings):
    dt = time.time() - t0
    timings.append((label, dt))
    print(f"[TIMING] {label}: {dt:.1f}s", flush=True)
    return time.time()


async def main():
    timings = []
    t = time.time()

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    t = stamp("prune", t, timings)

    await cognee.add(BOOK, dataset_name=DATASET)
    t = stamp("add() - ingestion", t, timings)

    await gliner_cognify(
        datasets=[DATASET],
        workers=WORKERS,
        chunk_size=512,
        chunks_per_batch=256,
        gliner_batch_size=32,
    )
    t = stamp("gliner_cognify() - full pipeline", t, timings)

    from cognee.infrastructure.databases.graph import get_graph_engine

    graph_engine = await get_graph_engine()
    nodes, edges = await graph_engine.get_graph_data()
    type_counts = {}
    for _, props in nodes:
        node_type = props.get("type", "?")
        type_counts[node_type] = type_counts.get(node_type, 0) + 1
    t = stamp("graph stats read", t, timings)

    print(f"\n[RESULT] nodes={len(nodes)} edges={len(edges)}", flush=True)
    print(f"[RESULT] node types: {type_counts}", flush=True)
    total = sum(dt for _, dt in timings)
    print(f"\n[TIMING SUMMARY] total={total:.1f}s", flush=True)
    for label, dt in timings:
        print(f"  {label}: {dt:.1f}s", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
