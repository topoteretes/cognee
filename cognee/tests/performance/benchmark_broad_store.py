"""Benchmark BROAD retrieval with synthetic graph data and no model or database services.

Run from the repository root:
    uv run python cognee/tests/performance/benchmark_broad_store.py --http-probe

To compare a previous file-backed store, save its broad_store.py outside the checkout
and pass --store-source /path/to/broad_store.py. Both runs use a 512 MB coverage threshold;
the previous store's 32 MB default would otherwise benchmark a different retrieval route.

The mock graph generates exactly the requested bytes of ASCII CSV, lazily per graph
batch. The vector adapter rejects every call. A fixed SQL decision replaces the LLM;
real document parsing, schema description, SQLite execution and cleanup are timed.
This measures retrieval, not ingestion, embedding, network latency or LLM answering.
"""

import argparse
import asyncio
import importlib.util
import json
import os
import platform
import resource
import socket
import sqlite3
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import UUID

os.environ.setdefault("TELEMETRY_DISABLED", "1")
os.environ.setdefault("ENV", "dev")

from cognee.modules.retrieval import broad_retriever
from cognee.modules.retrieval.broad_retriever import BroadRetriever, DecisionLarge

HEADER = "id,status,amount,payload\n"
SQL = "SELECT COUNT(*), SUM(amount) FROM \"records_0000\" WHERE status = 'open'"


class MockGraph:
    """Generate each document once; hold only the batch requested by BROAD."""

    def __init__(self, dataset_bytes: int, documents: int):
        size, remainder = divmod(dataset_bytes, documents)
        self.sizes = {str(UUID(int=i + 1)): size + (i < remainder) for i in range(documents)}
        self.generated_bytes = self.rows = self.open_rows = self.open_sum = 0
        self.generation_seconds = 0.0
        self.visited = set()

    async def get_filtered_graph_data(self, filters):
        await asyncio.sleep(0)
        if filters[0]["type"] == ["SchemaTable"]:
            return [], []
        return [
            (document, {"type": "TextDocument", "name": f"records-{i:04d}.csv"})
            for i, document in enumerate(self.sizes)
        ], []

    def document(self, size: int) -> str:
        started = time.perf_counter()
        remaining = size - len(HEADER)
        if remaining < 32:
            raise ValueError("Each benchmark document must fit its header and one CSV row")
        lines = [HEADER]
        while remaining:
            width = min(1000, remaining)
            if 0 < remaining - width < 32:
                width = remaining
            index = self.rows
            status = "open" if index % 2 == 0 else "done"
            amount = index % 10
            prefix = f"{index:010d},{status},{amount},"
            lines.append(prefix + "x" * (width - len(prefix) - 1) + "\n")
            self.rows += 1
            if status == "open":
                self.open_rows += 1
                self.open_sum += amount
            remaining -= width
        text = "".join(lines)
        assert len(text) == size
        self.generated_bytes += size
        self.generation_seconds += time.perf_counter() - started
        return text

    async def get_neighborhood(self, documents, depth=1):
        await asyncio.sleep(0)
        nodes, edges = [], []
        for document in documents:
            assert document not in self.visited, "A document was loaded more than once"
            self.visited.add(document)
            chunk = f"chunk-{document}"
            nodes.append(
                (
                    chunk,
                    {
                        "type": "DocumentChunk",
                        "chunk_index": 0,
                        "text": await asyncio.to_thread(self.document, self.sizes[document]),
                    },
                )
            )
            edges.append((chunk, document, "is_part_of", {}))
        print(f"Fetched {self.generated_bytes:,} source bytes", flush=True)
        return nodes, edges


class MockVector:
    async def search(self, *args, **kwargs):
        raise AssertionError("The complete dataset must be loaded; vector sampling is forbidden")


@asynccontextmanager
async def http_probe(enabled):
    """Measure real localhost HTTP responses while retrieval uses the same event loop."""
    metrics = {}
    if not enabled:
        yield metrics
        return

    import httpx
    import uvicorn
    from fastapi import FastAPI

    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"ok": True}

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    config = uvicorn.Config(app, log_level="error", access_log=False, lifespan="off")
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve(sockets=[sock]))
    samples, errors = [], []
    running = True

    async def probe():
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=10) as client:
            while running:
                started = time.perf_counter()
                try:
                    response = await client.get("/health")
                    assert response.status_code == 200 and response.json() == {"ok": True}
                except httpx.HTTPError as error:
                    errors.append(type(error).__name__)
                else:
                    if running:
                        samples.append(time.perf_counter() - started)
                await asyncio.sleep(0.05)

    try:
        while not server.started:
            if server_task.done():
                await server_task
                raise RuntimeError("HTTP probe server stopped before startup")
            await asyncio.sleep(0.01)
        task = asyncio.create_task(probe())
        try:
            yield metrics
        finally:
            running = False
            await task
    finally:
        server.should_exit = True
        await server_task
        sock.close()
    ordered = sorted(samples)
    metrics.update(
        responses_during_retrieval=len(samples),
        errors=errors,
        max_seconds=max(samples) if samples else None,
        p95_seconds=ordered[int(0.95 * (len(ordered) - 1))] if ordered else None,
    )
    assert samples and not errors, metrics


def load_store(source: Path | None):
    if source is None:
        return broad_retriever.RecordStore
    spec = importlib.util.spec_from_file_location("benchmark_store", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.RecordStore


async def benchmark(args):
    graph = MockGraph(args.dataset_mb * 1_000_000, args.documents)
    engine = SimpleNamespace(graph=graph, vector=MockVector())
    timings = {"parse_insert": 0.0, "describe": 0.0, "sql": 0.0, "close": 0.0}
    storage = {}
    parent_store = load_store(args.store_source)

    class TimedStore(parent_store):
        def add_document(self, *values):
            started = time.perf_counter()
            super().add_document(*values)
            timings["parse_insert"] += time.perf_counter() - started

        def describe(self, question):
            started = time.perf_counter()
            result = super().describe(question)
            timings["describe"] += time.perf_counter() - started
            return result

        def run(self, sql):
            started = time.perf_counter()
            result = super().run(sql)
            timings["sql"] += time.perf_counter() - started
            return result

        def close(self):
            storage["database_files"] = self.connection.execute("PRAGMA database_list").fetchall()
            pages = self.connection.execute("PRAGMA page_count").fetchone()[0]
            page_size = self.connection.execute("PRAGMA page_size").fetchone()[0]
            storage["sqlite_bytes"] = pages * page_size
            started = time.perf_counter()
            super().close()
            timings["close"] += time.perf_counter() - started

    class FixedSQLRetriever(BroadRetriever):
        async def document_sizes(self, documents):
            return {document: graph.sizes[document] for document in documents}

        async def _ask(self, text_input, prompt, response_model):
            assert response_model is DecisionLarge
            return DecisionLarge(sql=SQL)

    async def unified():
        return engine

    def forbid_models(*args, **kwargs):
        raise AssertionError("This benchmark must never call an LLM")

    running = True
    largest_gap = 0.0

    async def heartbeat():
        nonlocal largest_gap
        previous = time.perf_counter()
        while running:
            await asyncio.sleep(0.01)
            now = time.perf_counter()
            largest_gap = max(largest_gap, now - previous)
            previous = now

    heartbeat_task = asyncio.create_task(heartbeat())
    await asyncio.sleep(0)
    # The baseline needs an explicit override to take the same complete-data path.
    config = {}
    if args.store_source or args.store_mb is not None:
        config["store_mb"] = (
            args.store_mb if args.store_mb is not None else broad_retriever.BROAD_STORE_MB
        )
    started, cpu_started = time.perf_counter(), time.process_time()
    try:
        with (
            patch.object(broad_retriever, "RecordStore", TimedStore),
            patch.object(broad_retriever, "get_unified_engine", unified),
            patch.object(
                broad_retriever.LLMGateway, "acreate_structured_output", side_effect=forbid_models
            ),
        ):
            async with http_probe(args.http_probe) as http_metrics:
                found = await FixedSQLRetriever(**config).get_retrieved_objects(
                    "How many open records are there, and what is their total amount?"
                )
        timings["retrieval_total"] = time.perf_counter() - started
        timings["cpu"] = time.process_time() - cpu_started
    finally:
        running = False
        await heartbeat_task

    assert graph.generated_bytes == args.dataset_mb * 1_000_000
    assert len(graph.visited) == args.documents
    assert not found.coverage, "The benchmark unexpectedly used partial coverage"
    assert found.rows == [(graph.open_rows, graph.open_sum)], found.rows
    timings["source_generation"] = graph.generation_seconds
    timings["max_event_loop_gap"] = largest_gap
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = {
        "store": "baseline" if args.store_source else "in_memory",
        "dataset_bytes": graph.generated_bytes,
        "documents": args.documents,
        "records": graph.rows,
        "result": found.rows,
        "coverage_complete": True,
        "model_calls": 0,
        "vector_searches": 0,
        "http_probe": http_metrics,
        "timings_seconds": {key: round(value, 4) for key, value in timings.items()},
        "peak_rss_bytes": peak_rss if sys.platform == "darwin" else peak_rss * 1024,
        **storage,
        "python": platform.python_version(),
        "sqlite": sqlite3.sqlite_version,
        "platform": platform.platform(),
    }
    encoded = json.dumps(result, indent=2)
    print(encoded, flush=True)
    if args.json_output:
        args.json_output.write_text(encoded + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-mb", type=int, default=512)
    parser.add_argument("--store-mb", type=float)
    parser.add_argument("--http-probe", action="store_true")
    parser.add_argument("--documents", type=int, default=100)
    parser.add_argument("--store-source", type=Path)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    if args.dataset_mb <= 0 or args.documents <= 0:
        parser.error("dataset-mb and documents must be positive")
    asyncio.run(benchmark(args))


if __name__ == "__main__":
    main()
