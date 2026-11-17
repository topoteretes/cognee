"""Reproducible pgvector write benchmark; uses a new schema and removes it per run.

Example: TEST_PGVECTOR_URL=postgresql+asyncpg://... python -m \
cognee.tests.performance.postgres_storage_benchmark --points 23109 --pairs 5 --output results.json
Requires an otherwise idle scratch database with vector and pg_stat_statements installed.
Never point this benchmark at a shared server: it resets statement statistics.
"""

import argparse
import asyncio
import json
from pathlib import Path
from time import perf_counter
from uuid import NAMESPACE_URL, uuid4, uuid5

import numpy as np
from sqlalchemy import text

from cognee.infrastructure.databases.unified.storage_trace import collect_storage_events
from cognee.infrastructure.databases.vector.embeddings.EmbeddingEngine import EmbeddingResult
from cognee.infrastructure.databases.vector.embeddings.EmbeddingGateway import EmbeddingGateway
from cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter import PGVectorAdapter
from cognee.infrastructure.engine import DataPoint


class Point(DataPoint):
    text: str
    metadata: dict = {"index_fields": ["text"]}


class Embeddings:
    model = "deterministic-local-benchmark"

    def __init__(self, dimensions, latency_ms):
        self.dimensions = dimensions
        self.latency = latency_ms / 1000
        self.requests = 0

    def get_batch_size(self):
        return 32

    def get_vector_size(self):
        return self.dimensions

    async def embed_text(self, texts):
        self.requests += 1
        if self.latency:
            await asyncio.sleep(self.latency)
        return EmbeddingResult(
            [
                np.random.default_rng(int(value)).standard_normal(self.dimensions, dtype=np.float32)
                for value in texts
            ],
            tokens=0,
        )


async def run(args, writers):
    engine = Embeddings(args.dimensions, args.latency_ms)
    adapter = PGVectorAdapter(args.url, None, EmbeddingGateway(engine))
    schema = "bench_" + uuid4().hex[:12]
    async with adapter.engine.begin() as connection:
        await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
    await adapter.close()
    adapter = PGVectorAdapter(args.url, None, EmbeddingGateway(engine), schema=schema)
    events = []
    points = [
        Point(id=uuid5(NAMESPACE_URL, f"sdk1470:{i}"), text=str(i)) for i in range(args.points)
    ]
    slots = asyncio.Semaphore(args.embedding_workers)
    gate = asyncio.Semaphore(writers) if writers else None

    async def write(batch):
        async with slots:
            await adapter.create_data_points(
                "Benchmark_text", batch, use_copy=bool(writers), write_gate=gate
            )

    async def ingest():
        await asyncio.gather(
            *(write(points[start : start + 128]) for start in range(0, len(points), 128))
        )

    try:
        await adapter.create_collection("Benchmark_text", payload_schema=Point)
        if args.reingest:
            await ingest()
            engine.requests = 0
        if args.statement_stats:
            async with adapter.engine.begin() as connection:
                await connection.execute(text("SELECT pg_stat_statements_reset()"))
        started = perf_counter()
        with collect_storage_events(events.append):
            await ingest()
        elapsed = perf_counter() - started
        async with adapter.engine.connect() as connection:
            count = (
                await connection.execute(text(f'SELECT count(*) FROM "{schema}"."Benchmark_text"'))
            ).scalar_one()
            assert count == len(points)
            stats = []
            if args.statement_stats:
                stats = [
                    dict(row._mapping)
                    for row in (
                        await connection.execute(
                            text(
                                "SELECT query, calls, total_exec_time, rows FROM pg_stat_statements "
                                "WHERE dbid=(SELECT oid FROM pg_database WHERE datname=current_database()) "
                                "AND query NOT LIKE '%pg_stat_statements%' ORDER BY total_exec_time DESC"
                            )
                        )
                    ).all()
                ]
        return {
            "writers": writers,
            "copy": bool(writers),
            "elapsed_s": elapsed,
            "points": count,
            "embedding_requests": engine.requests,
            "latency_ms": args.latency_ms,
            "reingest": args.reingest,
            "timeline": events,
            "statements": stats,
        }
    finally:
        await adapter.close()
        # A separate unpinned adapter cleans only this run's generated schema.
        cleanup = PGVectorAdapter(args.url, None, EmbeddingGateway(engine))
        async with cleanup.engine.begin() as connection:
            await connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        await cleanup.close()


async def main(args):
    results = []
    output = Path(args.output)
    for pair in range(args.pairs):
        variants = args.writers if pair % 2 == 0 else list(reversed(args.writers))
        for writers in variants:
            group_started = perf_counter()
            if args.concurrent_calls == 1:
                rows = [await run(args, writers)]
            else:
                if args.statement_stats:
                    raise ValueError(
                        "Disable statement_stats for concurrent calls to avoid overlapping resets"
                    )
                rows = await asyncio.gather(
                    *(run(args, writers) for _ in range(args.concurrent_calls))
                )
            group_elapsed = perf_counter() - group_started
            for row in rows:
                row["group_elapsed_s"] = group_elapsed
                row["pair"] = pair
                row["concurrent_calls"] = args.concurrent_calls
                results.append(row)
                print(
                    json.dumps(
                        {k: v for k, v in row.items() if k not in ("timeline", "statements")}
                    ),
                    flush=True,
                )
            output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    import os

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.getenv("TEST_PGVECTOR_URL"))
    parser.add_argument("--points", type=int, default=23109)
    parser.add_argument("--dimensions", type=int, default=3072)
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--writers", type=int, nargs="+", default=[0, 1, 2, 4])
    parser.add_argument("--latency-ms", type=float, default=0)
    parser.add_argument("--embedding-workers", type=int, default=31)
    parser.add_argument("--concurrent-calls", type=int, default=1)
    parser.add_argument("--statement-stats", action="store_true")
    parser.add_argument("--reingest", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not args.url:
        parser.error("Pass --url or TEST_PGVECTOR_URL")
    if (
        min(args.points, args.dimensions, args.pairs, args.embedding_workers, args.concurrent_calls)
        < 1
    ):
        parser.error(
            "Points, dimensions, pairs, embedding workers and concurrent calls must be positive"
        )
    if args.latency_ms < 0 or any(value < 0 for value in args.writers):
        parser.error("Latency and writer counts cannot be negative")
    asyncio.run(main(args))
