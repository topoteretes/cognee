"""Embedding concurrency benchmark.

Sends one fixed synthetic corpus through cognee's real indexing path
(``index_data_points`` -> ``EmbeddingGateway`` -> engine -> dispatch seam) once
per ``EMBEDDING_MAX_CONCURRENT_REQUESTS`` value, against a real embedding
provider, and reports throughput, request latency and overload signals for each
value. Nothing is stored: the vector store is a stub that only embeds.

Run by hand, for example against a local TEI server:

    python cognee/tests/performance/embedding_concurrency_benchmark.py \\
        --provider openai_compatible --endpoint http://localhost:8080/v1 \\
        --model BAAI/bge-small-en-v1.5 --dimensions 384 --caps 4,16,32,64,128,256
"""

import argparse
import asyncio
import json
import os
import random
import resource
import subprocess
import threading
import time
from collections import Counter

import litellm
import psutil

import cognee.infrastructure.databases.vector.embeddings.config as embedding_config_module
import cognee.infrastructure.databases.vector.embeddings.LiteLLMEmbeddingEngine as litellm_engine_module
from cognee.infrastructure.databases.vector.embeddings.config import get_embedding_config
from cognee.infrastructure.databases.vector.embeddings.EmbeddingGateway import EmbeddingGateway
from cognee.infrastructure.databases.vector.embeddings.get_embedding_engine import (
    get_embedding_engine,
)
from cognee.infrastructure.engine import DataPoint
from cognee.shared import rate_limiting
from cognee.shared.overload_policy import overload_evidence
from cognee.tasks.storage.index_data_points import index_data_points


async def main(args) -> list[dict]:
    texts = synthetic_corpus(args.chunks, args.words_per_chunk, args.seed)
    results = []
    for cap in args.caps:
        configure_run(args, cap)
        results.append(await run_once(cap, texts))
        print_row(results[-1])
    write_results(args.out, args, results)
    return results


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--provider", required=True, help="EMBEDDING_PROVIDER, e.g. openai")
    parser.add_argument("--model", required=True, help="EMBEDDING_MODEL")
    parser.add_argument("--endpoint", default=None, help="EMBEDDING_ENDPOINT")
    parser.add_argument("--dimensions", type=int, required=True, help="EMBEDDING_DIMENSIONS")
    parser.add_argument("--api-key", default=None, help="EMBEDDING_API_KEY (else LLM_API_KEY)")
    parser.add_argument("--caps", default="4,16,32,64,128,256", help="comma-separated caps")
    parser.add_argument("--chunks", type=int, default=20_000)
    parser.add_argument("--words-per-chunk", type=int, default=220)  # about 300 tokens
    parser.add_argument("--seed", type=int, default=1539)
    parser.add_argument("--out", required=True, help="JSON file for the results")
    args = parser.parse_args(argv)
    args.caps = [int(cap) for cap in args.caps.split(",")]
    return args


_VOCABULARY = (
    "memory", "graph", "entity", "relation", "document", "chunk", "vector", "search",
    "answer", "question", "dataset", "pipeline", "provider", "request", "latency",
    "throughput", "server", "model", "token", "batch", "index", "store", "retrieve",
    "embed", "cluster", "node", "edge", "source", "fact", "context",
)  # fmt: skip


def synthetic_corpus(chunks: int, words_per_chunk: int, seed: int) -> list[str]:
    """Deterministic, unique chunks: the same text every run, nothing a cache can reuse."""
    rng = random.Random(seed)
    return [
        f"chunk {index}: " + " ".join(rng.choices(_VOCABULARY, k=words_per_chunk))
        for index in range(chunks)
    ]


def configure_run(args, cap: int) -> None:
    """Point cognee at the provider with this cap, and start from a clean dispatch state."""
    settings = {
        "EMBEDDING_PROVIDER": args.provider,
        "EMBEDDING_MODEL": args.model,
        "EMBEDDING_DIMENSIONS": str(args.dimensions),
        "EMBEDDING_MAX_CONCURRENT_REQUESTS": str(cap),
        "EMBEDDING_RATE_LIMIT_ENABLED": "false",
        "MOCK_EMBEDDING": "false",
    }
    if args.endpoint:
        settings["EMBEDDING_ENDPOINT"] = args.endpoint
    if args.api_key:
        settings["EMBEDDING_API_KEY"] = args.api_key
    os.environ.update(settings)
    get_embedding_config.cache_clear()
    # Each run starts unthrottled, with fresh slots: no run inherits another's throttling.
    rate_limiting._concurrency_state.clear()
    rate_limiting._embedding_rate_limiter = None
    embedding_config_module.embedding_overload_policy._paced_until = 0.0


async def run_once(cap: int, texts: list[str]) -> dict:
    """Index ``texts`` once through cognee's real path; return what happened."""
    engine = TimedEngine(get_embedding_engine())
    failures = FailureRecorder(embedding_config_module.embedding_overload_policy)
    store = EmbedOnlyStore(EmbeddingGateway(engine))
    points = [BenchmarkPoint(text=text) for text in texts]

    probes = Probes()
    probes.start()
    started = time.monotonic()
    try:
        await index_data_points(points, vector_engine=store)
    finally:
        failures.detach()
        await probes.stop()
    elapsed = time.monotonic() - started
    return summarize(cap, len(texts), elapsed, engine.latencies, failures) | probes.report(elapsed)


def _at(ordered: list[float], share: float) -> float | None:
    """The value at ``share`` (0..1) of an ascending list; None when it is empty."""
    return ordered[min(len(ordered) - 1, int(share * len(ordered)))] if ordered else None


def _process_cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


class Probes:
    """Where the time goes in one run: process CPU, event-loop lag, provider calls, connections."""

    def start(self) -> None:
        self._cpu_started = _process_cpu_seconds()
        self.loop_lag = LoopLagProbe()
        self.calls = CallTimers()
        self.connections = ConnectionSampler()
        self.loop_lag.start()
        self.calls.attach()
        self.connections.start()

    async def stop(self) -> None:
        self._cpu_seconds = _process_cpu_seconds() - self._cpu_started
        await self.loop_lag.stop()
        self.calls.detach()
        self.connections.stop()

    def report(self, elapsed: float) -> dict:
        return {
            "cpu_seconds": round(self._cpu_seconds, 1),
            "cpu_share": round(self._cpu_seconds / elapsed, 2) if elapsed else None,
            **self.loop_lag.report(),
            **self.calls.report(),
            "max_https_connections": self.connections.peak,
        }


class LoopLagProbe:
    """How late a task that sleeps 50 ms wakes up: a busy event loop wakes it late."""

    INTERVAL = 0.05

    def start(self) -> None:
        self.lags: list[float] = []
        self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        while True:
            planned = time.monotonic() + self.INTERVAL
            await asyncio.sleep(self.INTERVAL)
            self.lags.append(time.monotonic() - planned)

    async def stop(self) -> None:
        self._task.cancel()
        await asyncio.gather(self._task, return_exceptions=True)

    def report(self) -> dict:
        ordered = sorted(self.lags) or [0.0]
        return {
            "loop_lag_p50_ms": round(_at(ordered, 0.50) * 1000, 1),
            "loop_lag_p95_ms": round(_at(ordered, 0.95) * 1000, 1),
            "loop_lag_max_ms": round(ordered[-1] * 1000, 1),
        }


class CallTimers:
    """Times litellm's provider call (network plus litellm's own response handling, no slot
    wait) and cognee's vector decoding, by wrapping the names the LiteLLM engine calls."""

    def attach(self) -> None:
        self.provider_calls: list[float] = []
        self.decode_seconds = 0.0
        self._aembedding = litellm.aembedding
        self._decode = litellm_engine_module.decode_embedding

        async def timed_aembedding(*args, **kwargs):
            started = time.monotonic()
            try:
                return await self._aembedding(*args, **kwargs)
            finally:
                self.provider_calls.append(time.monotonic() - started)

        def timed_decode(*args, **kwargs):
            started = time.monotonic()
            try:
                return self._decode(*args, **kwargs)
            finally:
                self.decode_seconds += time.monotonic() - started

        litellm.aembedding = timed_aembedding
        litellm_engine_module.decode_embedding = timed_decode

    def detach(self) -> None:
        litellm.aembedding = self._aembedding
        litellm_engine_module.decode_embedding = self._decode

    def report(self) -> dict:
        ordered = sorted(self.provider_calls)
        return {
            "provider_call_p50": _at(ordered, 0.50),
            "provider_call_p95": _at(ordered, 0.95),
            "decode_seconds": round(self.decode_seconds, 2),
        }


class ConnectionSampler:
    """Peak count of this process's established HTTPS connections, sampled off the loop."""

    INTERVAL = 0.5

    def start(self) -> None:
        self.peak = 0
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        process = psutil.Process()
        while not self._stopped.wait(self.INTERVAL):
            open_https = sum(
                1
                for connection in process.net_connections(kind="tcp")
                if connection.status == psutil.CONN_ESTABLISHED
                and connection.raddr
                and connection.raddr.port == 443
            )
            self.peak = max(self.peak, open_https)

    def stop(self) -> None:
        self._stopped.set()
        self._thread.join()


class TimedEngine:
    """The real engine, timing each ``embed_text`` call: one provider request, including the
    wait for a request slot, any pacing and the engine's retries."""

    def __init__(self, engine):
        self._engine = engine
        self.latencies: list[float] = []

    async def embed_text(self, text: list[str]):
        started = time.monotonic()
        try:
            return await self._engine.embed_text(text)
        finally:
            self.latencies.append(time.monotonic() - started)

    def __getattr__(self, name: str):
        return getattr(self._engine, name)


class FailureRecorder:
    """Counts every failed request attempt the seam reports, by cause; the policy still reacts."""

    def __init__(self, policy):
        self._policy = policy
        self._on_error = policy.on_error
        self.by_cause: Counter = Counter()
        self.throttling_started = False
        policy.on_error = self._record

    def _record(self, error: BaseException) -> None:
        self.by_cause[overload_evidence(error) or type(error).__name__] += 1
        self._on_error(error)
        self.throttling_started = self.throttling_started or self._policy.is_paced()

    def detach(self) -> None:
        del self._policy.on_error  # back to the class method


class BenchmarkPoint(DataPoint):
    text: str
    metadata: dict = {"index_fields": ["text"]}


class EmbedOnlyStore:
    """The two vector-store calls ``index_data_points`` makes: embed each batch, store nothing."""

    def __init__(self, embedding_engine):
        self.embedding_engine = embedding_engine

    async def create_vector_index(self, _type_name, _field_name) -> None:
        return None

    async def index_data_points(self, _type_name, _field_name, points) -> None:
        await self.embedding_engine.embed_text([point.text for point in points])


def summarize(cap, chunks, elapsed, latencies, failures) -> dict:
    ordered = sorted(latencies)
    return {
        "cap": cap,
        "chunks": chunks,
        "requests": len(ordered),
        "seconds": round(elapsed, 2),
        "chunks_per_second": round(chunks / elapsed, 1) if elapsed else None,
        "latency_p50": _at(ordered, 0.50),
        "latency_p95": _at(ordered, 0.95),
        "latency_max": ordered[-1] if ordered else None,
        "failed_attempts": dict(failures.by_cause),
        "throttling_started": failures.throttling_started,
    }


def print_row(row: dict) -> None:
    def ms(seconds):
        return f"{seconds * 1000:.0f}ms" if seconds is not None else "-"

    print(
        f"cap={row['cap']:>4}  {row['seconds']:>8.2f}s  {row['chunks_per_second']:>8} chunks/s  "
        f"p50={ms(row['latency_p50'])}  p95={ms(row['latency_p95'])}  "
        f"failed={row['failed_attempts'] or 0}  throttled={row['throttling_started']}  "
        f"cpu={row['cpu_share']}  lag_p95={row['loop_lag_p95_ms']}ms  "
        f"call_p50={ms(row['provider_call_p50'])}  decode={row['decode_seconds']}s  "
        f"conns={row['max_https_connections']}",
        flush=True,
    )


def write_results(path: str, args, results: list[dict]) -> None:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    settings = {key: value for key, value in vars(args).items() if key != "api_key"}
    with open(path, "w") as file:
        json.dump({"commit": commit, "settings": settings, "results": results}, file, indent=2)


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
