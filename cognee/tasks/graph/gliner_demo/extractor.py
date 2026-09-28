"""Thin wrapper around the ``gliner2`` local runtime.

One extractor per model name is loaded lazily and reused for the process; a
per-call load would dominate runtime. Inference is synchronous torch, so the
async helpers run it under ``asyncio.to_thread``.

Model batches run concurrently on one shared model. A single forward pass keeps
only 2-3 cores busy, because DeBERTa's small matrices do not spread across
torch's intra-op pool, so running several batches at once is what uses the rest
of the CPU. Torch releases the GIL inside its kernels, so threads run truly in
parallel without a second copy of the model. A process-wide pool bounds the
concurrency across every pipeline in the process, which also bounds memory:
each in-flight batch holds its own activations. With one thread, a call runs
its batches one after another.

Calls on the shared model need no lock. The runtime's per-call state changes
are idempotent for inference (eval mode, ``is_training=False``), and this was
checked directly: two pipelines cognified at once with the lock removed, calls
overlapping on the model, produced a graph identical to the sequential one.

Extraction can also be spread over several processes (GLINER_INFERENCE_PROCESSES,
or ``gliner_processes`` on cognify/remember; default 1). The calling process
keeps every Nth batch and spawned workers run the rest, each with its own model
and thread pool. Workers are shut down after WORKER_IDLE_SECONDS without work,
so a one-off ingest does not pin a model copy per worker for the life of the
process. On one machine threads already use the CPU, so processes mostly cost
memory; they are the building block for spreading work across machines.

There is one thread pool per process and one worker pool, each grown to the
largest size any call asked for. A call's own limit is enforced by how many
batches it keeps in flight, so calls with different settings share the pools
instead of stacking pools on top of each other.

Concurrency never changes the output. The long-text path is reproduced step for
step (the same windows, the same batches of windows in the same order, the same
merge), and torch keeps its intra-op thread count, so every batch computes
bit-identical scores whichever thread runs it.

Texts are extracted with ``batch_extract_long``: cognee chunks are cut against
the embedding model's token budget and routinely exceed the encoder's 512-token
window, and plain ``batch_extract`` silently loses almost everything past it.
The long path scans overlapping word windows and merges them, so no offset
handling happens on our side. ``overlap_policy="longest"`` resolves nested
spans inside the model.
"""

from __future__ import annotations

import asyncio
import multiprocessing
import os
import threading
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import (
    FIRST_COMPLETED,
    Executor,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any

from cognee.modules.cognify.config import get_cognify_config
from cognee.shared.logging_utils import get_logger
from cognee.shared.model_download_notice import log_model_load

from .schema import GlinerSchema

logger = get_logger("gliner.extractor")

DEFAULT_MODEL = "fastino/gliner2.5-base-v1"
# Approximate download sizes, for the first-use notice.
MODEL_SIZE_HINTS = {DEFAULT_MODEL: "about 750 MB"}
DEFAULT_THRESHOLD = 0.5
DEFAULT_BATCH_SIZE = 16
# Word-level window the runtime scans a long text with; 384 words stays under the
# 512-token DeBERTa window for ordinary prose, 64 words of overlap re-attaches
# mentions cut by a window boundary.
DEFAULT_WINDOW_WORDS = 384
DEFAULT_WINDOW_OVERLAP_WORDS = 64
OVERLAP_POLICY = "longest"

INSTALL_HINT = (
    "The GLiNER extraction path needs the `gliner2` package. "
    'Install it with: pip install "cognee[gliner]"'
)


class GlinerNotInstalledError(ImportError):
    """Raised when ``gliner2`` cannot be imported."""

    def __init__(self, message: str = INSTALL_HINT):
        super().__init__(message)


# Sizing the inference pool, from measurements on War and Peace (10-core
# machine, batch_size 16): throughput peaks when concurrent batches equal half
# of torch's intra-op thread count, and each concurrent batch of 16 windows
# holds about 1.7 GB of activations on top of the loaded model.
CORES_PER_CONCURRENT_BATCH = 2
BYTES_PER_CONCURRENT_BATCH = 2 * 1024**3  # at DEFAULT_BATCH_SIZE, rounded up
MEMORY_RESERVE_BYTES = 4 * 1024**3  # left free for the rest of the system
# Each extra process loads its own model: about 2.9 GB measured, rounded up.
BYTES_PER_MODEL = 3 * 1024**3
# Linux containers (Docker --cpus/--memory, Kubernetes limits) enforce CPU and
# memory through cgroups, which neither torch's thread count nor /proc/meminfo
# reflects: a pod limited to 4 GB still sees the node's free memory.
CGROUP_ROOT = Path("/sys/fs/cgroup")

_extractors: dict[str, Any] = {}
_load_lock = threading.Lock()
_pool_lock = threading.Lock()
# (batch_size, requested processes, requested threads) -> (processes, threads per process)
_resolved_concurrency: dict[tuple, tuple[int, int]] = {}
_thread_pool_instance: ThreadPoolExecutor | None = None
_thread_pool_size = 0
_process_pool_instance: ProcessPoolExecutor | None = None
_process_pool_model: str | None = None
_process_pool_workers = 0
_process_pool_users = 0
_process_pool_last_used = 0.0
_reaper: threading.Timer | None = None
# Worker processes are dropped after this long without work; the next call
# that needs them starts fresh ones (about 10 s each, for the model load).
WORKER_IDLE_SECONDS = 60.0
# Set in worker processes only, by _worker_init.
_worker: dict[str, Any] = {}


def require_gliner2() -> None:
    """Fail fast with an install hint when the optional dependency is missing."""
    try:
        import gliner2
    except ImportError as error:
        raise GlinerNotInstalledError() from error


def hub_model_cached(model_name: str) -> tuple[bool, str]:
    """Whether the hub model is already in the local cache, and where that cache is.

    A local directory counts as cached. Zero-network: ``try_to_load_from_cache``
    only looks at the cache on disk, the same one ``from_pretrained`` reads.
    """
    from huggingface_hub import constants, try_to_load_from_cache

    if os.path.isdir(model_name):
        return True, model_name
    cached = isinstance(try_to_load_from_cache(model_name, "config.json"), str)
    return cached, constants.HF_HUB_CACHE


def load_extractor(model_name: str = DEFAULT_MODEL) -> Any:
    """Load (once) and return the ``gliner2`` extractor for ``model_name``."""
    with _load_lock:
        extractor = _extractors.get(model_name)
        if extractor is not None:
            return extractor

        require_gliner2()
        from gliner2 import AutoExtractor

        cached, cache_dir = hub_model_cached(model_name)
        log_model_load(
            logger,
            model=model_name,
            cached=cached,
            cache_dir=cache_dir,
            size_hint=MODEL_SIZE_HINTS.get(model_name),
            location_var="HF_HOME",
        )
        started = time.perf_counter()
        extractor = AutoExtractor.from_pretrained(model_name)
        logger.info("GLiNER model %s ready in %.1fs", model_name, time.perf_counter() - started)
        _extractors[model_name] = extractor
        return extractor


async def get_extractor(model_name: str = DEFAULT_MODEL) -> Any:
    return await asyncio.to_thread(load_extractor, model_name)


def _read_cgroup(*names: str) -> str | None:
    """The first readable cgroup file among ``names`` (v2 name first, then v1)."""
    for name in names:
        try:
            return (CGROUP_ROOT / name).read_text().strip()
        except OSError:
            continue
    return None


def container_cpu_limit() -> float | None:
    """CPUs a cgroup quota grants this process, or None when there is no quota.

    Docker ``--cpus`` and Kubernetes CPU limits are a CFS quota: the process
    still sees every host CPU, it just gets throttled past the quota.
    """
    v2 = _read_cgroup("cpu.max")  # "max 100000" or "200000 100000"
    if v2:
        quota, period = v2.split()[:2]
        return None if quota == "max" else int(quota) / int(period)
    quota = _read_cgroup("cpu/cpu.cfs_quota_us")
    period = _read_cgroup("cpu/cpu.cfs_period_us")
    if quota and period and int(quota) > 0:
        return int(quota) / int(period)
    return None


def container_memory_free() -> int | None:
    """Bytes left under a cgroup memory limit, or None when there is no limit."""
    limit = _read_cgroup("memory.max", "memory/memory.limit_in_bytes")
    usage = _read_cgroup("memory.current", "memory/memory.usage_in_bytes")
    # cgroup v1 reports "no limit" as a number near 2**63.
    if not limit or not usage or limit == "max" or int(limit) >= 2**60:
        return None
    return max(0, int(limit) - int(usage))


def auto_inference_threads(batch_size: int = DEFAULT_BATCH_SIZE, processes: int = 1) -> int:
    """How many model batches each process can run at once, at full speed.

    The machine's total is the tighter of two bounds. The CPU bound is half of
    the CPUs torch will use: its intra-op thread count, which torch sizes to the
    physical cores, lowered to a container's CPU quota. The memory bound keeps
    every concurrent batch's activations inside available memory, less a
    reserve and the model copy each extra process loads, where "available" is
    the tighter of the machine's free memory and a container's remaining
    limit, so a busy machine or a small pod gets fewer threads instead of
    swapping or an OOM kill. The total is split evenly across the processes.
    """
    import psutil
    import torch

    cpus = torch.get_num_threads()
    quota = container_cpu_limit()
    if quota is not None:
        cpus = min(cpus, max(1, int(quota)))
    cpu_bound = cpus // CORES_PER_CONCURRENT_BATCH

    available = psutil.virtual_memory().available
    container_free = container_memory_free()
    if container_free is not None:
        available = min(available, container_free)
    available -= (processes - 1) * BYTES_PER_MODEL
    per_batch = BYTES_PER_CONCURRENT_BATCH * batch_size / DEFAULT_BATCH_SIZE
    memory_bound = 1 + int(max(0, available - MEMORY_RESERVE_BYTES) // per_batch)
    total = max(1, min(cpu_bound, memory_bound))
    if processes > total:
        logger.warning(
            "GLiNER inference: %d processes asked for, but this machine has room for "
            "%d concurrent model batch(es); each process still runs one, so memory may "
            "be tighter than the sizing intends. Consider fewer processes.",
            processes,
            total,
        )
    return max(1, total // processes)


def inference_processes(requested: int | None = None) -> int:
    """Processes sharing extraction: ``requested``, else GLINER_INFERENCE_PROCESSES."""
    processes = get_cognify_config().gliner_inference_processes if requested is None else requested
    if processes < 1:
        raise ValueError(f"GLiNER inference processes must be >= 1, got {processes}")
    return processes


def inference_threads(
    batch_size: int = DEFAULT_BATCH_SIZE, processes: int = 1, requested: int | None = None
) -> int:
    """Threads per process: ``requested``, else GLINER_INFERENCE_THREADS; 0 means auto."""
    configured = get_cognify_config().gliner_inference_threads if requested is None else requested
    if configured < 0:
        raise ValueError(f"GLINER_INFERENCE_THREADS must be >= 0, got {configured}")
    return configured or auto_inference_threads(batch_size, processes)


def _concurrency(
    batch_size: int, processes: int | None = None, threads: int | None = None
) -> tuple[int, int]:
    """Resolve (processes, threads per process) once per call shape.

    Auto-sizing reads free memory, which drops once the pools fill it, so the
    first answer is kept for the life of the process.
    """
    key = (batch_size, processes, threads)
    with _pool_lock:
        if key not in _resolved_concurrency:
            resolved_processes = inference_processes(processes)
            resolved_threads = inference_threads(batch_size, resolved_processes, threads)
            _resolved_concurrency[key] = (resolved_processes, resolved_threads)
            logger.info(
                "GLiNER inference: %d process(es) x %d concurrent model batch(es)",
                resolved_processes,
                resolved_threads,
            )
        return _resolved_concurrency[key]


def _thread_pool(threads: int) -> ThreadPoolExecutor:
    """The one process-wide pool, grown to the largest size any call asked for.

    Growing replaces the pool; batches already running finish on the old one.
    The pool bounds how many batches the whole process runs at once, and each
    call bounds its own share by how many it keeps in flight (``_run_batches``).
    """
    global _thread_pool_instance, _thread_pool_size
    with _pool_lock:
        if _thread_pool_instance is None or _thread_pool_size < threads:
            if _thread_pool_instance is not None:
                _thread_pool_instance.shutdown(wait=False)
            _thread_pool_instance = ThreadPoolExecutor(threads, thread_name_prefix="gliner")
            _thread_pool_size = threads
        return _thread_pool_instance


def _inference_pool(
    batch_size: int, processes: int | None = None, threads: int | None = None
) -> ThreadPoolExecutor | None:
    """This process's thread pool for the given call shape, or None for one thread."""
    resolved_threads = _concurrency(batch_size, processes, threads)[1]
    return _thread_pool(resolved_threads) if resolved_threads > 1 else None


def _process_pool(model_name: str, workers: int) -> ProcessPoolExecutor:
    """The one worker pool, grown to the most workers any call asked for.

    Spawned rather than forked: forking a process that already runs torch
    threads is unsafe. Spawned workers import the calling script, which is why
    a script that uses processes needs an ``if __name__ == "__main__":`` guard.
    Every caller must hand the pool back through ``_release_process_pool``.
    """
    global _process_pool_instance, _process_pool_model, _process_pool_workers
    global _process_pool_users
    with _pool_lock:
        replace = (
            _process_pool_instance is None
            or _process_pool_model != model_name
            or _process_pool_workers < workers
        )
        if replace:
            _drop_process_pool_locked(wait=False)
            _process_pool_instance = ProcessPoolExecutor(
                workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_worker_init,
                initargs=(model_name,),
            )
            _process_pool_model, _process_pool_workers = model_name, workers
        _process_pool_users += 1
        return _process_pool_instance


def _release_process_pool(broken: bool = False) -> None:
    """Hand the worker pool back; drop it if broken, else start the idle clock."""
    global _process_pool_users, _process_pool_last_used
    with _pool_lock:
        _process_pool_users -= 1
        _process_pool_last_used = time.monotonic()
        if broken:
            # A worker died (killed for memory, crashed). A broken pool stays
            # broken, so drop it: the next call starts fresh workers.
            _drop_process_pool_locked(wait=False)
        elif _process_pool_users == 0:
            _arm_reaper_locked()


def _drop_process_pool_locked(wait: bool) -> None:
    global _process_pool_instance, _process_pool_model, _process_pool_workers
    pool, _process_pool_instance = _process_pool_instance, None
    _process_pool_model, _process_pool_workers = None, 0
    if pool is not None:
        pool.shutdown(wait=wait, cancel_futures=not wait)


def _arm_reaper_locked() -> None:
    global _reaper
    if _reaper is None:
        _reaper = threading.Timer(WORKER_IDLE_SECONDS, _reap_idle_workers)
        _reaper.daemon = True
        _reaper.start()


def _reap_idle_workers() -> None:
    """Timer callback: drop the worker pool once it has sat idle long enough."""
    global _reaper
    with _pool_lock:
        _reaper = None
        if _process_pool_instance is None:
            return
        idle = time.monotonic() - _process_pool_last_used
        if _process_pool_users == 0 and idle >= WORKER_IDLE_SECONDS:
            logger.info("GLiNER inference: worker processes idle, shutting them down")
            _drop_process_pool_locked(wait=False)
        else:
            _arm_reaper_locked()


def reset_inference_pool() -> None:
    """Shut every pool down so the next call sizes new ones (tests, config changes)."""
    global _thread_pool_instance, _thread_pool_size, _reaper
    with _pool_lock:
        if _reaper is not None:
            _reaper.cancel()
            _reaper = None
        if _thread_pool_instance is not None:
            _thread_pool_instance.shutdown(wait=True)
        _thread_pool_instance, _thread_pool_size = None, 0
        _drop_process_pool_locked(wait=True)
        _resolved_concurrency.clear()


def _run_batches(
    extractor: Any,
    built: Any,
    batches: Sequence[list[str]],
    threads: int,
    *,
    threshold: float,
    batch_size: int,
    window_words: int,
) -> list[list[dict[str, Any]]]:
    """Run model batches with at most ``threads`` in flight; one result list per batch.

    One thread means the historic behaviour: the batches run in turn on the
    calling thread. More means the shared pool, with this call submitting only
    ``threads`` batches at a time, so a call with a small limit never occupies
    pool threads it is not allowed to use.
    """

    def run_batch(batch: list[str]) -> list[dict[str, Any]]:
        return extractor.batch_extract(
            batch,
            built,
            batch_size=batch_size,
            threshold=threshold,
            num_workers=0,
            format_results=True,
            include_confidence=True,
            include_spans=True,
            max_len=window_words,
            overlap_policy=OVERLAP_POLICY,
        )

    if threads <= 1:
        return [run_batch(batch) for batch in batches]

    pool = _thread_pool(threads)
    results: list = [None] * len(batches)
    queue = iter(enumerate(batches))
    pending: dict = {}

    def top_up() -> None:
        while len(pending) < threads:
            try:
                index, batch = next(queue)
            except StopIteration:
                return
            pending[pool.submit(run_batch, batch)] = index

    top_up()
    while pending:
        done, _ = wait(pending, return_when=FIRST_COMPLETED)
        for future in done:
            results[pending.pop(future)] = future.result()
        top_up()
    return results


def _worker_init(model_name: str) -> None:
    """Worker process start-up: load the model once."""
    _worker["extractor"] = load_extractor(model_name)


def _worker_run_share(
    schema: GlinerSchema, batches: list[list[str]], threads: int, options: dict[str, Any]
) -> list[list[dict[str, Any]]]:
    """Worker process entry point: run this process's share of the batches."""
    extractor = _worker["extractor"]
    built = build_gliner_schema(extractor, schema)
    return _run_batches(extractor, built, batches, threads, **options)


def _extract_long_concurrently(
    extractor: Any,
    texts: Sequence[str],
    built: Any,
    *,
    threads: int,
    processes: int = 1,
    schema: GlinerSchema | None = None,
    process_pool: Executor | None = None,
    threshold: float,
    batch_size: int,
    window_words: int,
    window_overlap_words: int,
) -> list[Mapping[str, Any]]:
    """``batch_extract_long`` with its model batches spread over threads and processes.

    Mirrors the runtime's long-text path step for step: the same split into
    overlapping word windows, the same batches of ``batch_size`` windows in the
    same order, the same merge. Batches are dealt round-robin: every
    ``processes``-th batch stays here, the rest go to the worker processes in
    ``process_pool``. Only where each batch runs differs.
    """
    from gliner2.inference.chunking import merge_chunk_results, split_text_into_chunks
    from gliner2.processing.word_splitter import word_splitter_from

    splitter = word_splitter_from(extractor)
    windows = [
        split_text_into_chunks(
            text,
            chunk_size=window_words,
            chunk_overlap=window_overlap_words,
            word_splitter=splitter,
        )
        for text in texts
    ]
    window_texts = [window.text for document in windows for window in document]
    batches = [
        window_texts[start : start + batch_size]
        for start in range(0, len(window_texts), batch_size)
    ]
    options = {"threshold": threshold, "batch_size": batch_size, "window_words": window_words}

    shares = [batches[index::processes] for index in range(processes)]
    remote = {}
    if processes > 1:
        if process_pool is None or schema is None:
            raise ValueError("processes > 1 needs a process_pool and the schema for the workers")
        remote = {
            index: process_pool.submit(_worker_run_share, schema, share, threads, options)
            for index, share in enumerate(shares)
            if index > 0 and share
        }
    per_batch: list = [None] * len(batches)
    per_batch[0::processes] = _run_batches(extractor, built, shares[0], threads, **options)
    for index, future in remote.items():
        per_batch[index::processes] = future.result()
    window_results = [result for batch in per_batch for result in batch]

    merged, offset = [], 0
    scalar_labels = extractor._scalar_entity_labels(built)
    overlap_policy = extractor._resolved_overlap_policy(OVERLAP_POLICY)
    for text, document in zip(texts, windows):
        merged.append(
            merge_chunk_results(
                text,
                document,
                window_results[offset : offset + len(document)],
                include_confidence=True,
                include_spans=True,
                scalar_entity_labels=scalar_labels,
                overlap_policy=overlap_policy,
            )
        )
        offset += len(document)
    return merged


def build_gliner_schema(extractor: Any, schema: GlinerSchema) -> Any:
    """Turn a resolved :class:`GlinerSchema` into the runtime's schema builder."""
    builder = extractor.create_schema()
    if schema.entity_types:
        builder = builder.entities(dict(schema.entity_types))
    if schema.relation_types:
        builder = builder.relations(dict(schema.relation_types))
    return builder


def extract_batch(
    extractor: Any,
    texts: Sequence[str],
    schema: GlinerSchema,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    batch_size: int = DEFAULT_BATCH_SIZE,
    window_words: int = DEFAULT_WINDOW_WORDS,
    window_overlap_words: int = DEFAULT_WINDOW_OVERLAP_WORDS,
    processes: int | None = None,
    threads: int | None = None,
    model_name: str = DEFAULT_MODEL,
) -> list[Mapping[str, Any]]:
    """Run one batched entity+relation extraction; one result dict per text.

    ``processes`` and ``threads`` override GLINER_INFERENCE_PROCESSES and
    GLINER_INFERENCE_THREADS for this call; ``model_name`` is what worker
    processes load.
    """
    if not texts:
        return []
    if schema.is_empty:
        return [{} for _ in texts]

    built = build_gliner_schema(extractor, schema)
    processes, threads = _concurrency(batch_size, processes, threads)
    if processes == 1 and threads <= 1:
        return extractor.batch_extract_long(
            list(texts),
            built,
            batch_size=batch_size,
            threshold=threshold,
            include_confidence=True,
            include_spans=True,
            chunk_size=window_words,
            chunk_overlap=window_overlap_words,
            overlap_policy=OVERLAP_POLICY,
        )

    process_pool = _process_pool(model_name, processes - 1) if processes > 1 else None
    broken = False
    try:
        return _extract_long_concurrently(
            extractor,
            texts,
            built,
            threads=threads,
            processes=processes,
            schema=schema,
            process_pool=process_pool,
            threshold=threshold,
            batch_size=batch_size,
            window_words=window_words,
            window_overlap_words=window_overlap_words,
        )
    except BrokenProcessPool:
        broken = True
        raise
    finally:
        if process_pool is not None:
            _release_process_pool(broken)


def extract_once(
    extractor: Any,
    text: str,
    schema: GlinerSchema,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    processes: int | None = None,
    threads: int | None = None,
) -> Mapping[str, Any]:
    """Run one unchunked entity+relation extraction.

    Runs on this process's thread pool for the same ``processes`` / ``threads``
    as the extraction, so a call's schema probes stay inside its concurrency.
    """
    if not text or schema.is_empty:
        return {}

    built = build_gliner_schema(extractor, schema)

    def run() -> Mapping[str, Any]:
        return extractor.extract(
            text,
            built,
            threshold=threshold,
            include_confidence=False,
            include_spans=False,
            overlap_policy=OVERLAP_POLICY,
        )

    # Inside the same concurrency as the extraction, so a call's schema probes
    # count against the limit it asked for.
    resolved_threads = _concurrency(DEFAULT_BATCH_SIZE, processes, threads)[1]
    if resolved_threads <= 1:
        return run()
    return _thread_pool(resolved_threads).submit(run).result()


async def extract_batch_async(
    extractor: Any, texts: Sequence[str], schema: GlinerSchema, **options
):
    return await asyncio.to_thread(extract_batch, extractor, texts, schema, **options)
