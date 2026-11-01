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
each in-flight batch holds its own activations. With one thread there is no
pool; a lock makes calls take turns, so one batch runs at a time.

Calls on the shared model need no lock for correctness (the one-thread lock
only bounds memory). The runtime writes three things per
call, all idempotent for inference: eval mode, ``is_training=False``, and, on
the ``extract`` path with no ``max_len`` (the schema probe), a lazily cached
default collator that two concurrent probes may each construct, with equivalent
results. This was checked directly: two pipelines cognified at once with the
lock removed, calls overlapping on the model, produced a graph identical to the
sequential one.

Extraction can also be spread over several processes (GLINER_INFERENCE_PROCESSES,
default 1). The calling process
keeps every Nth batch and spawned workers run the rest, each with its own model
and thread pool. Workers are shut down after WORKER_IDLE_SECONDS without work,
so a one-off ingest does not pin a model copy per worker for the life of the
process. On one machine threads already use the CPU, so processes mostly cost
memory; they are the building block for spreading work across machines.

There is one thread pool per process and one worker pool per model, sized once
from the configuration and never replaced while in use, so every call shares
them and the machine-sized bound holds across the whole process.

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
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import (
    Executor,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
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
# reflects: a pod limited to 4 GB still sees the node's free memory. The limit
# files live in the process's own cgroup directory, named by /proc/self/cgroup:
# the root of the mount only inside a private cgroup namespace (Docker's
# default), elsewhere a path such as /kubepods.slice/.../cri-containerd-x.scope.
CGROUP_ROOT = Path("/sys/fs/cgroup")
PROC_SELF_CGROUP = Path("/proc/self/cgroup")
# cgroup v1 reports "no limit" as a number near 2**63.
CGROUP_V1_NO_LIMIT = 2**60

_extractors: dict[str, Any] = {}
_load_lock = threading.Lock()
_pool_lock = threading.Lock()
# With one thread a call runs its batches on the calling thread, outside any
# pool, so this lock makes concurrent callers take turns on the model: without
# it every pipeline's batch would be in flight at once, each holding its own
# activations.
_inference_lock = threading.Lock()
# (processes, threads per process) the pools are sized to; see _pool_sizes.
_sizes: tuple[int, int] | None = None
_thread_pool_instance: ThreadPoolExecutor | None = None
# model name -> its worker processes
_worker_pools: dict[str, _WorkerPool] = {}
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


def _read_cgroup(path: Path) -> str | None:
    """The stripped text of one cgroup file, or None when it cannot be read."""
    try:
        return path.read_text().strip()
    except OSError:
        return None


def _cgroup_dirs(controller: str) -> list[Path]:
    """The directories whose ``controller`` limits apply to this process, nearest first.

    Resolved from ``/proc/self/cgroup``. A line ``0::/a/b`` is the cgroup v2
    hierarchy, mounted at the root; ``3:cpu,cpuacct:/a/b`` is a cgroup v1
    hierarchy, mounted under its controller names. The process's own directory
    is listed first, then every ancestor up to the mount root, since a limit
    on any ancestor (a Kubernetes pod's slice, a Docker container's scope)
    binds the process as much as its own. Inside a private cgroup namespace the
    process is at ``/`` and only the root is listed. No file (macOS, Windows)
    means no cgroups, so nothing is listed.
    """
    lines = _read_cgroup(PROC_SELF_CGROUP)
    if not lines:
        return []
    dirs: list[Path] = []
    for line in lines.splitlines():
        _, controllers, path = line.split(":", 2)
        if controllers == "":
            mount = CGROUP_ROOT
        elif controller in controllers.split(","):
            mount = CGROUP_ROOT / controllers
        else:
            continue
        cgroup = mount / path.lstrip("/")
        dirs.append(cgroup)
        while cgroup != mount:
            cgroup = cgroup.parent
            dirs.append(cgroup)
    return dirs


def container_cpu_limit() -> float | None:
    """CPUs the tightest cgroup quota grants this process, or None when there is none.

    Docker ``--cpus`` and Kubernetes CPU limits are a CFS quota: the process
    still sees every host CPU, it just gets throttled past the quota. Every
    level from the process's cgroup up to the root is read and the smallest
    quota wins.
    """
    quotas: list[float] = []
    for cgroup in _cgroup_dirs("cpu"):
        v2 = _read_cgroup(cgroup / "cpu.max")  # "max 100000" or "200000 100000"
        if v2 is not None:
            quota, period = v2.split()[:2]
            if quota != "max":
                quotas.append(int(quota) / int(period))
            continue
        quota = _read_cgroup(cgroup / "cpu.cfs_quota_us")
        period = _read_cgroup(cgroup / "cpu.cfs_period_us")
        if quota and period and int(quota) > 0:
            quotas.append(int(quota) / int(period))
    return min(quotas) if quotas else None


def container_memory_free() -> int | None:
    """Bytes left under the tightest cgroup memory limit, or None when there is none.

    Each level from the process's cgroup up to the root is read; what is left
    under a limit is that level's limit minus that level's usage (an ancestor's
    usage includes its other children), and the smallest remainder wins.
    """
    remaining: list[int] = []
    for cgroup in _cgroup_dirs("memory"):
        limit = _read_cgroup(cgroup / "memory.max")
        usage = _read_cgroup(cgroup / "memory.current")
        if limit is None:
            limit = _read_cgroup(cgroup / "memory.limit_in_bytes")
            usage = _read_cgroup(cgroup / "memory.usage_in_bytes")
        if not limit or not usage or limit == "max" or int(limit) >= CGROUP_V1_NO_LIMIT:
            continue
        remaining.append(max(0, int(limit) - int(usage)))
    return min(remaining) if remaining else None


def auto_inference_threads(processes: int = 1) -> int:
    """How many model batches each process can run at once, at full speed.

    The machine's total is the tighter of two bounds. The CPU bound is half of
    the CPUs torch will use: its intra-op thread count, which torch sizes to the
    physical cores, lowered to a container's CPU quota. The memory bound keeps
    every concurrent batch's activations inside available memory, less a
    reserve and the model copy each extra process loads, where "available" is
    the tighter of the machine's free memory and a container's remaining
    limit, so a busy machine or a small pod gets fewer threads instead of
    swapping or an OOM kill. The total is split evenly across the processes.

    The memory bound assumes batches of ``DEFAULT_BATCH_SIZE`` windows, the
    pipeline's default. Concurrency is resolved once per call shape and the
    first caller is the per-document schema probe, which has no batch size of
    its own, so sizing on a caller's batch size would size on the wrong one. A
    pipeline built with a larger ``gliner_batch_size`` holds proportionally
    more activations per batch and should set ``GLINER_INFERENCE_THREADS``
    itself.
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
    memory_bound = 1 + int(max(0, available - MEMORY_RESERVE_BYTES) // BYTES_PER_CONCURRENT_BATCH)
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


def inference_processes() -> int:
    """Processes sharing extraction (GLINER_INFERENCE_PROCESSES)."""
    return get_cognify_config().gliner_inference_processes


def inference_threads(processes: int = 1) -> int:
    """Threads per process (GLINER_INFERENCE_THREADS), auto-sized when 0."""
    configured = get_cognify_config().gliner_inference_threads
    return configured or auto_inference_threads(processes)


def _pool_sizes() -> tuple[int, int]:
    """(processes, threads per process) that this process's pools are sized to.

    Resolved once, from the configuration: auto-sizing reads free memory, which
    drops once the pools fill it, so the first answer is kept for the life of
    the process. Worker processes are handed theirs by ``_worker_init``.
    """
    global _sizes
    with _pool_lock:
        if _sizes is None:
            processes = inference_processes()
            _sizes = (processes, inference_threads(processes))
            logger.info("GLiNER inference: %d process(es) x %d concurrent model batch(es)", *_sizes)
        return _sizes


def _thread_pool() -> ThreadPoolExecutor | None:
    """The process-wide pool for model batches, or None for one configured thread.

    Created once at the configured size and never replaced while the process
    runs, so a call can hold on to it; it bounds how many batches the whole
    process runs at once.
    """
    global _thread_pool_instance
    threads = _pool_sizes()[1]
    if threads <= 1:
        return None
    with _pool_lock:
        if _thread_pool_instance is None:
            _thread_pool_instance = ThreadPoolExecutor(threads, thread_name_prefix="gliner")
        return _thread_pool_instance


@dataclass
class _WorkerPool:
    """One model's worker processes, and the calls currently using them."""

    model_name: str
    executor: ProcessPoolExecutor
    users: int = 0
    last_used: float = 0.0


def _acquire_worker_pool(model_name: str) -> _WorkerPool:
    """Borrow the model's worker pool: GLINER_INFERENCE_PROCESSES - 1 processes.

    Created once per model at the configured size and never replaced while in
    use. Spawned rather than forked: forking a process that already runs torch
    threads is unsafe. Spawned workers import the calling script, which is why
    a script that uses processes needs an ``if __name__ == "__main__":`` guard.
    Every caller must hand the pool back through ``_release_worker_pool``.
    """
    processes, threads = _pool_sizes()
    with _pool_lock:
        pool = _worker_pools.get(model_name)
        if pool is None:
            executor = ProcessPoolExecutor(
                processes - 1,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_worker_init,
                initargs=(model_name, threads),
            )
            pool = _worker_pools[model_name] = _WorkerPool(model_name, executor)
        pool.users += 1
        return pool


def _release_worker_pool(pool: _WorkerPool, broken: bool = False) -> None:
    """Hand a borrowed pool back; drop it if broken, else start the idle clock."""
    with _pool_lock:
        pool.users -= 1
        pool.last_used = time.monotonic()
        if broken:
            # A worker died (killed for memory, crashed). A broken pool stays
            # broken, so drop it: the next call starts fresh workers. Only this
            # pool: a fresh one another call already started is left alone.
            _drop_worker_pool_locked(pool)
        elif pool.users == 0:
            _arm_reaper_locked()


def _drop_worker_pool_locked(pool: _WorkerPool, wait: bool = False) -> None:
    if _worker_pools.get(pool.model_name) is pool:
        del _worker_pools[pool.model_name]
    pool.executor.shutdown(wait=wait, cancel_futures=not wait)


def _arm_reaper_locked() -> None:
    global _reaper
    if _reaper is None:
        _reaper = threading.Timer(WORKER_IDLE_SECONDS, _reap_idle_workers)
        _reaper.daemon = True
        _reaper.start()


def _reap_idle_workers() -> None:
    """Timer callback: drop worker pools that have sat idle long enough."""
    global _reaper
    with _pool_lock:
        _reaper = None
        now = time.monotonic()
        for pool in list(_worker_pools.values()):
            if pool.users == 0 and now - pool.last_used >= WORKER_IDLE_SECONDS:
                logger.info("GLiNER inference: worker processes idle, shutting them down")
                _drop_worker_pool_locked(pool)
        if _worker_pools:
            _arm_reaper_locked()


def reset_inference_pool() -> None:
    """Shut every pool down so the next call sizes new ones (tests, config changes)."""
    global _thread_pool_instance, _reaper, _sizes
    with _pool_lock:
        if _reaper is not None:
            _reaper.cancel()
            _reaper = None
        if _thread_pool_instance is not None:
            _thread_pool_instance.shutdown(wait=True)
        _thread_pool_instance = None
        for pool in list(_worker_pools.values()):
            _drop_worker_pool_locked(pool, wait=True)
        _sizes = None


def _run_batches(
    extractor: Any,
    built: Any,
    batches: Sequence[list[str]],
    *,
    threshold: float,
    batch_size: int,
    window_words: int,
) -> list[list[dict[str, Any]]]:
    """Run model batches; one result list per batch.

    With no pool (one configured thread) the batches run in turn on the calling
    thread, taking turns with other callers' batches on the inference lock.
    Otherwise they all go to the shared pool, whose size bounds how many run at
    once across every caller in the process.
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

    pool = _thread_pool()
    if pool is None:
        results = []
        for batch in batches:
            with _inference_lock:
                results.append(run_batch(batch))
        return results

    futures = [pool.submit(run_batch, batch) for batch in batches]
    try:
        return [future.result() for future in futures]
    except BaseException:
        _abandon_shares(futures)
        raise


def _worker_init(model_name: str, threads: int) -> None:
    """Worker process start-up: load the model once, take the parent's thread count."""
    global _sizes
    _sizes = (1, threads)
    _worker["extractor"] = load_extractor(model_name)


def _worker_run_share(
    schema: GlinerSchema, batches: list[list[str]], options: dict[str, Any]
) -> list[list[dict[str, Any]]]:
    """Worker process entry point: run this process's share of the batches."""
    extractor = _worker["extractor"]
    built = build_gliner_schema(extractor, schema)
    return _run_batches(extractor, built, batches, **options)


def _abandon_shares(futures: Iterable[Future]) -> None:
    """After a failed call: cancel its queued shares and wait out the running ones,
    so the caller releases the workers only once none of its work is left on them."""
    futures = list(futures)
    for future in futures:
        future.cancel()
    wait(futures)


def _extract_long_concurrently(
    extractor: Any,
    texts: Sequence[str],
    built: Any,
    *,
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
            index: process_pool.submit(_worker_run_share, schema, share, options)
            for index, share in enumerate(shares)
            if index > 0 and share
        }
    per_batch: list = [None] * len(batches)
    try:
        per_batch[0::processes] = _run_batches(extractor, built, shares[0], **options)
        for index, future in remote.items():
            per_batch[index::processes] = future.result()
    except BaseException:
        _abandon_shares(remote.values())
        raise
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
    model_name: str,
) -> list[Mapping[str, Any]]:
    """Run one batched entity+relation extraction; one result dict per text.

    Spread over GLINER_INFERENCE_PROCESSES processes and GLINER_INFERENCE_THREADS
    threads each; worker processes load ``model_name``, so it must name the
    model ``extractor`` was loaded from.
    """
    if not texts:
        return []
    if schema.is_empty:
        return [{} for _ in texts]

    built = build_gliner_schema(extractor, schema)
    processes, threads = _pool_sizes()
    if processes == 1 and threads <= 1:
        with _inference_lock:
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

    worker_pool = _acquire_worker_pool(model_name) if processes > 1 else None
    broken = False
    try:
        return _extract_long_concurrently(
            extractor,
            texts,
            built,
            processes=processes,
            schema=schema,
            process_pool=worker_pool.executor if worker_pool is not None else None,
            threshold=threshold,
            batch_size=batch_size,
            window_words=window_words,
            window_overlap_words=window_overlap_words,
        )
    except BrokenProcessPool:
        broken = True
        raise
    finally:
        if worker_pool is not None:
            _release_worker_pool(worker_pool, broken)


def extract_once(
    extractor: Any,
    text: str,
    schema: GlinerSchema,
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> Mapping[str, Any]:
    """Run one unchunked entity+relation extraction."""
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

    # Through the same pool as batch extraction, so a schema probe counts
    # against the same concurrency (and memory) bound.
    pool = _thread_pool()
    if pool is not None:
        return pool.submit(run).result()
    with _inference_lock:
        return run()


async def extract_batch_async(
    extractor: Any, texts: Sequence[str], schema: GlinerSchema, **options
):
    return await asyncio.to_thread(extract_batch, extractor, texts, schema, **options)
