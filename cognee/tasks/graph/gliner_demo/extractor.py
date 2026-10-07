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
import os
import threading
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
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
_pool: ThreadPoolExecutor | None = None
_pool_size: int | None = None
_pool_lock = threading.Lock()
# With one thread there is no pool to bound concurrency, so this lock makes
# concurrent callers take turns on the model: without it every pipeline's
# batch would be in flight at once, each holding its own activations.
_inference_lock = threading.Lock()


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


def auto_inference_threads() -> int:
    """How many model batches this machine can run at once, at full speed.

    The CPU bound is half of the CPUs torch will use: its intra-op thread
    count, which torch sizes to the physical cores, lowered to a container's
    CPU quota. The memory bound keeps every concurrent batch's activations
    inside available memory, less a reserve, where "available" is the tighter
    of the machine's free memory and a container's remaining limit, so a busy
    machine or a small pod gets fewer threads instead of swapping or an OOM kill.

    The memory bound assumes batches of ``DEFAULT_BATCH_SIZE`` windows, the
    pipeline's default. The pool is sized once per process and its first user
    is the per-document schema probe, which has no batch size of its own, so
    sizing on a caller's batch size would size on the wrong one. A pipeline
    built with a larger ``gliner_batch_size`` holds proportionally more
    activations per batch and should set ``GLINER_INFERENCE_THREADS`` itself.
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
    memory_bound = 1 + int(max(0, available - MEMORY_RESERVE_BYTES) // BYTES_PER_CONCURRENT_BATCH)
    return max(1, min(cpu_bound, memory_bound))


def inference_threads() -> int:
    """The configured concurrency (GLINER_INFERENCE_THREADS), auto-sized when 0."""
    configured = get_cognify_config().gliner_inference_threads
    return configured or auto_inference_threads()


def _inference_pool() -> ThreadPoolExecutor | None:
    """The process-wide pool that runs model batches, or None for one thread.

    Sized once, on first use, and shared by every pipeline in the process, so
    concurrent documents queue behind one another instead of multiplying the
    memory in flight. The size assumes ``DEFAULT_BATCH_SIZE`` windows per
    batch (see ``auto_inference_threads``); a call's own batch size does not
    resize it.
    """
    global _pool, _pool_size
    with _pool_lock:
        if _pool_size is None:
            _pool_size = inference_threads()
            if _pool_size > 1:
                _pool = ThreadPoolExecutor(_pool_size, thread_name_prefix="gliner")
            logger.info("GLiNER inference: %d concurrent model batch(es)", _pool_size)
        return _pool


def reset_inference_pool() -> None:
    """Shut the pool down so the next call sizes a new one (tests, config changes)."""
    global _pool, _pool_size
    with _pool_lock:
        if _pool is not None:
            _pool.shutdown(wait=True)
        _pool, _pool_size = None, None


def _extract_long_concurrently(
    extractor: Any,
    pool: ThreadPoolExecutor,
    texts: Sequence[str],
    built: Any,
    *,
    threshold: float,
    batch_size: int,
    window_words: int,
    window_overlap_words: int,
) -> list[Mapping[str, Any]]:
    """``batch_extract_long`` with its model batches spread over ``pool``.

    Mirrors the runtime's long-text path step for step: the same split into
    overlapping word windows, the same batches of ``batch_size`` windows in the
    same order, the same merge. Only the thread that runs each batch differs.
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

    batches = [
        window_texts[start : start + batch_size]
        for start in range(0, len(window_texts), batch_size)
    ]
    window_results = [result for part in pool.map(run_batch, batches) for result in part]

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
) -> list[Mapping[str, Any]]:
    """Run one batched entity+relation extraction; one result dict per text."""
    if not texts:
        return []
    if schema.is_empty:
        return [{} for _ in texts]

    built = build_gliner_schema(extractor, schema)
    pool = _inference_pool()
    if pool is not None:
        return _extract_long_concurrently(
            extractor,
            pool,
            texts,
            built,
            threshold=threshold,
            batch_size=batch_size,
            window_words=window_words,
            window_overlap_words=window_overlap_words,
        )
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
    pool = _inference_pool()
    if pool is not None:
        return pool.submit(run).result()
    with _inference_lock:
        return run()


async def extract_batch_async(
    extractor: Any, texts: Sequence[str], schema: GlinerSchema, **options
):
    return await asyncio.to_thread(extract_batch, extractor, texts, schema, **options)
