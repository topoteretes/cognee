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
each in-flight batch holds its own activations. With one thread, calls are
serialized behind a lock as before.

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

_extractors: dict[str, Any] = {}
_load_lock = threading.Lock()
_inference_lock = threading.Lock()
_pool: ThreadPoolExecutor | None = None
_pool_size: int | None = None
_pool_lock = threading.Lock()


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


def auto_inference_threads(batch_size: int = DEFAULT_BATCH_SIZE) -> int:
    """How many model batches this machine can run at once, at full speed.

    The CPU bound is half of torch's intra-op thread count (torch sizes that to
    the physical cores). The memory bound keeps every concurrent batch's
    activations inside currently available memory, less a reserve, so a
    busy machine gets fewer threads instead of swapping.
    """
    import psutil
    import torch

    cpu_bound = torch.get_num_threads() // CORES_PER_CONCURRENT_BATCH
    per_batch = BYTES_PER_CONCURRENT_BATCH * batch_size / DEFAULT_BATCH_SIZE
    spare = psutil.virtual_memory().available - MEMORY_RESERVE_BYTES
    memory_bound = 1 + int(max(0, spare) // per_batch)
    return max(1, min(cpu_bound, memory_bound))


def inference_threads(batch_size: int = DEFAULT_BATCH_SIZE) -> int:
    """The configured concurrency (GLINER_INFERENCE_THREADS), auto-sized when 0."""
    configured = get_cognify_config().gliner_inference_threads
    if configured < 0:
        raise ValueError(f"GLINER_INFERENCE_THREADS must be >= 0, got {configured}")
    return configured or auto_inference_threads(batch_size)


def _inference_pool(batch_size: int) -> ThreadPoolExecutor | None:
    """The process-wide pool that runs model batches, or None for one thread.

    Sized once, on first use, and shared by every pipeline in the process, so
    concurrent documents queue behind one another instead of multiplying the
    memory in flight.
    """
    global _pool, _pool_size
    with _pool_lock:
        if _pool_size is None:
            _pool_size = inference_threads(batch_size)
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
    pool = _inference_pool(batch_size)
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
    pool = _inference_pool(DEFAULT_BATCH_SIZE)
    if pool is not None:
        return pool.submit(run).result()
    with _inference_lock:
        return run()


async def extract_batch_async(
    extractor: Any, texts: Sequence[str], schema: GlinerSchema, **options
):
    return await asyncio.to_thread(extract_batch, extractor, texts, schema, **options)
