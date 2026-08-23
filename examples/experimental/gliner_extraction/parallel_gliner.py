"""Speed layer for GLiNER extraction: text packing + multi-process workers.

Two independent optimizations, composable:

1. **Packing** — cognee's paragraph chunker emits many small chunks (~1k
   chars against a 512-token budget), and the pipeline delivers them in
   small groups. Packing concatenates consecutive chunk texts up to
   `target_chars` per GLiNER input and remaps the returned spans back to the
   source chunks, so each forward pass carries content instead of padding.
   Relations whose head and tail land in different source chunks are dropped
   (they were never extractable before packing either).

2. **Worker pool** — a ProcessPoolExecutor with one GLiNER model per worker
   (~2 GB RAM each, loaded once at first use). Torch intra-op threads are
   capped per worker to avoid oversubscribing the CPU. Extraction scales
   close to linearly with workers.
"""

import asyncio
import os
import pathlib
from bisect import bisect_right
from concurrent.futures import ProcessPoolExecutor

_SEPARATOR = "\n\n"


def ensure_model_available(model_name: str = "fastino/gliner2-base-v1") -> bool:
    """Make sure the GLiNER checkpoint is in the local HuggingFace cache.

    Announces the download when one is needed (first run only, ~450 MB for
    gliner2-base) instead of silently blocking, and pre-downloads in the
    calling process so multiple worker processes never race the download.
    Returns True when a download happened.
    """
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(model_name, local_files_only=True)
        return False
    except Exception:
        pass

    print(
        f"⬇ Downloading GLiNER2 model '{model_name}' from HuggingFace "
        "(~450 MB for gliner2-base-v1, one-time; cached under ~/.cache/huggingface)...",
        flush=True,
    )
    snapshot_download(model_name)
    print(f"✓ Model '{model_name}' downloaded.", flush=True)
    return True


# --- worker process side -----------------------------------------------------

_worker_model = None


def _init_worker(model_name: str, torch_threads: int):
    global _worker_model
    import torch

    torch.set_num_threads(torch_threads)
    from gliner2 import GLiNER2

    _worker_model = GLiNER2.from_pretrained(model_name)


def _worker_extract(texts, entity_types, relation_types, threshold, batch_size):
    schema = _worker_model.create_schema().entities(entity_types).relations(relation_types)
    return _worker_model.batch_extract(
        texts, schema, batch_size=batch_size, threshold=threshold, include_spans=True
    )


# --- packing ------------------------------------------------------------------


def pack_texts(texts: list[str], target_chars: int = 1800):
    """Greedily pack consecutive texts into inputs of about target_chars.

    Returns (packed_texts, packs) where packs[i] is a list of
    (source_index, start_offset, end_offset) segments inside packed_texts[i].
    """
    packed_texts, packs = [], []
    current, segments, offset = [], [], 0

    for index, text in enumerate(texts):
        if current and offset + len(text) > target_chars:
            packed_texts.append(_SEPARATOR.join(current))
            packs.append(segments)
            current, segments, offset = [], [], 0
        segments.append((index, offset, offset + len(text)))
        current.append(text)
        offset += len(text) + len(_SEPARATOR)

    if current:
        packed_texts.append(_SEPARATOR.join(current))
        packs.append(segments)

    return packed_texts, packs


def _locate(segments, span_start):
    """Find the segment containing a span start; None if it falls between."""
    starts = [seg[1] for seg in segments]
    position = bisect_right(starts, span_start) - 1
    if position < 0:
        return None
    index, start, end = segments[position]
    return (index, start) if span_start < end else None


def unpack_results(packed_results: list[dict], packs, num_texts: int) -> list[dict]:
    """Map packed extraction results back to one result per source text."""
    results = [{"entities": {}, "relation_extraction": {}} for _ in range(num_texts)]

    for result, segments in zip(packed_results, packs):
        for entity_type, values in result.get("entities", {}).items():
            for value in values:
                located = _locate(segments, value["start"])
                if located is None:
                    continue
                index, segment_start = located
                adjusted = {
                    **value,
                    "start": value["start"] - segment_start,
                    "end": value["end"] - segment_start,
                }
                results[index]["entities"].setdefault(entity_type, []).append(adjusted)

        for relation, pairs in result.get("relation_extraction", {}).items():
            for pair in pairs:
                if not isinstance(pair, dict):
                    continue
                head_loc = _locate(segments, pair["head"]["start"])
                tail_loc = _locate(segments, pair["tail"]["start"])
                if head_loc is None or tail_loc is None or head_loc[0] != tail_loc[0]:
                    continue  # cross-chunk relation: not attributable to one chunk
                index, segment_start = head_loc
                adjusted = {
                    "head": {
                        **pair["head"],
                        "start": pair["head"]["start"] - segment_start,
                        "end": pair["head"]["end"] - segment_start,
                    },
                    "tail": {
                        **pair["tail"],
                        "start": pair["tail"]["start"] - segment_start,
                        "end": pair["tail"]["end"] - segment_start,
                    },
                }
                results[index]["relation_extraction"].setdefault(relation, []).append(adjusted)

    return results


# --- pool ---------------------------------------------------------------------


class GLiNERWorkerPool:
    """Multi-process GLiNER extraction with per-worker models."""

    def __init__(
        self,
        model_name: str = "fastino/gliner2-base-v1",
        workers: int = 3,
        torch_threads_per_worker: int | None = None,
    ):
        cpu_count = os.cpu_count() or 4
        threads = torch_threads_per_worker or max(1, cpu_count // workers)
        # Download once in the parent (with a visible notice) so N workers
        # load from cache instead of racing the download.
        ensure_model_available(model_name)
        # Workers import this module by path; make sure spawn can find it.
        module_dir = str(pathlib.Path(__file__).parent)
        existing = os.environ.get("PYTHONPATH", "")
        if module_dir not in existing.split(os.pathsep):
            os.environ["PYTHONPATH"] = module_dir + (os.pathsep + existing if existing else "")
        self.workers = workers
        self._executor = ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(model_name, threads),
        )

    async def extract(
        self,
        texts: list[str],
        entity_types: dict | list,
        relation_types: dict | list,
        threshold: float = 0.5,
        batch_size: int = 32,
    ) -> list[dict]:
        """Extract over texts, sharded across worker processes; order preserved."""
        if not texts:
            return []
        shard_count = min(self.workers, len(texts))
        shards = [texts[i::shard_count] for i in range(shard_count)]
        loop = asyncio.get_running_loop()
        shard_results = await asyncio.gather(
            *[
                loop.run_in_executor(
                    self._executor,
                    _worker_extract,
                    shard,
                    entity_types,
                    relation_types,
                    threshold,
                    batch_size,
                )
                for shard in shards
            ]
        )
        # Reassemble round-robin sharding back into input order.
        results = [None] * len(texts)
        for shard_index, shard_result in enumerate(shard_results):
            for position, result in enumerate(shard_result):
                results[shard_index + position * shard_count] = result
        return results

    def shutdown(self):
        self._executor.shutdown(wait=False, cancel_futures=True)
