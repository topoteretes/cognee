"""
Memory leak test for cognee with LanceDB.

Downloads a long text from Project Gutenberg, then runs multiple
add → cognify → search cycles while tracking process memory usage.
Closes the LanceDB connection after each cycle to release session caches.

Usage:
    python ./cognee/tests/test_memory_leak.py [--parts N]
"""

import argparse
import asyncio
import gc
import os
import pathlib
import threading
import time
import urllib.request
from typing import Any, cast

import psutil

import cognee

# The old SubprocessGraphDBWrapper / SubprocessVectorDBWrapper have been
# replaced by in-adapter subprocess proxies that only import the native DB
# library in the child. The adapter itself surfaces subprocess state via the
# ``_session`` attribute.


GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/2600/pg2600.txt"
TEXT_FILENAME = "war_and_peace.txt"
NUM_CYCLES = 100
MEMORY_POLL_INTERVAL_SEC = 2


def download_text(dest_path: str) -> str:
    """Download a long public-domain text if not already cached."""
    if os.path.exists(dest_path):
        return dest_path

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    print(f"Downloading {GUTENBERG_URL} ...")
    urllib.request.urlretrieve(GUTENBERG_URL, dest_path)
    size_mb = os.path.getsize(dest_path) / (1024 * 1024)
    print(f"Downloaded {size_mb:.2f} MB to {dest_path}")
    return dest_path


def split_text_file(source_path: str, num_parts: int, dest_dir: str) -> list[str]:
    """Split a text file into num_parts roughly equal chunks by lines."""
    with open(source_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    os.makedirs(dest_dir, exist_ok=True)
    total = len(lines)
    chunk_size = total // num_parts
    parts = []

    for i in range(num_parts):
        start = i * chunk_size
        end = start + chunk_size if i < num_parts - 1 else total
        part_path = os.path.join(dest_dir, f"part_{i + 1}.txt")
        with open(part_path, "w", encoding="utf-8") as f:
            f.writelines(lines[start:end])
        parts.append(part_path)
        print(f"  Part {i + 1}: lines {start + 1}-{end} ({end - start} lines) → {part_path}")

    return parts


def get_memory_mb() -> float:
    """Force GC and return RSS of the current process in MB."""
    gc.collect()
    proc = psutil.Process(os.getpid())
    return proc.memory_info().rss / (1024 * 1024)


def memory_monitor(stop_event: threading.Event, log: list):
    """Background thread that records memory snapshots."""
    while not stop_event.is_set():
        entry = {
            "time": time.monotonic(),
            "rss_mb": get_memory_mb(),
        }
        log.append(entry)
        stop_event.wait(MEMORY_POLL_INTERVAL_SEC)


def _wrap_graph_method(original, method_name):
    """Wrap a graph engine method with before/after memory logging."""

    async def wrapper(*args, **kwargs):
        before = get_memory_mb()
        result = await original(*args, **kwargs)
        after = get_memory_mb()
        delta = after - before
        sign = "+" if delta >= 0 else ""
        print(f"    [graph] {method_name:20s}  {before:.1f} → {after:.1f} MB  ({sign}{delta:.1f})", flush=True)
        return result

    return wrapper


_all_vector_engines: list = []
_all_graph_engines: list = []


def _graph_provider_from_factory_args(*args, **kwargs) -> str:
    provider = args[0] if args else kwargs.get("graph_database_provider", "")
    return str(provider).lower()


def _vector_provider_from_factory_args(*args, **kwargs) -> str:
    provider = args[0] if args else kwargs.get("vector_db_provider", "")
    return str(provider).lower()


def _force_graph_subprocess_args(*args, **kwargs):
    mutable_args = list(args)
    if mutable_args:
        if len(mutable_args) >= 10:
            mutable_args[9] = True
        else:
            kwargs["graph_database_subprocess_enabled"] = True
    else:
        kwargs["graph_database_subprocess_enabled"] = True
    return tuple(mutable_args), kwargs


def _force_vector_subprocess_args(*args, **kwargs):
    mutable_args = list(args)
    if mutable_args:
        if len(mutable_args) >= 10:
            mutable_args[9] = True
        else:
            kwargs["vector_db_subprocess_enabled"] = True
    else:
        kwargs["vector_db_subprocess_enabled"] = True
    return tuple(mutable_args), kwargs


def _assert_graph_engine_mode(engine: Any, *args, require_subprocess: bool, **kwargs) -> None:
    if require_subprocess and _graph_provider_from_factory_args(*args, **kwargs) == "kuzu":
        if getattr(engine, "_session", None) is None:
            raise AssertionError(
                "Expected Kuzu graph engine to run in a subprocess when --subprocess is enabled"
            )


def _assert_vector_engine_mode(engine: Any, *args, require_subprocess: bool, **kwargs) -> None:
    if require_subprocess and _vector_provider_from_factory_args(*args, **kwargs) == "lancedb":
        if getattr(engine, "_session", None) is None:
            raise AssertionError(
                "Expected LanceDB vector engine to run in a subprocess when --subprocess is enabled"
            )


def install_graph_memory_logging(*, require_subprocess: bool = False):
    """Monkey-patch the _create_graph_engine factory so every engine it creates
    gets memory-logging wrappers on its methods AND is tracked for cleanup.

    Multi-tenant mode creates a separate engine per user+dataset combination,
    so we cannot patch a single instance.  Instead we intercept the factory
    that all code paths funnel through.

    We replace the cached function with a new @lru_cache that wraps the
    original, and clear the old cache to avoid stale instances holding locks.
    """
    import importlib
    from functools import lru_cache

    factory_mod = importlib.import_module("cognee.infrastructure.databases.graph.get_graph_engine")
    factory_mod_any = cast(Any, factory_mod)

    old_cached = factory_mod_any._create_graph_engine
    original_create = old_cached.__wrapped__
    old_cached.cache_clear()

    methods_to_wrap = [
        "add_nodes", "add_edges", "query",
        "get_connections", "get_graph_data",
        "extract_node", "extract_nodes",
    ]

    @lru_cache
    def patched_create(*args, **kwargs):
        if require_subprocess and _graph_provider_from_factory_args(*args, **kwargs) == "kuzu":
            args, kwargs = _force_graph_subprocess_args(*args, **kwargs)

        engine = original_create(*args, **kwargs)
        _assert_graph_engine_mode(
            engine,
            *args,
            require_subprocess=require_subprocess,
            **kwargs,
        )

        _all_graph_engines.append(engine)
        for name in methods_to_wrap:
            if hasattr(engine, name):
                setattr(engine, name, _wrap_graph_method(getattr(engine, name), name))
        print(f"  [graph-patch] patched engine {type(engine).__name__} id={id(engine)}", flush=True)
        return engine

    factory_mod_any._create_graph_engine = patched_create
    print("  [graph-patch] factory installed", flush=True)


def install_vector_engine_tracking(*, require_subprocess: bool = False):
    """Monkey-patch _create_vector_engine so we can track all LanceDB adapter
    instances created by the multi-tenant system.
    """
    import importlib
    from functools import lru_cache

    factory_mod = importlib.import_module(
        "cognee.infrastructure.databases.vector.create_vector_engine"
    )
    factory_mod_any = cast(Any, factory_mod)

    old_cached = factory_mod_any._create_vector_engine
    original_create = old_cached.__wrapped__
    old_cached.cache_clear()

    @lru_cache
    def tracked_create(*args, **kwargs):
        if require_subprocess and _vector_provider_from_factory_args(*args, **kwargs) == "lancedb":
            args, kwargs = _force_vector_subprocess_args(*args, **kwargs)

        engine = original_create(*args, **kwargs)
        _assert_vector_engine_mode(
            engine,
            *args,
            require_subprocess=require_subprocess,
            **kwargs,
        )
        _all_vector_engines.append(engine)
        print(f"  [vector-track] tracking {type(engine).__name__} id={id(engine)}", flush=True)
        return engine

    factory_mod_any._create_vector_engine = tracked_create
    print("  [vector-track] factory installed", flush=True)



async def run_cycle(cycle: int, part_path: str, num_cycles: int):
    """Single add → cognify → search cycle, appending data to the same dataset."""
    dataset_name = "memleak_dataset"

    from cognee.modules.search.types import SearchType

    print(f"\n{'='*60}")
    print(f"Cycle {cycle + 1}/{num_cycles}  —  {os.path.basename(part_path)}")
    print(f"{'='*60}")

    print(f"  before add        — RSS: {get_memory_mb():.1f} MB")
    await cognee.add([part_path], dataset_name)
    print(f"  after add         — RSS: {get_memory_mb():.1f} MB")

    print(f"  before cognify    — RSS: {get_memory_mb():.1f} MB")
    await cognee.cognify([dataset_name])
    print(f"  after cognify     — RSS: {get_memory_mb():.1f} MB")

    print(f"  before search     — RSS: {get_memory_mb():.1f} MB")
    search_results = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION,
        query_text="main characters and events",
    )
    print(f"  after search      — RSS: {get_memory_mb():.1f} MB  (results: {len(search_results)})")

    gc.collect()
    print(f"  after gc          — RSS: {get_memory_mb():.1f} MB")


def print_summary(mem_log: list):
    """Print a table of memory snapshots and overall delta."""
    if not mem_log:
        return

    t0 = mem_log[0]["time"]
    print(f"\n{'='*60}")
    print("Memory usage over time")
    print(f"{'='*60}")
    print(f"{'Elapsed (s)':>12}  {'RSS (MB)':>10}")
    print(f"{'-'*12}  {'-'*10}")
    for entry in mem_log:
        elapsed = entry["time"] - t0
        print(f"{elapsed:>12.1f}  {entry['rss_mb']:>10.1f}")

    start_mb = mem_log[0]["rss_mb"]
    end_mb = mem_log[-1]["rss_mb"]
    peak_mb = max(e["rss_mb"] for e in mem_log)
    print(f"\nStart: {start_mb:.1f} MB")
    print(f"End:   {end_mb:.1f} MB")
    print(f"Peak:  {peak_mb:.1f} MB")
    print(f"Delta: {end_mb - start_mb:+.1f} MB")


async def main(max_parts=None, subprocess_graph=False, subprocess_vector=False):
    # -- configure cognee to use lancedb with isolated directories --
    cognee.config.set_vector_db_config(
        {
            "vector_db_provider": "lancedb",
            "vector_db_subprocess_enabled": subprocess_vector,
        }
    )
    cognee.config.set_graph_db_config(
        {"graph_database_subprocess_enabled": subprocess_graph}
    )

    base_dir = pathlib.Path(__file__).parent.resolve()
    cognee.config.data_root_directory(
        str(base_dir / ".data_storage" / "test_memory_leak")
    )
    cognee.config.system_root_directory(
        str(base_dir / ".cognee_system" / "test_memory_leak")
    )

    # -- download and split text --
    text_file_path = download_text(
        str(base_dir / "test_data" / TEXT_FILENAME)
    )
    parts_dir = str(base_dir / "test_data" / "memory_leak_parts")
    part_paths = split_text_file(text_file_path, NUM_CYCLES, parts_dir)

    # Install tracking before any database access so the test can enforce that
    # local engines are subprocess-backed from the start when requested.
    install_graph_memory_logging(require_subprocess=subprocess_graph)
    install_vector_engine_tracking(require_subprocess=subprocess_vector)

    # -- clean slate --
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    # Explicitly close and clear cached engines created during prune so later
    # cycles start from a clean engine state.
    import importlib as _il

    _gfm = _il.import_module("cognee.infrastructure.databases.graph.get_graph_engine")
    _gfm._create_graph_engine.cache_clear()
    _vfm = _il.import_module("cognee.infrastructure.databases.vector.create_vector_engine")
    _vfm._create_vector_engine.cache_clear()
    gc.collect()

    # -- start memory monitor --
    mem_log: list[dict] = []
    stop_event = threading.Event()
    monitor_thread = threading.Thread(
        target=memory_monitor, args=(stop_event, mem_log), daemon=True
    )
    monitor_thread.start()

    # -- run cycles, each with a different chunk of the book --
    parts_to_run = part_paths[:max_parts] if max_parts is not None else part_paths
    total = len(parts_to_run)
    for cycle, part_path in enumerate(parts_to_run):
        await run_cycle(cycle, part_path, total)

    # -- stop monitor and report --
    stop_event.set()
    monitor_thread.join(timeout=5)

    print_summary(mem_log)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cognee memory leak test")
    parser.add_argument(
        "--parts", type=int, default=None,
        help="Number of parts to process (default: all)",
    )
    parser.add_argument(
        "--subprocess", action="store_true", default=False,
        help="Run KuzuAdapter and LanceDBAdapter against subprocess-backed DB clients",
    )
    args = parser.parse_args()
    asyncio.run(
        main(
            max_parts=args.parts,
            subprocess_graph=args.subprocess,
            subprocess_vector=args.subprocess,
        )
    )
