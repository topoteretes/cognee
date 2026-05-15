"""
Scenario test: subprocess-backed Kuzu + LanceDB with a tight LRU cache.

Repeats the following ``N`` times, each pair in its own two fresh datasets:
    1. An add → cognify → search cycle on a small inline snippet.
    2. An add → cognify → search cycle on a distinct large public-domain
       text lazily downloaded from Project Gutenberg to the system temp dir.

Total cycles executed = 2 × N. At ``--cycles 20`` that's 40 cycles drawing
from 20 distinct large texts.

After every cycle, prints the RSS of the main process and all of its
children (the subprocess DB workers).

Usage:
    python ./cognee/tests/test_subprocess_rss.py [options]

Options (all have sensible defaults):
    --cycles N             Rounds of (small + large), 1–20 (default: 3).
                           Total cycles executed = 2 × N.
    --lru-cache-size N     DATABASE_MAX_LRU_CACHE_SIZE (default: 2).
    --kuzu-buffer-mb N     Kuzu buffer pool size in MiB (default: 32).
    --kuzu-num-threads N   Max threads for Kuzu queries (default: 1).
    --subprocess, --no-subprocess
                           Toggle the subprocess-backed adapters for both
                           Kuzu and LanceDB. Default: on.

DATABASE_MAX_LRU_CACHE_SIZE must be set before cognee is imported — the
``@closing_lru_cache`` decorator captures it at import time — so argument
parsing and env-var setup happen at module top, before any ``import cognee``.
"""

from __future__ import annotations

import argparse
import os


def _cycles_type(raw: str) -> int:
    try:
        n = int(raw)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--cycles must be an integer, got {raw!r}")
    if not 1 <= n <= 20:
        raise argparse.ArgumentTypeError(f"--cycles must be between 1 and 20 inclusive, got {n}")
    return n


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Subprocess-backed RSS benchmark for Kuzu + LanceDB.",
    )
    parser.add_argument(
        "--cycles",
        type=_cycles_type,
        default=3,
        help=(
            "Rounds of (small + large), 1–20 (default: 3). Each round runs "
            "one small-text cycle followed by one large-text cycle in a fresh "
            "dataset each. Total cycles = 2 × N."
        ),
    )
    parser.add_argument(
        "--lru-cache-size",
        type=int,
        default=2,
        help="DATABASE_MAX_LRU_CACHE_SIZE for adapter LRU caches (default: 2).",
    )
    parser.add_argument(
        "--kuzu-buffer-mb",
        type=int,
        default=32,
        help="Kuzu buffer pool size in MiB (default: 32).",
    )
    parser.add_argument(
        "--kuzu-num-threads",
        type=int,
        default=1,
        help="Max threads used by Kuzu for query execution (default: 1).",
    )
    parser.add_argument(
        "--subprocess",
        dest="subprocess_enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use subprocess-backed adapters for both graph and vector stores. "
            "Disable with --no-subprocess to run everything in the main process "
            "for comparison. Default: on."
        ),
    )
    return parser.parse_args(argv)


# Parse args first so we can set DATABASE_MAX_LRU_CACHE_SIZE BEFORE importing
# cognee. The LRU cache decorator reads it at class-definition time; setting
# the env var after ``import cognee`` has no effect.
ARGS = _parse_args()
os.environ["DATABASE_MAX_LRU_CACHE_SIZE"] = str(ARGS.lru_cache_size)

import asyncio  # noqa: E402
import gc  # noqa: E402
import pathlib  # noqa: E402
import tempfile  # noqa: E402
import urllib.request  # noqa: E402

import psutil  # noqa: E402

import cognee  # noqa: E402
from cognee.modules.search.types import SearchType  # noqa: E402

from cognee_db_workers.harness import collect_garbage_in_all_workers  # noqa: E402


# Twenty distinct public-domain Gutenberg books (each roughly 400 KB – 1.5 MB).
# One is used per large-round cycle; with ``--cycles 20`` we use all of them.
LARGE_TEXTS: list[tuple[str, str]] = [
    ("pride_and_prejudice.txt", "https://www.gutenberg.org/cache/epub/1342/pg1342.txt"),
    ("frankenstein.txt", "https://www.gutenberg.org/cache/epub/84/pg84.txt"),
    ("sherlock_holmes.txt", "https://www.gutenberg.org/cache/epub/1661/pg1661.txt"),
    ("moby_dick.txt", "https://www.gutenberg.org/cache/epub/2701/pg2701.txt"),
    ("tale_of_two_cities.txt", "https://www.gutenberg.org/cache/epub/98/pg98.txt"),
    ("alice_in_wonderland.txt", "https://www.gutenberg.org/cache/epub/11/pg11.txt"),
    ("dracula.txt", "https://www.gutenberg.org/cache/epub/345/pg345.txt"),
    ("dorian_gray.txt", "https://www.gutenberg.org/cache/epub/174/pg174.txt"),
    ("wuthering_heights.txt", "https://www.gutenberg.org/cache/epub/768/pg768.txt"),
    ("jane_eyre.txt", "https://www.gutenberg.org/cache/epub/1260/pg1260.txt"),
    ("huckleberry_finn.txt", "https://www.gutenberg.org/cache/epub/76/pg76.txt"),
    ("dubliners.txt", "https://www.gutenberg.org/cache/epub/2814/pg2814.txt"),
    ("treasure_island.txt", "https://www.gutenberg.org/cache/epub/120/pg120.txt"),
    ("war_of_the_worlds.txt", "https://www.gutenberg.org/cache/epub/36/pg36.txt"),
    ("metamorphosis.txt", "https://www.gutenberg.org/cache/epub/5200/pg5200.txt"),
    ("little_women.txt", "https://www.gutenberg.org/cache/epub/514/pg514.txt"),
    ("anne_of_green_gables.txt", "https://www.gutenberg.org/cache/epub/45/pg45.txt"),
    ("tom_sawyer.txt", "https://www.gutenberg.org/cache/epub/74/pg74.txt"),
    ("emma.txt", "https://www.gutenberg.org/cache/epub/158/pg158.txt"),
    ("the_iliad.txt", "https://www.gutenberg.org/cache/epub/6130/pg6130.txt"),
]

SMALL_TEXT = (
    "Ada Lovelace worked with Charles Babbage on the Analytical Engine in the 1840s. "
    "She is often credited as the first computer programmer for her notes on the machine."
)

# Lazy-download cache directory. Using the system temp dir means repeated
# runs share one cached copy across the whole machine instead of duplicating
# per-repo-checkout.
LARGE_TEXT_CACHE_DIR = pathlib.Path(tempfile.gettempdir()) / "cognee_subprocess_rss_texts"


def download_text(dest_path: pathlib.Path, url: str) -> str:
    """Lazy downloader: fetch only if the file isn't already cached."""
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return str(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} ...", flush=True)
    urllib.request.urlretrieve(url, str(dest_path))
    print(
        f"  saved {dest_path.stat().st_size / 1024:.1f} KB to {dest_path}",
        flush=True,
    )
    return str(dest_path)


RSS_HISTORY: list[dict] = []


def print_rss(label: str) -> None:
    # Run gc in every live subprocess worker so their RSS reflects reachable
    # objects only (no uncollected cycles). Best-effort; a mid-shutdown or
    # crashed session is skipped silently.
    collect_garbage_in_all_workers()
    # Then gc in the main process so parent RSS is comparable.
    gc.collect()
    # PyArrow's default memory pool is a bump allocator that doesn't give
    # pages back to the OS on its own — every cognify cycle builds pyarrow
    # tables (embeddings, LanceDB writes, …) and the pool keeps growing
    # even after the tables are collected. ``release_unused`` returns any
    # pages not currently backing live allocations, which is essential for
    # accurate per-cycle parent-RSS measurement.
    try:
        import pyarrow as _pa

        _pa.default_memory_pool().release_unused()
    except Exception:
        pass

    proc = psutil.Process(os.getpid())
    parent_mb = proc.memory_info().rss / (1024 * 1024)

    child_entries = []
    total_children_mb = 0.0
    for child in proc.children(recursive=True):
        try:
            rss_mb = child.memory_info().rss / (1024 * 1024)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        total_children_mb += rss_mb
        try:
            name = child.name()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            name = "?"
        try:
            cmdline = " ".join(child.cmdline())
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            cmdline = ""
        child_entries.append((child.pid, name, rss_mb, cmdline))

    total_mb = parent_mb + total_children_mb
    print(f"\n[{label}] RSS summary", flush=True)
    print(f"  parent   pid={proc.pid:<6}  {parent_mb:8.1f} MB", flush=True)
    for pid, name, rss_mb, cmdline in child_entries:
        print(f"  child    pid={pid:<6}  {rss_mb:8.1f} MB  ({name})", flush=True)
        if cmdline:
            print(f"           cmd: {cmdline}", flush=True)
    print(
        f"  total    children={len(child_entries)}  "
        f"children_rss={total_children_mb:.1f} MB  parent+children={total_mb:.1f} MB",
        flush=True,
    )

    RSS_HISTORY.append(
        {
            "label": label,
            "parent_mb": parent_mb,
            "children_mb": total_children_mb,
            "total_mb": total_mb,
            "num_children": len(child_entries),
        }
    )


def print_rss_history() -> None:
    if not RSS_HISTORY:
        return

    print(f"\n{'=' * 78}", flush=True)
    print("Per-cycle memory totals (parent + all children)", flush=True)
    print(f"{'=' * 78}", flush=True)
    print(
        f"{'#':>3}  {'label':<28}  {'parent MB':>10}  {'children MB':>12}  "
        f"{'#ch':>4}  {'total MB':>10}",
        flush=True,
    )
    print(f"{'-' * 3}  {'-' * 28}  {'-' * 10}  {'-' * 12}  {'-' * 4}  {'-' * 10}", flush=True)
    baseline_total = RSS_HISTORY[0]["total_mb"]
    peak_total = max(e["total_mb"] for e in RSS_HISTORY)
    for i, entry in enumerate(RSS_HISTORY):
        print(
            f"{i:>3}  {entry['label'][:28]:<28}  {entry['parent_mb']:>10.1f}  "
            f"{entry['children_mb']:>12.1f}  {entry['num_children']:>4d}  "
            f"{entry['total_mb']:>10.1f}",
            flush=True,
        )

    last_total = RSS_HISTORY[-1]["total_mb"]
    print(f"{'-' * 78}", flush=True)
    print(f"  baseline total : {baseline_total:8.1f} MB", flush=True)
    print(f"  final total    : {last_total:8.1f} MB", flush=True)
    print(f"  peak total     : {peak_total:8.1f} MB", flush=True)
    print(f"  delta (final-baseline): {last_total - baseline_total:+.1f} MB", flush=True)


async def run_cycle(cycle_index: int, total_cycles: int, dataset_name: str, data) -> None:
    print(f"\n{'=' * 60}", flush=True)
    print(
        f"Cycle {cycle_index}/{total_cycles}  —  dataset='{dataset_name}'",
        flush=True,
    )
    print(f"{'=' * 60}", flush=True)

    await cognee.add(data, dataset_name)
    await cognee.cognify([dataset_name])

    results = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION,
        query_text="What is this text about?",
        datasets=[dataset_name],
    )
    print(f"  search returned {len(results)} result(s)", flush=True)

    print_rss(f"after cycle {cycle_index}")


async def main() -> None:
    buffer_pool_bytes = max(1, ARGS.kuzu_buffer_mb) * 1024 * 1024
    rounds = ARGS.cycles
    total_cycles = 2 * rounds

    print(
        f"Running with: rounds={rounds} of (small + large) = {total_cycles} cycles total, "
        f"lru_cache_size={ARGS.lru_cache_size}, "
        f"kuzu_buffer_mb={ARGS.kuzu_buffer_mb}, "
        f"kuzu_num_threads={ARGS.kuzu_num_threads}, "
        f"subprocess={ARGS.subprocess_enabled}",
        flush=True,
    )

    cognee.config.set_graph_db_config(
        {
            "graph_database_provider": "kuzu",
            "graph_database_subprocess_enabled": ARGS.subprocess_enabled,
            "kuzu_num_threads": ARGS.kuzu_num_threads,
            "kuzu_buffer_pool_size": buffer_pool_bytes,
        }
    )
    cognee.config.set_vector_db_config(
        {
            "vector_db_provider": "lancedb",
            "vector_db_subprocess_enabled": ARGS.subprocess_enabled,
        }
    )

    base_dir = pathlib.Path(__file__).parent.resolve()
    cognee.config.data_root_directory(str(base_dir / ".data_storage" / "test_subprocess_rss"))
    cognee.config.system_root_directory(str(base_dir / ".cognee_system" / "test_subprocess_rss"))

    # Lazy-download exactly the N large texts we need — one per round.
    selected_texts = LARGE_TEXTS[:rounds]
    large_text_paths = [
        download_text(LARGE_TEXT_CACHE_DIR / filename, url) for filename, url in selected_texts
    ]

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    print_rss("baseline (after prune)")

    cycle = 0
    for i in range(1, rounds + 1):
        # Small cycle first, then the matching large text for this round,
        # each in its own fresh dataset.
        cycle += 1
        await run_cycle(cycle, total_cycles, f"small_{i}", SMALL_TEXT)

        cycle += 1
        await run_cycle(cycle, total_cycles, f"large_{i}", [large_text_paths[i - 1]])

    print("\nAll cycles complete.", flush=True)

    print_rss_history()


if __name__ == "__main__":
    asyncio.run(main())
