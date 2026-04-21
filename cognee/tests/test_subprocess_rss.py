"""
Scenario test: subprocess-backed Kuzu + LanceDB with a tight LRU cache.

Runs 9 add → cognify → search cycles, each in its own dataset, in three
phases: 3 cycles on a small inline snippet, then 3 on chunks of a downloaded
public-domain text, then 3 more on the small snippet. After every cycle,
prints the RSS of the main process and all of its children (the subprocess
DB workers).

DATABASE_MAX_LRU_CACHE_SIZE is set to 2 before cognee is imported so the
factory caches in get_graph_engine / create_vector_engine evict aggressively.

Usage:
    python ./cognee/tests/test_subprocess_rss.py
"""

import os

os.environ["DATABASE_MAX_LRU_CACHE_SIZE"] = "2"

import asyncio
import gc
import pathlib
import urllib.request

import psutil

import cognee
from cognee.modules.search.types import SearchType


# One distinct public-domain book per large cycle (each ~400 KB – 1 MB).
LARGE_TEXTS: list[tuple[str, str]] = [
    ("pride_and_prejudice.txt", "https://www.gutenberg.org/cache/epub/1342/pg1342.txt"),
    ("frankenstein.txt", "https://www.gutenberg.org/cache/epub/84/pg84.txt"),
    ("sherlock_holmes.txt", "https://www.gutenberg.org/cache/epub/1661/pg1661.txt"),
]

SMALL_TEXT = (
    "Ada Lovelace worked with Charles Babbage on the Analytical Engine in the 1840s. "
    "She is often credited as the first computer programmer for her notes on the machine."
)

NUM_LARGE_CYCLES = len(LARGE_TEXTS)
NUM_SMALL_CYCLES_PER_PHASE = 3


def download_text(dest_path: str, url: str) -> str:
    if os.path.exists(dest_path):
        return dest_path
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    print(f"Downloading {url} ...", flush=True)
    urllib.request.urlretrieve(url, dest_path)
    print(f"  saved {os.path.getsize(dest_path) / 1024:.1f} KB to {dest_path}", flush=True)
    return dest_path


RSS_HISTORY: list[dict] = []


def print_rss(label: str) -> None:
    gc.collect()
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


TOTAL_CYCLES = NUM_LARGE_CYCLES + 2 * NUM_SMALL_CYCLES_PER_PHASE


async def run_cycle(cycle_index: int, dataset_name: str, data) -> None:
    print(f"\n{'=' * 60}", flush=True)
    print(f"Cycle {cycle_index}/{TOTAL_CYCLES}  —  dataset='{dataset_name}'", flush=True)
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
    cognee.config.set_graph_db_config(
        {
            "graph_database_provider": "kuzu",
            "graph_database_subprocess_enabled": True,
            "kuzu_num_threads": 1,
            "kuzu_buffer_pool_size": 16 * 1024 * 1024,
        }
    )
    cognee.config.set_vector_db_config(
        {
            "vector_db_provider": "lancedb",
            "vector_db_subprocess_enabled": True,
        }
    )

    base_dir = pathlib.Path(__file__).parent.resolve()
    cognee.config.data_root_directory(
        str(base_dir / ".data_storage" / "test_subprocess_rss")
    )
    cognee.config.system_root_directory(
        str(base_dir / ".cognee_system" / "test_subprocess_rss")
    )

    large_text_paths = [
        download_text(str(base_dir / "test_data" / filename), url)
        for filename, url in LARGE_TEXTS
    ]

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    print_rss("baseline (after prune)")

    cycle = 0

    # Phase 1: small text
    for i in range(1, NUM_SMALL_CYCLES_PER_PHASE + 1):
        cycle += 1
        await run_cycle(cycle, f"small_a_{i}", SMALL_TEXT)

    # Phase 2: one distinct large text per cycle
    for i, text_path in enumerate(large_text_paths, start=1):
        cycle += 1
        await run_cycle(cycle, f"large_{i}", [text_path])

    # Phase 3: small text again
    for i in range(1, NUM_SMALL_CYCLES_PER_PHASE + 1):
        cycle += 1
        await run_cycle(cycle, f"small_b_{i}", SMALL_TEXT)

    print("\nAll cycles complete.", flush=True)

    print_rss_history()


if __name__ == "__main__":
    asyncio.run(main())
