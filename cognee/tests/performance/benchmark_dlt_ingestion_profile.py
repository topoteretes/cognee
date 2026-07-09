"""Phase-0 benchmark + profiling harness for the DLT ingestion path (issue #3626).

This is the "profile first" deliverable: a reproducible, no-LLM-key benchmark that
measures **where time and memory actually go** in cognee's DLT ingestion path, with a
per-stage breakdown finer than wall-clock resolve/ingest totals.

It profiles the real ingestion functions directly (no ``cognee.add`` / no vector or
LLM init), because the DLT path never needs an LLM — structured rows bypass extraction:

    resolve_dlt_sources
      ├─ ingest_dlt_source
      │    ├─ dlt pipeline.run   (extract + normalize + load)   -> stage: dlt_run
      │    ├─ _extract_dlt_schema                                -> stage: schema_extract
      │    └─ _read_rows_from_tables (materializes all rows)     -> stage: row_readback
      ├─ Phase 1: get_unique_data_id per row (1 DB session/row)  -> stage: phase1_id_resolve
      └─ Phase 2: build DataItems (schema text + FK resolve)     -> stage: phase2_build
    ingest_data (the add-pipeline storage task)
      ├─ identify -> get_unique_data_id AGAIN per row (result    -> stage: ingest_redundant_id
      │              discarded: DataItem.data_id already set)
      ├─ save_data_item_to_storage / data_item_to_text_file      (per-row file I/O)
      └─ BinaryData.get_metadata (~3x/row, each spawns a thread) -> stage: metadata_threads
    orphan_cleanup (re-ingest only)
      └─ get_dataset_data loads the WHOLE dataset into Python    -> stage: orphan_cleanup

Each ``main()`` run does a **first ingest** and a **re-ingest with deletions**, so the
orphan-cleanup path (a named concern in #3626) is measured, not skipped.

Stage timings are collected by monkeypatching the target functions with thin timing
wrappers and snapshotting per-stage deltas around the resolve / ingest / cleanup phases
(so ``get_unique_data_id`` is attributed to *phase1* vs the *redundant ingest* call
separately). No production code is modified.

Isolation: the process points DATA_ROOT_DIRECTORY / SYSTEM_ROOT_DIRECTORY / DLT_DATA_DIR
at a temp dir and uses a fresh DB_NAME, so runs are hermetic. ``--sizes`` runs the sizes
*sequentially in one process* (each with a unique dataset name) — deliberately not a
subprocess fan-out, because on Windows dlt's parallel normalize/load re-imports __main__
in spawned children and deadlocks. Note cognee hardcodes
``pipeline_name="ingest_dlt_source"`` whose dlt state dir is global, so two *concurrent*
cognee DLT ingests would still collide — a finding, not something the harness relies on.

Usage (no API key needed)::

    # single size, full per-stage breakdown + both runs:
    python -m cognee.tests.performance.benchmark_dlt_ingestion_profile 2000

    # scan several sizes (spawns one isolated subprocess per size) + write JSON report:
    python -m cognee.tests.performance.benchmark_dlt_ingestion_profile --sizes 500,2000,5000

    # add a cProfile attribution pass on top of the stage timers:
    python -m cognee.tests.performance.benchmark_dlt_ingestion_profile 2000 --cprofile
"""

import os
import sys

# --- Environment isolation (must be set before importing cognee) -------------
os.environ.setdefault("ENABLE_BACKEND_ACCESS_CONTROL", "false")
os.environ.setdefault("REQUIRE_AUTHENTICATION", "false")
os.environ.setdefault("TELEMETRY_DISABLED", "1")
os.environ.setdefault("LITELLM_LOG", "ERROR")
os.environ.setdefault("LLM_API_KEY", "sk-noop-not-used-by-ingestion")
# Fresh relational DB per process => every run is a clean first-ingest.
os.environ.setdefault("DB_NAME", "bench_" + os.urandom(4).hex())
# Isolate ALL cognee state (relational + graph + vector dirs) per process, in a
# temp dir. This keeps the repo clean AND prevents lock contention: cognee opens
# the graph DB under SYSTEM_ROOT, so two processes sharing it block on a file
# lock (this is why the --sizes subprocess driver used to deadlock against the
# parent's `import cognee`).
import tempfile as _tempfile  # noqa: E402

_ROOT = _tempfile.mkdtemp(prefix="cognee_bench_" + os.urandom(3).hex() + "_")
os.environ.setdefault("DATA_ROOT_DIRECTORY", os.path.join(_ROOT, "data"))
os.environ.setdefault("SYSTEM_ROOT_DIRECTORY", os.path.join(_ROOT, "system"))
# Per-process dlt working dir => concurrent runs don't clobber dlt pipeline state.
os.environ.setdefault("DLT_DATA_DIR", os.path.join(_ROOT, "dlt"))

import asyncio  # noqa: E402
import cProfile  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import pstats  # noqa: E402
import time  # noqa: E402
import tracemalloc  # noqa: E402
import uuid  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import dlt  # noqa: E402  (pip install cognee[dlt])


N_DEPTS = 10
UNCAPPED = 10_000_000
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


# --------------------------------------------------------------------------- #
# Synthetic source: 3 tables with real FK fan-out so FK resolution is exercised.
#   departments (N_DEPTS) <- users (n_rows) <- orders (2*n_rows)
# --------------------------------------------------------------------------- #
def make_source(n_rows: int, drop_tail_frac: float = 0.0):
    """Build a single 3-table dlt source with FK fan-out.

    Returned as ONE ``@dlt.source`` (not a list of resources) so all tables are
    read back exactly once: passing separate DltResources makes
    ``resolve_dlt_sources`` run ``ingest_dlt_source`` per resource, and each run
    re-reads the whole schema (departments would be read 3x, users 2x, ...).

    ``drop_tail_frac`` removes a fraction of the tail rows of users/orders so a
    re-ingest with write_disposition="replace" produces orphans, exercising
    orphan_cleanup + incremental re-sync.
    """
    n_users = int(n_rows * (1.0 - drop_tail_frac))
    n_orders = int(2 * n_rows * (1.0 - drop_tail_frac))

    @dlt.source(name="bench")
    def bench_source():
        @dlt.resource(name="departments", primary_key="id")
        def departments():
            for i in range(N_DEPTS):
                yield {"id": i, "name": f"dept_{i}", "budget": i * 1000}

        @dlt.resource(name="users", primary_key="id")
        def users():
            for i in range(n_users):
                yield {
                    "id": i,
                    "name": f"user_{i}",
                    "email": f"user_{i}@example.com",
                    "dept_id": i % N_DEPTS,
                    "bio": f"Synthetic user {i} for DLT ingestion benchmarking.",
                }

        @dlt.resource(name="orders", primary_key="id")
        def orders():
            for i in range(n_orders):
                yield {
                    "id": i,
                    "user_id": i % max(n_users, 1),
                    "amount": round(i * 1.5, 2),
                    "status": "open" if i % 3 else "closed",
                }

        return departments, users, orders

    return bench_source()


# --------------------------------------------------------------------------- #
# Stage timing via monkeypatch.
# --------------------------------------------------------------------------- #
class Stages:
    """Accumulates (calls, seconds) per named stage. Snapshot deltas per phase."""

    def __init__(self):
        self.data: dict[str, list] = {}

    def add(self, name: str, seconds: float):
        slot = self.data.setdefault(name, [0, 0.0])
        slot[0] += 1
        slot[1] += seconds

    def snapshot(self) -> dict:
        return {k: (v[0], v[1]) for k, v in self.data.items()}

    def delta(self, before: dict) -> dict:
        out = {}
        for k, (calls, secs) in self.data.items():
            b_calls, b_secs = before.get(k, (0, 0.0))
            dc, ds = calls - b_calls, secs - b_secs
            if dc:
                out[k] = {"calls": dc, "seconds": round(ds, 4)}
        return out


STAGES = Stages()


def _wrap_async(mod, attr, stage_name):
    orig = getattr(mod, attr)

    async def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return await orig(*args, **kwargs)
        finally:
            STAGES.add(stage_name, time.perf_counter() - t0)

    setattr(mod, attr, wrapper)
    return orig


def _wrap_sync(mod_or_cls, attr, stage_name):
    orig = getattr(mod_or_cls, attr)

    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return orig(*args, **kwargs)
        finally:
            STAGES.add(stage_name, time.perf_counter() - t0)

    setattr(mod_or_cls, attr, wrapper)
    return orig


@contextmanager
def patch_timers():
    """Install per-stage timing wrappers, restore on exit."""
    # Use `import ... as` to bind the actual *module* objects; the ingestion
    # package __init__ re-exports functions of the same name, which would shadow
    # the submodules under `from ... import name`.
    # importlib.import_module returns the true submodule from sys.modules; a plain
    # `import a.b.c as x` would bind x to the parent package attribute, which the
    # ingestion __init__ has overwritten with the same-named function.
    import importlib

    rds = importlib.import_module("cognee.tasks.ingestion.resolve_dlt_sources")
    ids = importlib.import_module("cognee.tasks.ingestion.ingest_dlt_source")
    bd = importlib.import_module("cognee.modules.ingestion.data_types.BinaryData")
    idf = importlib.import_module("cognee.modules.ingestion.identify")

    saved = []
    # get_unique_data_id is imported by-name into two modules, so patch both
    # bindings. The Phase-1 call lives in resolve_dlt_sources; the *redundant*
    # per-row call inside ingest_data goes through identify's binding. Same stage
    # name -> the phase-window delta (see run_ingest) separates them: resolve
    # window = Phase 1, ingest window = the discarded identify() re-resolution.
    saved.append((rds, "get_unique_data_id", _wrap_async(rds, "get_unique_data_id", "id_resolve")))
    saved.append((idf, "get_unique_data_id", _wrap_async(idf, "get_unique_data_id", "id_resolve")))
    saved.append((rds, "ingest_dlt_source", _wrap_async(rds, "ingest_dlt_source", "dlt_ingest_total")))
    saved.append((ids, "_extract_dlt_schema", _wrap_async(ids, "_extract_dlt_schema", "schema_extract")))
    saved.append((ids, "_read_rows_from_tables", _wrap_async(ids, "_read_rows_from_tables", "row_readback")))
    saved.append((bd.BinaryData, "get_metadata", _wrap_sync(bd.BinaryData, "get_metadata", "metadata_threads")))
    try:
        yield
    finally:
        for obj, attr, orig in saved:
            setattr(obj, attr, orig)


# --------------------------------------------------------------------------- #
# One ingest run (resolve + ingest_data + optional orphan cleanup), instrumented.
# --------------------------------------------------------------------------- #
async def run_ingest(user, ds, resources, write_disposition, max_rows):
    from cognee.tasks.ingestion.resolve_dlt_sources import resolve_dlt_sources
    from cognee.tasks.ingestion import ingest_data

    # Measure *incremental* peak so multi-size --sizes runs aren't contaminated by
    # heap still alive from earlier sizes: report peak-above-the-start-baseline.
    import gc

    gc.collect()
    tracemalloc.start()
    tracemalloc.reset_peak()
    base_cur, _ = tracemalloc.get_traced_memory()

    snap0 = STAGES.snapshot()
    t0 = time.perf_counter()
    expanded, cleanup = await resolve_dlt_sources(
        resources, dataset_name=ds, user=user,
        primary_key="id", write_disposition=write_disposition, max_rows_per_table=max_rows,
    )
    t_resolve = time.perf_counter() - t0
    resolve_stages = STAGES.delta(snap0)
    _, peak_resolve = tracemalloc.get_traced_memory()

    snap1 = STAGES.snapshot()
    t1 = time.perf_counter()
    await ingest_data(expanded, ds, user)
    t_ingest = time.perf_counter() - t1
    ingest_stages = STAGES.delta(snap1)

    t_cleanup = 0.0
    cleanup_stages = {}
    if cleanup is not None:
        snap2 = STAGES.snapshot()
        t2 = time.perf_counter()
        await cleanup()
        t_cleanup = time.perf_counter() - t2
        cleanup_stages = STAGES.delta(snap2)

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    n_items = len(expanded) if isinstance(expanded, list) else 1
    return {
        "n_dataitems": n_items,
        "t_resolve": round(t_resolve, 3),
        "t_ingest": round(t_ingest, 3),
        "t_cleanup": round(t_cleanup, 3),
        "t_total": round(t_resolve + t_ingest + t_cleanup, 3),
        "rows_per_sec": round(n_items / (t_resolve + t_ingest + t_cleanup), 1),
        "peak_mem_mb": round(max(peak - base_cur, 0) / 1e6, 1),
        "peak_mem_resolve_mb": round(max(peak_resolve - base_cur, 0) / 1e6, 1),
        "stage_breakdown": {
            "resolve": resolve_stages,
            "ingest_data": ingest_stages,
            "orphan_cleanup": cleanup_stages,
        },
    }


async def prepare_db_and_user():
    """Create relational tables + default user without setup()/vector init."""
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.users.methods import get_default_user

    rel = get_relational_engine()
    if getattr(rel, "db_path", None):
        os.makedirs(os.path.dirname(rel.db_path), exist_ok=True)
    await rel.create_database()
    return await get_default_user()


async def bench_one_size(n_rows: int, max_rows: int, cprofile: bool) -> dict:
    user = await prepare_db_and_user()
    ds = f"bench_{n_rows}_{uuid.uuid4().hex[:8]}"

    with patch_timers():
        pr = cProfile.Profile() if cprofile else None
        if pr:
            pr.enable()

        # Run 1: first ingest (all rows).
        first = await run_ingest(user, ds, make_source(n_rows), "replace", max_rows)

        # Run 2: re-ingest with the tail 20% dropped -> orphans + incremental re-sync.
        reingest = await run_ingest(
            user, ds, make_source(n_rows, drop_tail_frac=0.2), "replace", max_rows
        )

        if pr:
            pr.disable()

    result = {
        "n_rows_config": n_rows,
        "max_rows_per_table": None if max_rows >= UNCAPPED else max_rows,
        "first_ingest": first,
        "reingest_with_deletions": reingest,
    }
    if pr:
        result["cprofile_top"] = _cprofile_top(pr, 20)
    return result


def _cprofile_top(pr: cProfile.Profile, n: int) -> list:
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(n)
    rows = []
    started = False
    for line in s.getvalue().splitlines():
        if "ncalls" in line:
            started = True
            continue
        if started and line.strip():
            rows.append(line.strip()[:150])
    return rows[:n]


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _fmt_stage_table(title: str, stages: dict) -> str:
    lines = [f"  {title}:"]
    flat = []
    for phase, d in stages.items():
        for stage, m in d.items():
            flat.append((f"{phase}/{stage}", m["calls"], m["seconds"]))
    flat.sort(key=lambda x: -x[2])
    for name, calls, secs in flat:
        lines.append(f"    {name:<34} {calls:>8} calls  {secs:>8.3f}s")
    return "\n".join(lines)


def print_run(label: str, r: dict):
    print(
        f"  [{label}] items={r['n_dataitems']} "
        f"resolve={r['t_resolve']}s ingest={r['t_ingest']}s cleanup={r['t_cleanup']}s "
        f"total={r['t_total']}s  {r['rows_per_sec']} items/s  peak={r['peak_mem_mb']}MB"
    )
    print(_fmt_stage_table("stage breakdown (seconds)", r["stage_breakdown"]))


def print_size_report(res: dict):
    cap = res["max_rows_per_table"]
    print(f"\n=== size={res['n_rows_config']} rows/table  (cap={cap or 'uncapped'}) ===")
    print_run("first ingest", res["first_ingest"])
    print_run("re-ingest (tail 20% dropped)", res["reingest_with_deletions"])
    if "cprofile_top" in res:
        print("  cProfile top-cumulative:")
        for line in res["cprofile_top"]:
            print(f"    {line}")


# --------------------------------------------------------------------------- #
# Multi-size driver (in-process, sequential)
# --------------------------------------------------------------------------- #
async def drive_sizes(sizes: list, max_rows: int):
    """Run each size sequentially in one process.

    Each size uses a unique dataset name (many datasets in one relational DB is
    normal cognee usage), so runs don't collide. This is intentionally NOT a
    subprocess fan-out: on Windows, spawning the benchmark as a child while dlt
    parallelizes normalize/load re-imports __main__ and deadlocks. Sequential
    in-process is robust and the numbers are process-warm-cache comparable.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    collected = []
    for n in sizes:
        print(f"\n>>> running size={n} ...", flush=True)
        res = await bench_one_size(n, max_rows, cprofile=False)
        collected.append(res)
        print_size_report(res)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "note": "Phase-0 DLT ingestion baseline (#3626). No LLM. SQLite destination. "
        "Sizes run sequentially in one process, each with a unique dataset name.",
        "results": collected,
    }
    out = os.path.join(RESULTS_DIR, "dlt_ingestion_baseline.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {out}")
    _print_scaling_summary(collected)


def _print_scaling_summary(results: list):
    print("\n=== scaling summary (first ingest) ===")
    print(f"  {'rows':>8} {'items':>8} {'total_s':>9} {'items/s':>9} {'peakMB':>8}")
    for r in results:
        fi = r["first_ingest"]
        print(f"  {r['n_rows_config']:>8} {fi['n_dataitems']:>8} "
              f"{fi['t_total']:>9} {fi['rows_per_sec']:>9} {fi['peak_mem_mb']:>8}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def _parse_args(argv):
    sizes = None
    n = 1000
    max_rows = UNCAPPED
    cprofile = "--cprofile" in argv
    for i, a in enumerate(argv):
        if a == "--sizes" and i + 1 < len(argv):
            sizes = [int(x) for x in argv[i + 1].split(",") if x.strip()]
        elif a == "--max-rows" and i + 1 < len(argv):
            max_rows = int(argv[i + 1])
        elif a.isdigit():
            n = int(a)
    return sizes, n, max_rows, cprofile


async def _amain(n, max_rows, cprofile):
    res = await bench_one_size(n, max_rows, cprofile)
    if os.environ.get("_BENCH_EMIT_JSON"):
        print("JSONRESULT " + json.dumps(res))
    else:
        print_size_report(res)


if __name__ == "__main__":
    sizes, n, max_rows, cprofile = _parse_args(sys.argv[1:])
    if sizes:
        asyncio.run(drive_sizes(sizes, max_rows))
    else:
        asyncio.run(_amain(n, max_rows, cprofile))
