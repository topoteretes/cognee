"""Phase 0 benchmark + profiling harness for DLT ingestion (#3626).

Profiles the two pure DLT ingestion functions directly, with **no LLM key** and
without touching the vector/graph engines (so nothing calls an embedding/LLM API):

  * ``resolve_dlt_sources``  -> dlt load + schema extract + row read-back +
                               id resolution + DataItem build
  * ``ingest_data``          -> the add path: per-row storage + classify + relational write

(Calling ``cognee.add`` instead would trigger ``setup()``/vector init, which probes
the embedding model and blocks on a real API call — irrelevant to ingestion perf.)

Reports per stage: wall-clock, rows/sec, peak memory (tracemalloc), plus a cProfile
attribution of cognee-side time to the real hotspots, and the default
``max_rows_per_table`` cap effect.

Run (no API key needed):
    python -m cognee.tests.performance.benchmark_dlt_ingestion
    python -m cognee.tests.performance.benchmark_dlt_ingestion 200 1000 3000
"""

import os
import asyncio
import cProfile
import io
import pstats
import sys
import time
import tracemalloc
import uuid

os.environ.setdefault("ENABLE_BACKEND_ACCESS_CONTROL", "false")
os.environ.setdefault("REQUIRE_AUTHENTICATION", "false")
os.environ.setdefault("TELEMETRY_DISABLED", "1")
os.environ.setdefault("LLM_API_KEY", "sk-noop-not-used")

# Isolate each process with its own relational DB file so every run is a clean
# first-ingest (cognee persists datasets; reusing one DB across runs collides on
# the default dataset's identity in the session).
os.environ["DB_NAME"] = "bench_" + os.urandom(4).hex()

import dlt  # noqa: E402  (pip install cognee[dlt])
import cognee  # noqa: E402,F401
from cognee.tasks.ingestion.resolve_dlt_sources import resolve_dlt_sources  # noqa: E402
from cognee.tasks.ingestion import ingest_data  # noqa: E402
from cognee.modules.users.methods import get_default_user  # noqa: E402

N_DEPTS = 10
UNCAPPED = 10_000_000


def make_resources(n_rows: int):
    @dlt.resource(name="departments", primary_key="id")
    def departments():
        for i in range(N_DEPTS):
            yield {"id": i, "name": f"dept_{i}", "budget": i * 1000}

    @dlt.resource(name="users", primary_key="id")
    def users():
        for i in range(n_rows):
            yield {
                "id": i,
                "name": f"user_{i}",
                "email": f"user_{i}@example.com",
                "dept_id": i % N_DEPTS,
                "bio": f"Synthetic user {i} for DLT ingestion benchmarking.",
            }

    return [departments(), users()]


async def run_once(user, n_rows: int, max_rows: int, label: str):
    ds = f"bench_{label}_{uuid.uuid4().hex[:8]}"
    resources = make_resources(n_rows)

    tracemalloc.start()
    t0 = time.perf_counter()
    expanded, _cleanup = await resolve_dlt_sources(
        resources, dataset_name=ds, user=user,
        primary_key="id", write_disposition="merge", max_rows_per_table=max_rows,
    )
    t_resolve = time.perf_counter() - t0

    t1 = time.perf_counter()
    await ingest_data(expanded, ds, user)
    t_ingest = time.perf_counter() - t1

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return t_resolve, t_ingest, peak / 1e6, len(expanded)


async def main(n_rows: int, max_rows: int, profile: bool):
    # One run per process: cognee holds dataset instances in a session across
    # calls, so multiple add-style runs in one process collide. Drive several
    # sizes by invoking this module once per size.
    # Create relational tables only (no setup()/vector init -> no embedding probe,
    # so this runs with no LLM key). Fresh DB_NAME per process => clean state.
    from cognee.infrastructure.databases.relational import get_relational_engine

    rel = get_relational_engine()
    os.makedirs(os.path.dirname(rel.db_path), exist_ok=True)
    await rel.create_database()
    user = await get_default_user()

    if profile:
        pr = cProfile.Profile()
        pr.enable()
        try:
            tr, ti, peak, ing = await run_once(user, n_rows, max_rows, "prof")
        finally:
            pr.disable()
        print(f"RESULT rows={ing} resolve={tr:.2f} ingest={ti:.2f} "
              f"total={tr + ti:.2f} rps={ing / (tr + ti):.0f} peakMB={peak:.1f}")
        s = io.StringIO()
        pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(25)
        print("--- cProfile top 25 by cumulative time ---")
        started = False
        for line in s.getvalue().splitlines():
            if "ncalls" in line:
                started = True
            if started and line.strip():
                print(line.rstrip()[:160])
        return

    tr, ti, peak, ing = await run_once(user, n_rows, max_rows, f"r{n_rows}")
    print(f"RESULT rows={ing} resolve={tr:.2f} ingest={ti:.2f} "
          f"total={tr + ti:.2f} rps={ing / (tr + ti):.0f} peakMB={peak:.1f}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    mx = int(sys.argv[2]) if len(sys.argv) > 2 else UNCAPPED
    prof = "profile" in sys.argv
    asyncio.run(main(n, mx, prof))
