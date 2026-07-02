"""Benchmark: "as of T" filtered read vs live read (issue #3650, Approach 1).

Approach 1 stores no copies — a version is a filter over run provenance. The
trade-off to prove is the cost of that filter: an as-of read scans the
artifacts' ``source_run_refs`` (one graph pass) and intersects them with the
allowed run-id set, instead of reading the live graph directly.

This script seeds N completed runs x M nodes each into a real Ladybug graph
(offline: no LLM, no embeddings, no relational DB — the allowed-run set is
computed in memory; in production it is one indexed query over
``pipeline_runs``) and reports, per history depth:

- live read:   full graph node scan (the baseline every reader pays),
- as-of read:  provenance scan + run-set filter (the versioning surcharge).

Run:  python examples/python/versioning_as_of_benchmark.py
"""

import asyncio
import statistics
import tempfile
import time
from uuid import uuid4

from cognee.infrastructure.databases.graph.ladybug.adapter import LadybugAdapter
from cognee.infrastructure.databases.provenance import make_source_ref_key
from cognee.infrastructure.engine import DataPoint
from cognee.modules.versioning.methods.as_of_read import _is_visible

RUN_COUNTS = [5, 10, 25, 50]
NODES_PER_RUN = 40
REPEATS = 5


class BenchNode(DataPoint):
    name: str
    metadata: dict = {"index_fields": ["name"]}


async def _seed(graph, dataset_id, run_count: int):
    run_ids = []
    for run_index in range(run_count):
        run_id = uuid4()
        run_ids.append(str(run_id))
        source_ref_key = make_source_ref_key(dataset_id, uuid4())
        nodes = [BenchNode(name=f"run{run_index}-node{i}") for i in range(NODES_PER_RUN)]
        await graph.add_nodes(nodes, source_ref_key=source_ref_key, pipeline_run_id=str(run_id))
    return run_ids


async def _time(coro_factory, repeats: int = REPEATS) -> float:
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        await coro_factory()
        samples.append(time.perf_counter() - started)
    return statistics.median(samples)


async def bench(run_count: int) -> tuple[float, float, int]:
    dataset_id = uuid4()
    with tempfile.TemporaryDirectory() as tmp:
        graph = LadybugAdapter(f"{tmp}/graph")
        try:
            run_ids = await _seed(graph, dataset_id, run_count)
            # "as of T" = first half of the history is visible.
            allowed = set(run_ids[: max(1, run_count // 2)])
            dataset_id_str = str(dataset_id)

            async def live_read():
                await graph.query("MATCH (n:Node) RETURN n.id, n.name, n.type", {})

            async def as_of_read():
                refs_by_node = await graph.find_all_node_source_run_refs()
                return {
                    node_id
                    for node_id, run_refs in refs_by_node.items()
                    if _is_visible(run_refs, allowed, dataset_id_str)
                }

            live = await _time(live_read)
            as_of = await _time(as_of_read)
            visible = len(await as_of_read())
            return live, as_of, visible
        finally:
            await graph.close()


async def main():
    total_nodes = {n: n * NODES_PER_RUN for n in RUN_COUNTS}
    print(
        f"{'runs':>5} {'nodes':>7} {'live read':>12} {'as-of read':>12} {'overhead':>9} {'visible':>8}"
    )
    for run_count in RUN_COUNTS:
        live, as_of, visible = await bench(run_count)
        overhead = as_of / live if live else float("inf")
        print(
            f"{run_count:>5} {total_nodes[run_count]:>7} "
            f"{live * 1000:>10.1f}ms {as_of * 1000:>10.1f}ms {overhead:>8.2f}x {visible:>8}"
        )


if __name__ == "__main__":
    asyncio.run(main())
