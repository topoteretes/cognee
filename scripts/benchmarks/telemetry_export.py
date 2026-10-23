"""Measure SDK-1168 exporter overhead with synthetic data and no external requests.

Run from the repo root: python scripts/benchmarks/telemetry_export.py --processes 1 4
Each worker imports the SDK, then measures the incremental exporter footprint.
The stress scenario raises only admission budgets to saturate the default queue.
"""

import argparse
import json
import multiprocessing
import os
import sys
import threading
import time
import tracemalloc
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ["TELEMETRY_DISABLED"] = "1"


def measure(args):
    import httpx
    import psutil

    from cognee.shared.telemetry_limits import TelemetryLimits
    from cognee.shared.telemetry_sender import TelemetrySender

    worker_id, seconds = args
    results = []
    for scenario, payload_size, unavailable in (
        ("default-burst", 512, False),
        ("large-payload", 15_000, False),
        ("saturated-outage", 15_000, True),
        ("concurrent-preparation", 32_000, False),
    ):
        limits = TelemetryLimits()
        if unavailable:
            limits = replace(limits, events_per_minute=1_000_000, bytes_per_minute=1_000_000_000)
        requests = 0

        def proxy(request, unavailable=unavailable):
            nonlocal requests
            requests += 1
            return httpx.Response(503 if unavailable else 200)

        sender = TelemetrySender(
            "http://synthetic.invalid", 0.2, httpx.MockTransport(proxy), limits=limits
        )
        concurrent = scenario == "concurrent-preparation"
        if concurrent:
            from cognee.shared import utils

            # Synthetic identity only; never read or transmit the host's credentials.
            os.environ.pop("TELEMETRY_DISABLED", None)
            os.environ["ENV"] = "prod"
            os.environ["TELEMETRY_ORIGIN"] = "sdk"
            os.environ["LLM_API_KEY"] = "synthetic-benchmark-key"
            utils._telemetry_sender = sender
            utils.get_anonymous_id = lambda: "synthetic-project"
            utils.get_persistent_id = lambda: "synthetic-machine"
            utils._derive_api_key_tracking_id.cache_clear()
        process = psutil.Process()
        baseline_rss = process.memory_info().rss
        peak_rss = baseline_rss
        stop = threading.Event()

        def observe(stop=stop, process=process):
            nonlocal peak_rss
            while not stop.wait(0.01):
                peak_rss = max(peak_rss, process.memory_info().rss)

        observer = threading.Thread(target=observe, daemon=True)
        observer.start()
        tracemalloc.start()
        payload = {
            "event_name": "Pipeline Run Completed",
            "properties": {"count": 1, "padding": "x" * payload_size},
        }
        if concurrent:
            # Raw properties exceed the wire cap but shrink before final encoding.
            payload["properties"] = {"url": "https://synthetic.invalid/" + "x" * payload_size}
        thread_count = 4 if concurrent else 1
        barrier = threading.Barrier(thread_count)

        def emit(barrier=barrier, concurrent=concurrent, payload=payload, sender=sender):
            latencies = []
            attempted = 0
            barrier.wait()
            started = time.monotonic()
            while time.monotonic() - started < seconds:
                before = time.monotonic()
                if concurrent:
                    utils.send_telemetry(
                        "cognee.recall", additional_properties=payload["properties"]
                    )
                else:
                    sender.enqueue(payload)
                elapsed = time.monotonic() - before
                # Fixed-size sample per thread; never grows with sustained volume.
                if len(latencies) < 1024:
                    latencies.append(elapsed)
                attempted += 1
            return attempted, latencies

        with ThreadPoolExecutor(max_workers=thread_count) as producers:
            measurements = list(producers.map(lambda _: emit(), range(thread_count)))
        attempted = sum(count for count, _ in measurements)
        latencies = [latency for _, sample in measurements for latency in sample]
        close_started = time.monotonic()
        sender.close()
        close_seconds = time.monotonic() - close_started
        _, peak_allocated = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        stop.set()
        observer.join()
        latencies.sort()
        state = sender.snapshot()
        assert state["peak_queue_bytes"] <= limits.queue_bytes
        assert state["peak_inflight_bytes"] <= 2 * limits.batch_bytes
        results.append(
            {
                "worker": worker_id,
                "scenario": scenario,
                "producer_threads": thread_count,
                "duration_seconds": seconds,
                "attempted": attempted,
                "http_requests": requests,
                "sample_admission_p99_ms": round(latencies[int(len(latencies) * 0.99)] * 1000, 3),
                "close_ms": round(close_seconds * 1000, 3),
                "peak_traced_bytes": peak_allocated,
                "rss_growth_bytes": peak_rss - baseline_rss,
                **state,
            }
        )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processes", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--seconds", type=float, default=5)
    args = parser.parse_args()
    for count in args.processes:
        with ProcessPoolExecutor(
            max_workers=count, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            results = list(pool.map(measure, [(i, args.seconds) for i in range(count)]))
        print(json.dumps({"processes": count, "results": results}), flush=True)


if __name__ == "__main__":
    main()
