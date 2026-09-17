"""Phase 3 of the perf harness must not report SUCCESS for work it did not do.

On a failed cognify the graph is empty; the searches return fast without
raising, status stays "success", and a real-looking float is recorded. Job
100109026857 (nightly 33585616743): three 402s on cognify, each followed by
'Search GRAPH_COMPLETION : 2.30s [success]'. Those six numbers became the
p50/p90/p99 uploaded to S3 and posted to Slack.
"""

import importlib.util
from pathlib import Path

import pytest

import cognee

_BENCH_PATH = (
    Path(__file__).resolve().parents[1]
    / "performance"
    / "statistics_percentile"
    / "bench_cognee.py"
)
_SPEC = importlib.util.spec_from_file_location("bench_cognee_skip", _BENCH_PATH)
assert _SPEC is not None and _SPEC.loader is not None
bench = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bench)

_402 = (
    'Remote cognify failed (402): {"detail":"Insufficient credits to run cognify. '
    'Only $10.00 of credits remain. Add credits and try again.",'
    '"reason":"insufficient_credits","operation":"cognify","remaining_usd":10.0}'
)


class FakeCloudClient:
    def __init__(self):
        self.search_calls = []

    async def add(self, *a, **k):
        return None

    async def cognify(self, *a, **k):
        raise RuntimeError(_402)

    async def search(self, *a, **k):
        self.search_calls.append((a, k))
        return []

    async def forget(self, *a, **k):
        return None

    async def close(self):
        return None


@pytest.mark.asyncio
async def test_search_is_skipped_after_a_failed_cognify(monkeypatch):
    fake = FakeCloudClient()

    async def _serve(**k):
        return fake

    # `import cognee` happens inside run_benchmark_cloud, so patch the real module.
    monkeypatch.setattr(cognee, "serve", _serve)
    results = await bench.run_benchmark_cloud(
        [{"title": "t", "content": "c"}],
        config={"tenant_url": "https://unused", "tenant_api_key": "k"},
    )
    assert fake.search_calls == [], "Phase 3 ran against an empty graph after cognify failed"
    status = results["status"]
    assert status["cognify"].startswith("failed:")
    for key, _ in bench.BENCHMARKED_SEARCH_TYPES:
        assert status[f"search_{key}"] == "skipped"
        assert f"search_time_{key}" not in results, "a skipped search must not fabricate a timing"
    assert results["success"] is False
    # elapsed-until-failure is the file's rule, and stays: cognify_time_s is
    # the (short) time the 402 took, not 0.0
    assert results["cognify_time_s"] >= 0.0


@pytest.mark.parametrize(
    "status, expected",
    [
        ({"add": "success", "cognify": "success"}, True),
        ({"add": "success", "cognify": "failed: 402"}, False),
        ({"add": "failed: x", "cognify": "success"}, False),
        ({"add": "skipped", "cognify": "skipped"}, False),
        ({}, False),
    ],
)
def test_ingest_succeeded(status, expected):
    assert bench._ingest_succeeded(status) is expected


def test_skip_search_phase_marks_every_query_and_clears_timings():
    status, t_search = {}, {k: 0.0 for k, _ in bench.BENCHMARKED_SEARCH_TYPES}
    bench._skip_search_phase(status, t_search)
    for key, _ in bench.BENCHMARKED_SEARCH_TYPES:
        assert status[f"search_{key}"] == "skipped"
        assert t_search[key] is None
