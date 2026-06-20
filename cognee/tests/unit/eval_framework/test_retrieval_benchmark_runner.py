import pytest

from cognee.eval_framework.retrieval_benchmark.runner import RetrievalBenchmarkRunner
from cognee.modules.search.types import SearchType


@pytest.mark.asyncio
async def test_runner_aggregates_per_search_type():
    runner = RetrievalBenchmarkRunner(search_types=[SearchType.CHUNKS, SearchType.SUMMARIES])

    async def fake_search(**kwargs):
        if kwargs["query_type"] == SearchType.CHUNKS:
            return [{"context_result": "alpha beta"}]
        return [{"context_result": "alpha"}]

    summary = await runner.run_instances(
        instances=[
            {
                "question": "What is alpha?",
                "golden_context": "alpha beta",
            }
        ],
        search_fn=fake_search,
    )

    assert "CHUNKS" in summary["per_search_type"]
    assert "SUMMARIES" in summary["per_search_type"]
    assert summary["per_search_type"]["CHUNKS"]["context_recall"]["mean"] == 1.0
    assert summary["per_search_type"]["SUMMARIES"]["context_recall"]["mean"] < 1.0
