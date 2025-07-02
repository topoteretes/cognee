import pytest
from cognee.eval_framework.corpus_builder.corpus_builder_executor import CorpusBuilderExecutor
from cognee.infrastructure.databases.graph import get_graph_engine
from unittest.mock import AsyncMock, patch

benchmark_options = ["HotPotQA", "Dummy", "TwoWikiMultiHop"]


@pytest.mark.parametrize("benchmark", benchmark_options)
def test_corpus_builder_load_corpus(benchmark):
    limit = 2
    corpus_builder = CorpusBuilderExecutor(benchmark, "Default")
    raw_corpus, questions = corpus_builder.load_corpus(limit=limit)
    assert len(raw_corpus) > 0, f"Corpus builder loads empty corpus for {benchmark}"
    assert len(questions) <= 2, (
        f"Corpus builder loads {len(questions)} for {benchmark} when limit is {limit}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("benchmark", benchmark_options)
@patch.object(CorpusBuilderExecutor, "run_cognee", new_callable=AsyncMock)
async def test_corpus_builder_build_corpus(mock_run_cognee, benchmark):
    limit = 2
    corpus_builder = CorpusBuilderExecutor(benchmark, "Default")
    questions = await corpus_builder.build_corpus(limit=limit)
    assert len(questions) <= 2, (
        f"Corpus builder loads {len(questions)} for {benchmark} when limit is {limit}"
    )
