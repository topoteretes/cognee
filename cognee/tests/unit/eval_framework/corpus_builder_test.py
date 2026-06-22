import pytest
from cognee.eval_framework.corpus_builder.corpus_builder_executor import CorpusBuilderExecutor
from unittest.mock import AsyncMock, patch
from cognee.eval_framework.benchmark_adapters.hotpot_qa_adapter import HotpotQAAdapter
from cognee.eval_framework.benchmark_adapters.dummy_adapter import DummyAdapter

benchmark_options = ["HotPotQA", "Dummy", "LogisticsSystem", "TwoWikiMultiHop"]

MOCK_HOTPOT_CORPUS = [
    {
        "_id": "1",
        "question": "Next to which country is Germany located?",
        "answer": "Netherlands",
        # HotpotQA uses "level"; TwoWikiMultiHop uses "type".
        "level": "easy",
        "type": "comparison",
        "context": [
            ["Germany", ["Germany is in Europe."]],
            ["Netherlands", ["The Netherlands borders Germany."]],
        ],
        "supporting_facts": [["Netherlands", 0]],
    }
]


@pytest.mark.parametrize("benchmark", benchmark_options)
def test_corpus_builder_load_corpus(benchmark):
    limit = 2
    if benchmark in ("HotPotQA", "TwoWikiMultiHop"):
        with patch.object(HotpotQAAdapter, "_get_raw_corpus", return_value=MOCK_HOTPOT_CORPUS):
            corpus_builder = CorpusBuilderExecutor(benchmark, "Default")
            raw_corpus, questions = corpus_builder.load_corpus(limit=limit)
    else:
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
    if benchmark in ("HotPotQA", "TwoWikiMultiHop"):
        with patch.object(HotpotQAAdapter, "_get_raw_corpus", return_value=MOCK_HOTPOT_CORPUS):
            corpus_builder = CorpusBuilderExecutor(benchmark, "Default")
            questions = await corpus_builder.build_corpus(limit=limit)
    else:
        corpus_builder = CorpusBuilderExecutor(benchmark, "Default")
        questions = await corpus_builder.build_corpus(limit=limit)

    assert len(questions) <= 2, (
        f"Corpus builder loads {len(questions)} for {benchmark} when limit is {limit}"
    )


@pytest.mark.asyncio
@patch.object(CorpusBuilderExecutor, "run_cognee", new_callable=AsyncMock)
async def test_corpus_builder_prepares_adapter_before_loading(mock_run_cognee):
    adapter = DummyAdapter()
    adapter.prepare_corpus = AsyncMock()

    corpus_builder = CorpusBuilderExecutor(adapter, "Default")
    await corpus_builder.build_corpus(limit=1)

    adapter.prepare_corpus.assert_awaited_once()
