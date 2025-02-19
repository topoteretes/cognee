import pytest
from evals.eval_framework.corpus_builder.corpus_builder_executor import CorpusBuilderExecutor
from cognee.infrastructure.databases.graph import get_graph_engine

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
async def test_corpus_builder_build_corpus(benchmark):
    limit = 2
    corpus_builder = CorpusBuilderExecutor(benchmark, "Default")
    questions = await corpus_builder.build_corpus(limit=limit)
    assert len(questions) <= 2, (
        f"Corpus builder loads {len(questions)} for {benchmark} when limit is {limit}"
    )
    graph_engine = await get_graph_engine()
    nodes, edges = await graph_engine.get_graph_data()
    assert len(nodes) > 0, f"Corpus builder builds empty graph for {benchmark}"
    assert len(edges) > 0, f"Corpus builder builds graph with no edges for {benchmark}"
