import cognee
import logging
from typing import Optional

from evals.eval_framework.benchmark_adapters.benchmark_adapters import BenchmarkAdapter
from cognee.shared.utils import setup_logging


class CorpusBuilderExecutor:
    benchmark_adapter = None
    raw_corpus = None
    questions = None

    async def build_corpus(self, limit: Optional[int] = None, benchmark="Dummy"):
        try:
            adapter_enum = BenchmarkAdapter(benchmark)
        except ValueError:
            raise ValueError(f"Unsupported benchmark: {benchmark}")

        self.adapter = adapter_enum.adapter_class()
        self.raw_corpus, self.questions = self.adapter.load_corpus(limit=limit)

        await self.run_cognee()

        return self.questions

    async def run_cognee(self):
        setup_logging(logging.ERROR)

        # Pruning system and databases
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)

        # Adding corpus elements to cognee metastore
        await cognee.add(self.raw_corpus)
        # Running cognify and building knowledge graph
        await cognee.cognify()
