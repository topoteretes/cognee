import cognee
import logging
from typing import Optional, Tuple, List, Dict

from evals.eval_framework.benchmark_adapters.benchmark_adapters import BenchmarkAdapter
from cognee.shared.utils import setup_logging


class CorpusBuilderExecutor:
    benchmark_adapter = None
    raw_corpus = None
    questions = None

    def load_corpus(
        self, limit: Optional[int] = None, benchmark: str = "Dummy"
    ) -> Tuple[List[Dict], List[str]]:
        try:
            adapter_enum = BenchmarkAdapter(benchmark)
        except ValueError:
            raise ValueError(f"Unsupported benchmark: {benchmark}")

        self.adapter = adapter_enum.adapter_class()
        self.raw_corpus, self.questions = self.adapter.load_corpus(limit=limit)

        return self.raw_corpus, self.questions

    async def build_corpus(self, limit: Optional[int] = None, benchmark: str = "Dummy"):
        self.load_corpus(limit=limit, benchmark=benchmark)
        await self.run_cognee()
        return self.questions

    async def run_cognee(self) -> None:
        setup_logging(logging.ERROR)

        # Pruning system and databases.
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)

        # Adding corpus elements to the cognee metastore.
        await cognee.add(self.raw_corpus)

        # Running cognify to build the knowledge graph.
        await cognee.cognify()
