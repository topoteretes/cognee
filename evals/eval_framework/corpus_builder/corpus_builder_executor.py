import cognee
import logging
from typing import Optional

from evals.eval_framework.benchmark_adapters.hotpot_qa_adapter import HotpotQAAdapter
from evals.eval_framework.benchmark_adapters.twowikimultihop_adapter import TwoWikiMultihopAdapter
from evals.eval_framework.benchmark_adapters.dummy_adapter import DummyAdapter
from cognee.shared.utils import setup_logging


class CorpusBuilderExecutor:
    benchmark_adapter_options = {
        "Dummy": DummyAdapter,
        "HotPotQA": HotpotQAAdapter,
        "TwoWikiMultiHop": TwoWikiMultihopAdapter,
    }

    benchmark_adapter = None
    raw_corpus = None
    questions = None

    async def build_corpus(self, limit: Optional[int] = None, benchmark="Dummy"):
        if benchmark not in self.benchmark_adapter_options:
            raise ValueError(f"Unsupported benchmark: {benchmark}")

        self.adapter = self.benchmark_adapter_options[benchmark]()
        self.raw_corpus, self.questions = self.adapter.load_corpus(limit=limit)

        await self.run_cognee()

        return self.questions

    async def run_cognee(self):
        setup_logging(logging.ERROR)

        # Pruning system and databases
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)

        # Adding corpus elements to cognee metastore
        for text in self.raw_corpus:
            await cognee.add(text)
            print(f"Added text: {text[:35]}...")

        # Running cognify and building knowledge graph
        await cognee.cognify()
