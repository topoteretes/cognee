import cognee
import logging
from typing import Optional, Tuple, List, Dict, Union, Any

from cognee.eval_framework.benchmark_adapters.benchmark_adapters import BenchmarkAdapter
from cognee.eval_framework.corpus_builder.task_getters.task_getters import TaskGetters
from cognee.eval_framework.corpus_builder.task_getters.base_task_getter import BaseTaskGetter
from cognee.modules.chunking.TextChunker import TextChunker
from cognee.shared.utils import setup_logging


class CorpusBuilderExecutor:
    def __init__(
        self, benchmark: Union[str, Any] = "Dummy", task_getter_type: str = "DEFAULT"
    ) -> None:
        if isinstance(benchmark, str):
            try:
                adapter_enum = BenchmarkAdapter(benchmark)
            except ValueError:
                raise ValueError(f"Unsupported benchmark: {benchmark}")
            self.adapter = adapter_enum.adapter_class()
        else:
            self.adapter = benchmark

        self.raw_corpus = None
        self.questions = None

        try:
            task_enum = TaskGetters(task_getter_type)
        except KeyError:
            raise ValueError(f"Invalid task getter type: {task_getter_type}")

        self.task_getter: BaseTaskGetter = task_enum.getter_class()

    def load_corpus(self, limit: Optional[int] = None) -> Tuple[List[Dict], List[str]]:
        self.raw_corpus, self.questions = self.adapter.load_corpus(limit=limit)
        return self.raw_corpus, self.questions

    async def build_corpus(
        self, limit: Optional[int] = None, chunk_size=1024, chunker=TextChunker
    ) -> List[str]:
        self.load_corpus(limit=limit)
        await self.run_cognee(chunk_size=chunk_size, chunker=chunker)
        return self.questions

    async def run_cognee(self, chunk_size=1024, chunker=TextChunker) -> None:
        setup_logging(logging.ERROR)

        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)

        await cognee.add(self.raw_corpus)

        tasks = await self.task_getter.get_tasks(chunk_size=chunk_size, chunker=chunker)
        await cognee.cognify(tasks=tasks)
