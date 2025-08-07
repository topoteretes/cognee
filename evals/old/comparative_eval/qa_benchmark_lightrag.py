import asyncio
import os
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import setup_logger
from qa_benchmark_base import QABenchmarkRAG, QABenchmarkConfig

load_dotenv()
setup_logger("lightrag", level="INFO")


@dataclass
class LightRAGConfig(QABenchmarkConfig):
    """Configuration for LightRAG QA benchmark."""

    # Storage parameters
    working_dir: str = "./lightrag_storage"

    # Query parameters
    query_mode: str = "hybrid"  # "naive", "local", "global", "hybrid"

    # Default results file
    results_file: str = "hotpot_qa_lightrag_results.json"


class QABenchmarkLightRAG(QABenchmarkRAG):
    """LightRAG implementation of QA benchmark."""

    def __init__(self, corpus, qa_pairs, config: LightRAGConfig):
        super().__init__(corpus, qa_pairs, config)
        self.config: LightRAGConfig = config

        # Ensure working directory exists
        if not os.path.exists(self.config.working_dir):
            os.makedirs(self.config.working_dir)

    async def initialize_rag(self) -> Any:
        """Initialize LightRAG with storage and pipeline setup."""
        lightrag = LightRAG(
            working_dir=self.config.working_dir,
            embedding_func=openai_embed,
            llm_model_func=gpt_4o_mini_complete,
        )

        await lightrag.initialize_storages()
        await initialize_pipeline_status()

        return lightrag

    async def cleanup_rag(self) -> None:
        """Clean up LightRAG storages."""
        if self.rag_client:
            await self.rag_client.finalize_storages()

    async def insert_document(self, document: str, document_id: int) -> None:
        """Insert document into LightRAG."""
        await self.rag_client.ainsert([document])

    async def query_rag(self, question: str) -> str:
        """Query LightRAG and return the answer."""
        result = await self.rag_client.aquery(
            question, param=QueryParam(mode=self.config.query_mode)
        )
        return result

    @property
    def system_name(self) -> str:
        """Return system name."""
        return "LightRAG"


if __name__ == "__main__":
    # Example usage
    config = LightRAGConfig(
        corpus_limit=5,  # Small test
        qa_limit=3,
        query_mode="hybrid",
        print_results=True,
    )

    benchmark = QABenchmarkLightRAG.from_jsons(
        corpus_file="hotpot_50_corpus.json", qa_pairs_file="hotpot_50_qa_pairs.json", config=config
    )

    results = benchmark.run()
