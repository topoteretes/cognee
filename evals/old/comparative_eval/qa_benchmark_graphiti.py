import asyncio
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from qa_benchmark_base import QABenchmarkRAG, QABenchmarkConfig

load_dotenv()


@dataclass
class GraphitiConfig(QABenchmarkConfig):
    """Configuration for Graphiti QA benchmark."""

    # Database parameters
    db_url: str = os.getenv("NEO4J_URI")
    db_user: str = os.getenv("NEO4J_USER")
    db_password: str = os.getenv("NEO4J_PASSWORD")

    # Model parameters
    model_name: str = "gpt-4o-mini"

    # Default results file
    results_file: str = "hotpot_qa_graphiti_results.json"


class QABenchmarkGraphiti(QABenchmarkRAG):
    """Graphiti implementation of QA benchmark."""

    def __init__(self, corpus, qa_pairs, config: GraphitiConfig):
        super().__init__(corpus, qa_pairs, config)
        self.config: GraphitiConfig = config
        self.llm = None

    async def initialize_rag(self) -> Any:
        """Initialize Graphiti and LLM."""
        graphiti = Graphiti(self.config.db_url, self.config.db_user, self.config.db_password)
        await graphiti.build_indices_and_constraints(delete_existing=True)

        # Initialize LLM
        self.llm = ChatOpenAI(model=self.config.model_name, temperature=0)

        return graphiti

    async def cleanup_rag(self) -> None:
        """Clean up Graphiti connection."""
        if self.rag_client:
            await self.rag_client.close()

    async def insert_document(self, document: str, document_id: int) -> None:
        """Insert document into Graphiti as an episode."""
        await self.rag_client.add_episode(
            name=f"Document {document_id}",
            episode_body=document,
            source=EpisodeType.text,
            source_description="corpus",
            reference_time=datetime.now(timezone.utc),
        )

    async def query_rag(self, question: str) -> str:
        """Query Graphiti and generate answer using LLM."""
        # Search Graphiti for relevant facts
        results = await self.rag_client.search(query=question, num_results=10)
        context = "\n".join(f"- {entry.fact}" for entry in results)

        # Generate answer using LLM
        messages = [
            {
                "role": "system",
                "content": "Answer minimally using provided facts. Respond with one word or phrase.",
            },
            {"role": "user", "content": f"Facts:\n{context}\n\nQuestion: {question}"},
        ]

        response = await self.llm.ainvoke(messages)
        answer = response.content

        # Store the QA interaction in Graphiti
        qa_memory = f"Question: {question}\nAnswer: {answer}"
        await self.rag_client.add_episode(
            name="QA Interaction",
            episode_body=qa_memory,
            source=EpisodeType.text,
            source_description="qa_interaction",
            reference_time=datetime.now(timezone.utc),
        )

        return answer

    @property
    def system_name(self) -> str:
        """Return system name."""
        return "Graphiti"


if __name__ == "__main__":
    # Example usage
    config = GraphitiConfig(
        corpus_limit=5,  # Small test
        qa_limit=3,
        print_results=True,
    )

    benchmark = QABenchmarkGraphiti.from_jsons(
        corpus_file="hotpot_50_corpus.json", qa_pairs_file="hotpot_50_qa_pairs.json", config=config
    )

    results = benchmark.run()
