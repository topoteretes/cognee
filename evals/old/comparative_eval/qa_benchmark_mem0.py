import asyncio
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from mem0 import Memory

from qa_benchmark_base import QABenchmarkRAG, QABenchmarkConfig

load_dotenv()


@dataclass
class Mem0Config(QABenchmarkConfig):
    """Configuration for Mem0 QA benchmark."""

    # Memory parameters
    user_id: str = "hotpot_qa_user"

    # Model parameters
    model_name: str = "gpt-4o-mini"

    # Default results file
    results_file: str = "hotpot_qa_mem0_results.json"


class QABenchmarkMem0(QABenchmarkRAG):
    """Mem0 implementation of QA benchmark."""

    def __init__(self, corpus, qa_pairs, config: Mem0Config):
        super().__init__(corpus, qa_pairs, config)
        self.config: Mem0Config = config
        self.openai_client = None

    async def initialize_rag(self) -> Any:
        """Initialize Mem0 Memory and OpenAI client."""
        memory = Memory()
        self.openai_client = OpenAI()
        return memory

    async def cleanup_rag(self) -> None:
        """Clean up resources (no cleanup needed for Mem0)."""
        pass

    async def insert_document(self, document: str, document_id: int) -> None:
        """Insert document into Mem0 as conversation messages."""
        # Create conversation messages format
        messages = [
            {"role": "system", "content": "This is a document to remember."},
            {"role": "user", "content": "Please remember this document."},
            {"role": "assistant", "content": document},
        ]

        # Add to memory (wrap sync call in async)
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.rag_client.add(messages, user_id=self.config.user_id)
        )

    async def query_rag(self, question: str) -> str:
        """Query Mem0 and generate answer using OpenAI."""
        # Search Mem0 for relevant memories
        relevant_memories = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.rag_client.search(query=question, user_id=self.config.user_id, limit=5),
        )

        # Format memories for context
        memories_str = "\n".join(f"- {entry['memory']}" for entry in relevant_memories["results"])

        # Generate response with OpenAI
        system_prompt = f"You are a helpful AI assistant. Answer the question based on the provided context.\n\nContext:\n{memories_str}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        # Call OpenAI API (wrap sync call in async)
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.openai_client.chat.completions.create(
                model=self.config.model_name, messages=messages
            ),
        )
        answer = response.choices[0].message.content

        # Store the QA interaction in Mem0
        qa_messages = messages + [{"role": "assistant", "content": answer}]
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.rag_client.add(qa_messages, user_id=self.config.user_id)
        )

        return answer

    @property
    def system_name(self) -> str:
        """Return system name."""
        return "Mem0"


if __name__ == "__main__":
    # Example usage
    config = Mem0Config(
        corpus_limit=5,  # Small test
        qa_limit=3,
        print_results=True,
    )

    benchmark = QABenchmarkMem0.from_jsons(
        corpus_file="hotpot_50_corpus.json", qa_pairs_file="hotpot_50_qa_pairs.json", config=config
    )

    results = benchmark.run()
