import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()


@dataclass
class QABenchmarkConfig:
    """Base configuration for QA benchmark pipelines."""

    corpus_limit: Optional[int] = None
    qa_limit: Optional[int] = None
    results_file: str = "hotpot_qa_results.json"
    print_results: bool = True


class QABenchmarkRAG(ABC):
    """Abstract base class for QA benchmarking with different RAG systems."""

    def __init__(
        self, corpus: List[str], qa_pairs: List[Dict[str, Any]], config: QABenchmarkConfig
    ):
        """Initialize the benchmark with corpus and QA data."""
        self.corpus = corpus
        self.qa_pairs = qa_pairs
        self.config = config
        self.rag_client = None

        # Apply limits if specified
        if config.corpus_limit is not None:
            self.corpus = self.corpus[: config.corpus_limit]
            print(f"Limited to first {config.corpus_limit} documents")

        if config.qa_limit is not None:
            self.qa_pairs = self.qa_pairs[: config.qa_limit]
            print(f"Limited to first {config.qa_limit} questions")

    @classmethod
    def from_jsons(
        cls, corpus_file: str, qa_pairs_file: str, config: QABenchmarkConfig
    ) -> "QABenchmarkRAG":
        """Create benchmark instance by loading data from JSON files."""
        print(f"Loading corpus from {corpus_file}...")
        with open(corpus_file) as file:
            corpus = json.load(file)

        print(f"Loading QA pairs from {qa_pairs_file}...")
        with open(qa_pairs_file) as file:
            qa_pairs = json.load(file)

        return cls(corpus, qa_pairs, config)

    @abstractmethod
    async def initialize_rag(self) -> Any:
        """Initialize the RAG system. Returns the RAG client."""
        pass

    @abstractmethod
    async def cleanup_rag(self) -> None:
        """Clean up RAG system resources."""
        pass

    @abstractmethod
    async def insert_document(self, document: str, document_id: int) -> None:
        """Insert a single document into the RAG system."""
        pass

    @abstractmethod
    async def query_rag(self, question: str) -> str:
        """Query the RAG system and return the answer."""
        pass

    @property
    @abstractmethod
    def system_name(self) -> str:
        """Return the name of the RAG system for logging."""
        pass

    async def load_corpus_to_rag(self) -> None:
        """Load corpus data into the RAG system."""
        print(f"Adding {len(self.corpus)} documents to {self.system_name}...")
        for i, document in enumerate(tqdm(self.corpus, desc="Adding documents")):
            await self.insert_document(document, i + 1)
        print(f"All documents added to {self.system_name}")

    async def answer_questions(self) -> List[Dict[str, Any]]:
        """Answer questions using the RAG system."""
        print(f"Processing {len(self.qa_pairs)} questions...")
        results = []

        for i, qa_pair in enumerate(self.qa_pairs):
            question = qa_pair.get("question")
            expected_answer = qa_pair.get("answer")

            print(f"Processing question {i + 1}/{len(self.qa_pairs)}: {question}")

            # Get answer from RAG system
            try:
                answer = await self.query_rag(question)
            except Exception as e:
                print(f"Error processing question {i + 1}: {e}")
                answer = f"Error: {str(e)}"

            result = {"question": question, "answer": answer, "golden_answer": expected_answer}

            if self.config.print_results:
                print(
                    f"Question {i + 1}: {question}\nResponse: {answer}\nExpected: {expected_answer}\n{'-' * 50}"
                )

            results.append(result)

        return results

    def save_results(self, results: List[Dict[str, Any]]) -> None:
        """Save results to JSON file."""
        if self.config.results_file:
            print(f"Saving results to {self.config.results_file}...")
            with open(self.config.results_file, "w", encoding="utf-8") as file:
                json.dump(results, file, indent=2)

    async def run_benchmark(self) -> List[Dict[str, Any]]:
        """Run the complete benchmark pipeline."""
        print(f"Starting QA benchmark for {self.system_name}...")

        try:
            # Initialize RAG system
            self.rag_client = await self.initialize_rag()

            # Load corpus
            await self.load_corpus_to_rag()

            # Answer questions
            results = await self.answer_questions()

            # Save results
            self.save_results(results)

            print(f"Results saved to {self.config.results_file}")
            print("Pipeline completed successfully")
            return results

        except Exception as e:
            print(f"An error occurred during benchmark: {e}")
            raise
        finally:
            # Cleanup
            if self.rag_client:
                await self.cleanup_rag()

    def run(self) -> List[Dict[str, Any]]:
        """Synchronous wrapper for the benchmark."""
        return asyncio.run(self.run_benchmark())
