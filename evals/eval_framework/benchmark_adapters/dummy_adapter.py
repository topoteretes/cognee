from typing import Optional

from evals.eval_framework.benchmark_adapters.base_benchmark_adapter import BaseBenchmarkAdapter


class DummyAdapter(BaseBenchmarkAdapter):
    def load_corpus(
        self, limit: Optional[int] = None, seed: int = 42
    ) -> tuple[list[str], list[dict[str, str]]]:
        corpus_list = [
            "The cognee is an AI memory engine that supports different vector and graph databases",
            "Neo4j is a graph database supported by cognee",
        ]
        question_answer_pairs = [
            {
                "answer": "Yes",
                "question": "Is Neo4j supported by cognee?",
                "type": "dummy",
            }
        ]

        return corpus_list, question_answer_pairs
