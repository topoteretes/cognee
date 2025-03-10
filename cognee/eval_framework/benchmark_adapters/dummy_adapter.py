from typing import Optional, Any, List, Union, Tuple

from cognee.eval_framework.benchmark_adapters.base_benchmark_adapter import BaseBenchmarkAdapter


class DummyAdapter(BaseBenchmarkAdapter):
    def load_corpus(
        self,
        limit: Optional[int] = None,
        seed: int = 42,
        load_golden_context: bool = False,
        instance_filter: Optional[Union[str, List[str], List[int]]] = None,
    ) -> Tuple[List[str], List[dict[str, Any]]]:
        corpus_list = [
            "The cognee is an AI memory engine that supports different vector and graph databases",
            "Neo4j is a graph database supported by cognee",
        ]
        qa_pair = {
            "answer": "Yes",
            "question": "Is Neo4j supported by cognee?",
            "type": "dummy",
        }

        if load_golden_context:
            qa_pair["golden_context"] = "Cognee supports Neo4j and NetworkX"

        question_answer_pairs = [qa_pair]

        # Instance filtering is not applicable for the dummy adapter as it always returns the same data
        # but we include the parameter for API consistency

        return corpus_list, question_answer_pairs
