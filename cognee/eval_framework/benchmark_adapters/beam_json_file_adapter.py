"""Load BEAM-shaped eval data from JSON (e.g. export from beam_load_one_conversation)."""

import json
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

from cognee.eval_framework.benchmark_adapters.base_benchmark_adapter import BaseBenchmarkAdapter


class BEAMJsonFileAdapter(BaseBenchmarkAdapter):
    """Expects keys `corpus` (str) and `probing_questions` (list of dicts)."""

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)

    def _load_json_payload(self) -> dict[str, Any]:
        data = json.loads(self.path.read_text(encoding="utf-8"))

        if not isinstance(data, dict):
            raise ValueError(f"Expected a JSON object in {self.path}")
        if "corpus" not in data:
            raise ValueError(f"Missing required key 'corpus' in {self.path}")
        if "probing_questions" not in data:
            raise ValueError(f"Missing required key 'probing_questions' in {self.path}")
        if not isinstance(data["corpus"], str):
            raise ValueError(f"Expected 'corpus' to be a string in {self.path}")
        if not isinstance(data["probing_questions"], list):
            raise ValueError(f"Expected 'probing_questions' to be a list in {self.path}")

        return data

    def load_corpus(
        self,
        limit: Optional[int] = None,
        seed: int = 42,
        load_golden_context: bool = False,
        instance_filter: Optional[Union[str, List[str], List[int]]] = None,
    ) -> Tuple[List[str], List[dict[str, Any]]]:
        data = self._load_json_payload()
        corpus = [data["corpus"]]
        questions: List[dict[str, Any]] = [dict(question) for question in data["probing_questions"]]

        for index, question in enumerate(questions):
            if not isinstance(question, dict):
                raise ValueError(
                    f"Expected probing_questions[{index}] to be an object in {self.path}"
                )
            for required_key in ("question", "answer", "question_type"):
                if required_key not in question:
                    raise ValueError(
                        f"Missing required key '{required_key}' in probing_questions[{index}] "
                        f"from {self.path}"
                    )

        if instance_filter is not None:
            questions = self._filter_instances(questions, instance_filter, id_key="question")

        if limit is not None and limit < len(questions):
            questions = questions[:limit]

        return corpus, questions
