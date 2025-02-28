from deepeval.test_case import LLMTestCase
from typing import Optional


class ExactMatchMetric:
    def __init__(self) -> None:
        self.score: Optional[float] = None
        self.reason: Optional[str] = None

    def measure(self, test_case: "LLMTestCase") -> float:
        actual = test_case.actual_output.strip().lower() if test_case.actual_output else ""
        expected = test_case.expected_output.strip().lower() if test_case.expected_output else ""
        self.score = 1.0 if actual == expected else 0.0
        self.reason = "Exact match" if self.score == 1.0 else "Not an exact match"
        return self.score
