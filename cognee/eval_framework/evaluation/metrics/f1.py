from collections import Counter
from deepeval.test_case import LLMTestCase
import re
from typing import Optional, Any


class F1ScoreMetric:
    def __init__(self) -> None:
        self.score: Optional[float] = None
        self.reason: Optional[str] = None

    def measure(self, test_case: "LLMTestCase") -> float:
        actual = (test_case.actual_output or "").lower()
        expected = (test_case.expected_output or "").lower()

        actual_tokens = [
            re.sub(r"\W+", "", token.strip())
            for token in actual.split()
            if re.sub(r"\W+", "", token.strip())
        ]

        expected_tokens = [
            re.sub(r"\W+", "", token.strip())
            for token in expected.split()
            if re.sub(r"\W+", "", token.strip())
        ]

        if not actual_tokens and not expected_tokens:
            self.score = 1.0
            self.reason = "Both actual and expected are empty"
            return self.score

        actual_counts = Counter(actual_tokens)
        expected_counts = Counter(expected_tokens)

        tp = sum(min(actual_counts[word], expected_counts[word]) for word in actual_counts)
        fp = sum(actual_counts[word] for word in actual_counts) - tp
        fn = sum(expected_counts[word] for word in expected_counts) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        self.score = f1
        self.reason = f"F1: {f1:.2f} (Precision: {precision:.2f}, Recall: {recall:.2f})"
        return self.score
