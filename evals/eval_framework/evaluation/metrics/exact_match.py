from deepeval.test_case import LLMTestCase


class ExactMatchMetric:
    def __init__(self):
        self.score = None
        self.reason = None

    def measure(self, test_case: LLMTestCase):
        actual = test_case.actual_output[0].strip() if test_case.actual_output else ""
        expected = test_case.expected_output.strip() if test_case.expected_output else ""
        self.score = 1.0 if actual == expected else 0.0
        self.reason = "Exact match" if self.score == 1.0 else "Not an exact match"
        return self.score
