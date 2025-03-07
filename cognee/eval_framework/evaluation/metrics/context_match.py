from deepeval.metrics import SummarizationMetric
from deepeval.test_case import LLMTestCase


class ContextMatchMetric(SummarizationMetric):
    def measure(self, test_case, _show_indicator=True):
        mapped_test_case = LLMTestCase(
            input=test_case.context[0],
            actual_output=test_case.retrieval_context,
        )
        return super().measure(mapped_test_case, _show_indicator)
