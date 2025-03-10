from deepeval.metrics import SummarizationMetric
from deepeval.test_case import LLMTestCase
from deepeval.metrics.summarization.schema import ScoreType
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.utils import get_or_create_event_loop
from deepeval.metrics.summarization.template import SummarizationTemplate
from deepeval.metrics.summarization.schema import Reason
from deepeval.metrics.utils import trimAndLoadJson


class ContextCoverageMetric(SummarizationMetric):
    def measure(
        self,
        test_case,
        _show_indicator: bool = True,
    ) -> float:
        mapped_test_case = LLMTestCase(
            input=test_case.context[0],
            actual_output=test_case.retrieval_context[0],
        )
        self.assessment_questions = None
        self.evaluation_cost = 0 if self.using_native_model else None
        with metric_progress_indicator(self, _show_indicator=_show_indicator):
            if self.async_mode:
                loop = get_or_create_event_loop()
                return loop.run_until_complete(
                    self.a_measure(mapped_test_case, _show_indicator=False)
                )
            else:
                self.coverage_verdicts = self._generate_coverage_verdicts(mapped_test_case)
                self.alignment_verdicts = []
                self.score = self._calculate_score(ScoreType.COVERAGE)
                self.reason = self._generate_reason()
                self.success = self.score >= self.threshold
                return self.score

    async def a_measure(
        self,
        test_case,
        _show_indicator: bool = True,
    ) -> float:
        self.evaluation_cost = 0 if self.using_native_model else None
        with metric_progress_indicator(
            self,
            async_mode=True,
            _show_indicator=_show_indicator,
        ):
            self.coverage_verdicts = await self._a_generate_coverage_verdicts(test_case)
            self.alignment_verdicts = []
            self.score = self._calculate_score(ScoreType.COVERAGE)
            self.reason = self._generate_reason()
            self.success = self.score >= self.threshold
            return self.score

    def is_successful(self) -> bool:
        return self.success
