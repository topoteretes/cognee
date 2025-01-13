from deepeval.metrics import BaseMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from evals.official_hotpot_metrics import exact_match_score, f1_score
from cognee.infrastructure.llm.prompts.llm_judge_prompts import llm_judge_prompts

correctness_metric = GEval(
    name="Correctness",
    model="gpt-4o-mini",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    evaluation_steps=[llm_judge_prompts["correctness"]],
)

comprehensiveness_metric = GEval(
    name="Comprehensiveness",
    model="gpt-4o-mini",
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    evaluation_steps=[llm_judge_prompts["comprehensiveness"]],
)

diversity_metric = GEval(
    name="Diversity",
    model="gpt-4o-mini",
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    evaluation_steps=[llm_judge_prompts["diversity"]],
)

empowerment_metric = GEval(
    name="Empowerment",
    model="gpt-4o-mini",
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    evaluation_steps=[llm_judge_prompts["empowerment"]],
)

directness_metric = GEval(
    name="Directness",
    model="gpt-4o-mini",
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    evaluation_steps=[llm_judge_prompts["directness"]],
)


class f1_score_metric(BaseMetric):
    """F1 score taken directly from the official hotpot benchmark
    implementation and wrapped into a deepeval metric."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def measure(self, test_case: LLMTestCase):
        f1, precision, recall = f1_score(
            prediction=test_case.actual_output,
            ground_truth=test_case.expected_output,
        )
        self.score = f1
        self.success = self.score >= self.threshold
        return self.score

    # Reusing regular measure as async F1 score is not implemented
    async def a_measure(self, test_case: LLMTestCase):
        return self.measure(test_case)

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "Official hotpot F1 score"


class em_score_metric(BaseMetric):
    """Exact Match score taken directly from the official hotpot benchmark
    implementation and wrapped into a deepeval metric."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def measure(self, test_case: LLMTestCase):
        self.score = exact_match_score(
            prediction=test_case.actual_output,
            ground_truth=test_case.expected_output,
        )
        self.success = self.score >= self.threshold
        return self.score

    # Reusing regular measure as async F1 score is not implemented
    async def a_measure(self, test_case: LLMTestCase):
        return self.measure(test_case)

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "Official hotpot EM score"
