from evals.deepeval_metrics import (
    correctness_metric,
    comprehensiveness_metric,
    diversity_metric,
    empowerment_metric,
    directness_metric,
    f1_score_metric,
    em_score_metric,
)
from evals.promptfoo_metrics import PromptfooMetric
from deepeval.metrics import AnswerRelevancyMetric
import deepeval.metrics
from cognee.infrastructure.llm.prompts.llm_judge_prompts import llm_judge_prompts

native_deepeval_metrics = {"AnswerRelevancy": AnswerRelevancyMetric}

custom_deepeval_metrics = {
    "Correctness": correctness_metric,
    "Comprehensiveness": comprehensiveness_metric,
    "Diversity": diversity_metric,
    "Empowerment": empowerment_metric,
    "Directness": directness_metric,
    "F1": f1_score_metric,
    "EM": em_score_metric,
}

promptfoo_metrics = {
    "promptfoo.correctness": PromptfooMetric(llm_judge_prompts["correctness"]),
    "promptfoo.comprehensiveness": PromptfooMetric(llm_judge_prompts["comprehensiveness"]),
    "promptfoo.diversity": PromptfooMetric(llm_judge_prompts["diversity"]),
    "promptfoo.empowerment": PromptfooMetric(llm_judge_prompts["empowerment"]),
    "promptfoo.directness": PromptfooMetric(llm_judge_prompts["directness"]),
}

qa_metrics = native_deepeval_metrics | custom_deepeval_metrics | promptfoo_metrics


def get_metric(metric_name: str):
    if metric_name in qa_metrics:
        metric = qa_metrics[metric_name]
    else:
        try:
            metric_cls = getattr(deepeval.metrics, metric_name)
            metric = metric_cls()
        except AttributeError:
            raise Exception(f"Metric {metric_name} not supported")

    if isinstance(metric, type):
        metric = metric()

    return metric
