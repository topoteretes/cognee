from evals.deepeval_metrics import (
    correctness_metric,
    comprehensiveness_metric,
    diversity_metric,
    empowerment_metric,
    directness_metric,
    f1_score_metric,
    em_score_metric,
)
from deepeval.metrics import AnswerRelevancyMetric
import deepeval.metrics
from evals.promptfoo_metrics import is_valid_promptfoo_metric, PromptfooMetric

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

qa_metrics = native_deepeval_metrics | custom_deepeval_metrics


def get_deepeval_metric(metric_name: str):
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


def get_metrics(metric_name_list: list[str]):
    metrics = {
        "deepeval_metrics": [],
    }

    promptfoo_metric_names = []

    for metric_name in metric_name_list:
        if (
            (metric_name in native_deepeval_metrics)
            or (metric_name in custom_deepeval_metrics)
            or hasattr(deepeval.metrics, metric_name)
        ):
            metric = get_deepeval_metric(metric_name)
            metrics["deepeval_metrics"].append(metric)
        elif is_valid_promptfoo_metric(metric_name):
            promptfoo_metric_names.append(metric_name)

    if len(promptfoo_metric_names) > 0:
        metrics["promptfoo_metrics"] = PromptfooMetric(promptfoo_metric_names)

    return metrics
