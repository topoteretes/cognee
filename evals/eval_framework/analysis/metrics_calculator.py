import json
from collections import defaultdict
import numpy as np
from typing import Dict, List, Tuple


def bootstrap_ci(scores, num_samples=10000, confidence_level=0.95):
    """Calculate bootstrap confidence intervals for a list of scores."""
    means = []
    n = len(scores)
    for _ in range(num_samples):
        sample = np.random.choice(scores, size=n, replace=True)
        means.append(np.mean(sample))

    lower_bound = np.percentile(means, (1 - confidence_level) / 2 * 100)
    upper_bound = np.percentile(means, (1 + confidence_level) / 2 * 100)
    return np.mean(scores), lower_bound, upper_bound


def load_metrics_data(json_file_path: str) -> List[Dict]:
    """Load metrics data from JSON file."""
    try:
        with open(json_file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find the file: {json_file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON from {json_file_path}: {e}")


def extract_metrics_and_details(
    data: List[Dict],
) -> Tuple[Dict[str, List[float]], Dict[str, List[Dict]]]:
    """Extract metrics scores and details from evaluation data."""
    metrics_data = defaultdict(list)
    metric_details = defaultdict(list)

    for entry in data:
        for metric, values in entry["metrics"].items():
            score = values["score"]
            metrics_data[metric].append(score)
            if "reason" in values:
                metric_details[metric].append(
                    {
                        "question": entry["question"],
                        "answer": entry["answer"],
                        "golden_answer": entry["golden_answer"],
                        "reason": values["reason"],
                        "score": score,
                    }
                )

    return metrics_data, metric_details


def save_aggregate_metrics(
    metrics_data: Dict[str, List[float]],
    ci_results: Dict[str, Tuple[float, float, float]],
    output_path: str,
) -> None:
    """Save aggregated metrics and confidence intervals to file."""
    aggregate_data = {
        metric: {
            "scores": scores,
            "mean": ci_results[metric][0],
            "ci_lower": ci_results[metric][1],
            "ci_upper": ci_results[metric][2],
        }
        for metric, scores in metrics_data.items()
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(aggregate_data, f, indent=4)


def calculate_metrics_statistics(
    json_data: str, aggregate_output_path: str
) -> Tuple[Dict[str, List[float]], Dict[str, List[Dict]], Dict[str, Tuple[float, float, float]]]:
    """Calculate metrics statistics and save aggregated results."""
    data = load_metrics_data(json_data)
    metrics_data, metric_details = extract_metrics_and_details(data)

    # Calculate confidence intervals
    ci_results = {}
    for metric, scores in metrics_data.items():
        mean_score, lower, upper = bootstrap_ci(scores)
        ci_results[metric] = (mean_score, lower, upper)

    # Save aggregate metrics
    save_aggregate_metrics(metrics_data, ci_results, aggregate_output_path)

    return metrics_data, metric_details, ci_results
