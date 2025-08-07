import json
import os
from pathlib import Path
from typing import Dict, List, Any


def read_results(dir_path: str) -> Dict[str, Any]:
    """Read all JSON files from the specified directory path."""
    results = {}
    dir_path = Path(dir_path)

    if not dir_path.exists():
        raise FileNotFoundError(f"Directory {dir_path} does not exist")

    for file_path in dir_path.glob("*.json"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                results[file_path.name] = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error reading {file_path}: {e}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    return results


def validate_file_results(
    data: List[Dict[str, Any]], filename: str, expected_keys: List[str] = None
) -> bool:
    """Validate that a single file's data has correct structure and keys."""
    if expected_keys is None:
        expected_keys = ["answer", "golden_answer", "metrics", "question"]

    if not isinstance(data, list):
        print(f"{filename} is not a list")
        return False

    if not data:
        print(f"{filename} is empty")
        return False

    expected_keys_set = set(expected_keys)

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            print(f"{filename}[{i}] is not a dictionary")
            return False

        current_keys = set(item.keys())
        if current_keys != expected_keys_set:
            missing_keys = expected_keys_set - current_keys
            extra_keys = current_keys - expected_keys_set
            print(f"Keys mismatch in {filename}[{i}]:")
            if missing_keys:
                print(f"  Missing keys: {missing_keys}")
            if extra_keys:
                print(f"  Extra keys: {extra_keys}")
            return False

        # Validate metrics if present
        if "metrics" in item:
            if not validate_metrics(item["metrics"]):
                print(f"Metrics validation failed in {filename}[{i}]")
                return False

    return True


def validate_metrics(metrics: Dict[str, Any], expected_metrics: List[str] = None) -> bool:
    """Validate that metrics have correct structure and expected keys."""
    if expected_metrics is None:
        expected_metrics = ["directllm_correctness", "deepeval_correctness", "EM", "f1"]

    if not isinstance(metrics, dict):
        print("Metrics is not a dictionary")
        return False

    # Check if all expected metrics are present
    metrics_keys = set(metrics.keys())
    expected_metrics_set = set(expected_metrics)

    missing_metrics = expected_metrics_set - metrics_keys
    if missing_metrics:
        print(f"Missing metrics: {missing_metrics}")
        return False

    # Validate each metric has a 'score' key with a numeric value.
    for metric_name, metric_data in metrics.items():
        # Handle both dictionary format (with score key) and direct numeric format
        if isinstance(metric_data, dict):
            if "score" not in metric_data:
                print(f"Metric '{metric_name}' does not have a 'score' key.")
                return False

            if not isinstance(metric_data["score"], (int, float)):
                print(f"Metric '{metric_name}' score is not a number: {metric_data['score']}")
                return False
        elif isinstance(metric_data, (int, float)):
            # Direct numeric format is also valid
            pass
        else:
            print(
                f"Metric '{metric_name}' value is neither a dictionary nor a number: {metric_data}"
            )
            return False

    return True


def validate_folder_results(results: Dict[str, Any], expected_keys: List[str] = None) -> bool:
    """Validate that all files have same length and all dictionaries contain same keys."""
    if expected_keys is None:
        expected_keys = ["answer", "golden_answer", "metrics", "question"]

    if not results:
        print("No results to validate")
        return False

    # Get lengths of all lists
    lengths = {name: len(data) for name, data in results.items()}

    # Check if all lists have same length
    if len(set(lengths.values())) > 1:
        print(f"List lengths differ: {lengths}")
        return False

    # Validate each file individually
    for filename, data in results.items():
        if not validate_file_results(data, filename, expected_keys):
            return False

    print("Validation passed: all lists have same length and all dictionaries have expected keys")
    print(f"Expected keys: {sorted(expected_keys)}")
    return True


if __name__ == "__main__":
    results = read_results("./data/cognee_graphsearch")
    # print(results)
    validate_folder_results(results)
