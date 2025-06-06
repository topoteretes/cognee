import json
import os
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd


def convert_metrics_file(json_path: str, metrics: List[str] = None) -> Dict[str, Any]:
    """Convert a single metrics JSON file to the desired format."""
    if metrics is None:
        metrics = ["correctness", "f1", "EM"]

    with open(json_path, "r") as f:
        data = json.load(f)

    # Extract filename without extension for system name
    filename = Path(json_path).stem

    # Convert to desired format
    result = {
        "system": filename,
        "Human-LLM Correctness": None,
        "Human-LLM Correctness Error": None,
    }

    # Add metrics dynamically based on the metrics list
    for metric in metrics:
        if metric in data:
            result[f"DeepEval {metric.title()}"] = data[metric]["mean"]
            result[f"DeepEval {metric.title()} Error"] = [
                data[metric]["ci_lower"],
                data[metric]["ci_upper"],
            ]
        else:
            print(f"Warning: Metric '{metric}' not found in {json_path}")

    return result


def convert_to_dataframe(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert results list to DataFrame with expanded error columns."""
    df_data = []

    for result in results:
        row = {}
        for key, value in result.items():
            if key.endswith("Error") and isinstance(value, list) and len(value) == 2:
                # Split error columns into lower and upper
                row[f"{key} Lower"] = value[0]
                row[f"{key} Upper"] = value[1]
            else:
                row[key] = value
        df_data.append(row)

    return pd.DataFrame(df_data)


def process_multiple_files(
    json_paths: List[str], output_path: str, metrics: List[str] = None
) -> None:
    """Process multiple JSON files and save concatenated results."""
    if metrics is None:
        metrics = ["correctness", "f1", "EM"]

    results = []

    for json_path in json_paths:
        try:
            converted = convert_metrics_file(json_path, metrics)
            results.append(converted)
            print(f"Processed: {json_path}")
        except Exception as e:
            print(f"Error processing {json_path}: {e}")

    # Save JSON results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved {len(results)} results to {output_path}")

    # Convert to DataFrame and save CSV
    df = convert_to_dataframe(results)
    csv_path = output_path.replace(".json", ".csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved DataFrame to {csv_path}")


if __name__ == "__main__":
    # Default metrics (can be customized here)
    # default_metrics = ['correctness', 'f1', 'EM']
    default_metrics = ["correctness"]

    # List JSON files in the current directory
    current_dir = ""
    json_files = [f for f in os.listdir(current_dir) if f.endswith(".json")]

    if json_files:
        print(f"Found {len(json_files)} JSON files:")
        for f in json_files:
            print(f"  - {f}")

        # Create full paths for JSON files and output file in current working directory
        json_full_paths = [os.path.join(current_dir, f) for f in json_files]
        output_file = os.path.join(current_dir, "converted_metrics.json")
        process_multiple_files(json_full_paths, output_file, default_metrics)
    else:
        print("No JSON files found in current directory")
