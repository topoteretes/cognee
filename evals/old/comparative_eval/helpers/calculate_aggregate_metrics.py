#!/usr/bin/env python3
"""Simple script to calculate aggregate metrics for multiple JSON files."""

import os
from cognee.eval_framework.analysis.metrics_calculator import calculate_metrics_statistics
from cognee.shared.logging_utils import get_logger

logger = get_logger()


def calculate_aggregates_for_files(json_paths: list[str]) -> None:
    """Calculate aggregate metrics for a list of JSON files."""
    for json_path in json_paths:
        if not os.path.exists(json_path):
            logger.error(f"File not found: {json_path}")
            continue

        # Generate output path for aggregate metrics in the same folder as input
        input_dir = os.path.dirname(json_path)
        base_name = os.path.splitext(os.path.basename(json_path))[0]
        output_path = os.path.join(input_dir, f"aggregate_metrics_{base_name}.json")

        try:
            logger.info(f"Calculating aggregate metrics for {json_path}")
            calculate_metrics_statistics(json_path, output_path)
            logger.info(f"Saved aggregate metrics to {output_path}")
        except Exception as e:
            logger.error(f"Failed to calculate metrics for {json_path}: {e}")


if __name__ == "__main__":
    dir_path = ""
    json_file_paths = [
        os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(".json")
    ]

    calculate_aggregates_for_files(json_file_paths)
    print("Done calculating aggregate metrics!")
