#!/usr/bin/env python3
"""
Postprocessing script to create benchmark summary JSON from cross-benchmark analysis results.
Converts CSV data into JSON format with confidence intervals.
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np


def validate_csv_exists(csv_path: str) -> bool:
    """Validate that the CSV file exists and is readable."""
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found: {csv_path}")
        return False

    if not csv_path.endswith(".csv"):
        print(f"❌ File is not a CSV: {csv_path}")
        return False

    print(f"✅ CSV file found: {csv_path}")
    return True


def read_summary_dataframe(csv_path: str) -> pd.DataFrame:
    """Read the cross-benchmark summary CSV into a DataFrame."""
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ Successfully loaded CSV with {len(df)} rows and {len(df.columns)} columns")
        return df
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        raise


def validate_required_columns(df: pd.DataFrame) -> bool:
    """Validate that the DataFrame has all required columns."""
    required_columns = [
        "benchmark",
        "directllm_correctness_avg",
        "deepeval_correctness_avg",
        "EM_avg",
        "f1_avg",
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"❌ Missing required columns: {missing_columns}")
        print(f"Available columns: {list(df.columns)}")
        return False

    print(f"✅ All required columns found: {required_columns}")
    return True


def load_cross_benchmark_data(csv_path: str) -> pd.DataFrame:
    """Load cross-benchmark summary CSV data."""
    print(f"📊 Loading cross-benchmark data from {csv_path}")

    # Validate file exists
    if not validate_csv_exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Read DataFrame
    df = read_summary_dataframe(csv_path)

    # Validate required columns
    if not validate_required_columns(df):
        raise ValueError("CSV missing required columns")

    print("✅ Successfully loaded cross-benchmark data")
    return df


def get_benchmark_analysis_path(benchmark_name: str, temp_dir: str) -> str:
    """Get the path to the analysis folder for a benchmark."""
    analysis_path = os.path.join(temp_dir, benchmark_name, "analysis")
    return analysis_path


def load_aggregate_metrics(benchmark_name: str, temp_dir: str) -> pd.DataFrame:
    """Load the metrics_aggregate.csv file for a benchmark."""
    analysis_path = get_benchmark_analysis_path(benchmark_name, temp_dir)
    aggregate_csv_path = os.path.join(analysis_path, "metrics_aggregate.csv")

    if not os.path.exists(aggregate_csv_path):
        raise FileNotFoundError(f"Aggregate metrics file not found: {aggregate_csv_path}")

    try:
        df = pd.read_csv(aggregate_csv_path, index_col=0)
        print(f"✅ Loaded aggregate metrics for {benchmark_name}: {len(df)} questions")
        return df
    except Exception as e:
        print(f"❌ Error loading aggregate metrics for {benchmark_name}: {e}")
        raise


def bootstrap_confidence_interval(
    data: List[float], n_bootstrap: int = 1000, confidence: float = 0.95
) -> List[float]:
    """Calculate bootstrap confidence interval for given data."""
    bootstrap_means = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        resampled = np.random.choice(data, size=len(data), replace=True)
        # Calculate mean of resampled data
        bootstrap_means.append(np.mean(resampled))

    # Calculate confidence interval
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    lower_bound = np.percentile(bootstrap_means, lower_percentile)
    upper_bound = np.percentile(bootstrap_means, upper_percentile)

    return [lower_bound, upper_bound]


def load_all_run_scores(benchmark_name: str, temp_dir: str, metric: str) -> List[float]:
    """Load all individual run scores for a metric from the metrics CSV files."""
    analysis_path = get_benchmark_analysis_path(benchmark_name, temp_dir)
    metrics_csv_path = os.path.join(analysis_path, f"metrics_{metric}.csv")

    if not os.path.exists(metrics_csv_path):
        raise FileNotFoundError(f"Metrics CSV file not found: {metrics_csv_path}")

    try:
        df = pd.read_csv(metrics_csv_path, index_col=0)
        # Flatten all scores (each row is a question, each column is a run)
        all_scores = df.values.flatten()
        # Remove any NaN values
        all_scores = all_scores[~np.isnan(all_scores)]

        print(
            f"  📊 {metric}: loaded {len(all_scores)} individual run scores from {len(df)} questions × {len(df.columns)} runs"
        )
        return all_scores.tolist()
    except Exception as e:
        print(f"❌ Error loading run scores for {metric} in {benchmark_name}: {e}")
        raise


def process_single_metric_with_bootstrap(
    benchmark_name: str, temp_dir: str, metric: str, cross_benchmark_mean: float = None
) -> Tuple[float, List[float]]:
    """Process a single metric: load run scores, calculate bootstrap CI, and return mean and CI."""
    print(f"📊 Calculating {metric} for {benchmark_name}")

    # Load all individual run scores for bootstrapping
    all_run_scores = load_all_run_scores(benchmark_name, temp_dir, metric)

    # Use provided mean or calculate from run scores
    if cross_benchmark_mean is not None:
        final_mean = round(cross_benchmark_mean, 3)
        print(f"  📊 {metric}: using cross-benchmark mean={final_mean:.3f}")
    else:
        final_mean = round(np.mean(all_run_scores), 3)
        print(f"  📊 {metric}: calculated mean from run scores={final_mean:.3f}")

    # Calculate bootstrap confidence interval from all run scores
    confidence_interval = bootstrap_confidence_interval(all_run_scores)

    # Round confidence interval to 3 decimal places
    confidence_interval = [round(ci, 3) for ci in confidence_interval]

    print(f"  📊 {metric}: run scores range=[{min(all_run_scores):.3f}, {max(all_run_scores):.3f}]")
    print(
        f"  ✅ {metric}: mean={final_mean:.3f}, CI=[{confidence_interval[0]:.3f}, {confidence_interval[1]:.3f}]"
    )
    return final_mean, confidence_interval


def process_single_benchmark(
    benchmark_name: str, temp_dir: str, cross_benchmark_means: Dict[str, float] = None
) -> Dict[str, Any]:
    """Process a single benchmark and return formatted data."""
    print(f"🔄 Processing benchmark: {benchmark_name}")

    # Define metrics to process
    metrics = ["directllm_correctness", "deepeval_correctness", "EM", "f1"]

    # Calculate values for each metric
    metric_values = {}
    for metric in metrics:
        try:
            # Get cross-benchmark mean if available
            cross_benchmark_mean = None
            if cross_benchmark_means and metric in cross_benchmark_means:
                cross_benchmark_mean = cross_benchmark_means[metric]

            mean, confidence_interval = process_single_metric_with_bootstrap(
                benchmark_name, temp_dir, metric, cross_benchmark_mean
            )
            metric_values[metric] = {"mean": mean, "confidence_interval": confidence_interval}
        except Exception as e:
            print(f"❌ Error processing {metric} for {benchmark_name}: {e}")
            return None

    print(f"✅ Successfully processed {benchmark_name} with {len(metric_values)} metrics")
    return metric_values


def extract_confidence_intervals(
    metric_values: Dict[str, Dict[str, Any]],
) -> Dict[str, List[float]]:
    """Extract confidence intervals from processed metric values."""
    print(f"📊 Extracting confidence intervals for {len(metric_values)} metrics")

    confidence_intervals = {}
    for metric, data in metric_values.items():
        if "confidence_interval" in data:
            confidence_intervals[metric] = data["confidence_interval"]
            print(
                f"  ✅ {metric}: CI=[{data['confidence_interval'][0]:.4f}, {data['confidence_interval'][1]:.4f}]"
            )
        else:
            print(f"  ❌ {metric}: No confidence interval found")
            confidence_intervals[metric] = [0.0, 0.0]  # Fallback

    return confidence_intervals


def map_metric_names(metric: str) -> str:
    """Map internal metric names to output format names."""
    mapping = {
        "directllm_correctness": "Human-LLM Correctness",
        "deepeval_correctness": "DeepEval Correctness",
        "f1": "DeepEval F1",
        "EM": "DeepEval EM",
    }
    return mapping.get(metric, metric)


def create_metric_entry(
    metric_name: str, mean: float, confidence_interval: List[float]
) -> Tuple[str, float, List[float]]:
    """Create a formatted metric entry with proper name mapping."""
    mapped_name = map_metric_names(metric_name)
    error_name = f"{mapped_name} Error"
    return mapped_name, mean, error_name, confidence_interval


def format_benchmark_entry(
    benchmark_name: str, means: Dict[str, float], confidence_intervals: Dict[str, List[float]]
) -> Dict[str, Any]:
    """Format benchmark data into required JSON structure."""
    print(f"📝 Formatting benchmark entry for {benchmark_name}")

    formatted_entry = {"system": benchmark_name}

    # Process each metric
    for metric, mean in means.items():
        if metric in confidence_intervals:
            mapped_name, mean_value, error_name, ci = create_metric_entry(
                metric, mean, confidence_intervals[metric]
            )

            # Add metric value (already rounded to 3 decimal places)
            formatted_entry[mapped_name] = mean_value
            # Add error interval (already rounded to 3 decimal places)
            formatted_entry[error_name] = ci

            print(f"  ✅ {mapped_name}: {mean_value:.3f}, Error: [{ci[0]:.3f}, {ci[1]:.3f}]")
        else:
            print(f"  ❌ {metric}: No confidence interval found")

    return formatted_entry


def validate_benchmark_folder(benchmark_name: str, temp_dir: str) -> bool:
    """Validate that a benchmark folder has the required analysis files."""
    analysis_path = get_benchmark_analysis_path(benchmark_name, temp_dir)

    if not os.path.exists(analysis_path):
        print("  ❌ Analysis folder not found: {analysis_path}")
        return False

    # Check for required metric files
    required_files = [
        "metrics_directllm_correctness.csv",
        "metrics_deepeval_correctness.csv",
        "metrics_EM.csv",
        "metrics_f1.csv",
    ]

    missing_files = []
    for file in required_files:
        file_path = os.path.join(analysis_path, file)
        if not os.path.exists(file_path):
            missing_files.append(file)

    if missing_files:
        print(f"  ❌ Missing required files: {missing_files}")
        return False

    print("  ✅ Benchmark folder validated")
    return True


def handle_processing_errors(benchmark_name: str, error: Exception) -> None:
    """Handle and log processing errors for a benchmark."""
    print(f"  ❌ Error processing {benchmark_name}: {error}")
    print(f"  📝 Skipping {benchmark_name} and continuing with next benchmark")


def process_all_benchmarks(temp_dir: str, max_benchmarks: int = 3) -> List[Dict[str, Any]]:
    """Process all benchmarks with optional limit for testing."""
    print(f"Processing benchmarks from {temp_dir} (max: {max_benchmarks})")

    # Load cross-benchmark summary to get benchmark names
    csv_path = os.path.join(temp_dir, "cross_benchmark_summary.csv")
    summary_df = load_cross_benchmark_data(csv_path)

    results = []
    processed_count = 0
    skipped_count = 0
    error_count = 0

    print(f"\n📊 Found {len(summary_df)} benchmarks to process")

    # Process each benchmark
    for _, row in summary_df.iterrows():
        if max_benchmarks is not None and processed_count >= max_benchmarks:
            print(f"⏹️  Reached max benchmark limit ({max_benchmarks})")
            break

        benchmark_name = row["benchmark"]
        total_benchmarks = len(summary_df)
        current_progress = processed_count + 1
        print(f"\n📊 Processing benchmark {current_progress}/{total_benchmarks}: {benchmark_name}")

        # Validate benchmark folder (PHASE 6 - IMPLEMENTED)
        if not validate_benchmark_folder(benchmark_name, temp_dir):
            print(f"  ⏭️  Skipping {benchmark_name} due to validation failure")
            skipped_count += 1
            continue

        # Get cross-benchmark means for this benchmark
        cross_benchmark_means = {
            "directllm_correctness": row.get("directllm_correctness_avg"),
            "deepeval_correctness": row.get("deepeval_correctness_avg"),
            "EM": row.get("EM_avg"),
            "f1": row.get("f1_avg"),
        }

        # Process single benchmark with error handling (PHASE 6 - IMPLEMENTED)
        try:
            metric_values = process_single_benchmark(
                benchmark_name, temp_dir, cross_benchmark_means
            )

            if metric_values:
                # Extract confidence intervals (PHASE 4 - IMPLEMENTED)
                print("📊 Extracting confidence intervals for {benchmark_name}")
                confidence_intervals = extract_confidence_intervals(metric_values)

                # Extract means for formatting
                means = {metric: data["mean"] for metric, data in metric_values.items()}

                # Format benchmark entry (PHASE 5 - IMPLEMENTED)
                formatted_entry = format_benchmark_entry(
                    benchmark_name, means, confidence_intervals
                )

                print(f"✅ Successfully processed and formatted {benchmark_name}")
                results.append(formatted_entry)
                processed_count += 1
            else:
                print(f"❌ Failed to process {benchmark_name}")
                error_count += 1

        except Exception as e:
            handle_processing_errors(benchmark_name, e)
            error_count += 1

    # Print final summary (PHASE 6 - IMPLEMENTED)
    print("\n📊 Processing Summary:")
    print(f"  ✅ Successfully processed: {processed_count}")
    print(f"  ⏭️  Skipped (validation): {skipped_count}")
    print(f"  ❌ Errors: {error_count}")
    print(f"  📁 Total benchmarks found: {len(summary_df)}")

    return results


def validate_output_data(results: List[Dict[str, Any]]) -> bool:
    """Validate that the output data has the correct structure."""
    if not results:
        print("❌ No results to save")
        return False

    print(f"📊 Validating {len(results)} benchmark results")

    for i, result in enumerate(results):
        # Check required fields
        if "system" not in result:
            print(f"❌ Result {i}: Missing 'system' field")
            return False

        # Check that we have metric data
        metric_count = 0
        for key in result.keys():
            if key != "system" and not key.endswith(" Error"):
                metric_count += 1

        if metric_count == 0:
            print(f"❌ Result {i}: No metric data found")
            return False

        print(f"  ✅ Result {i}: {result['system']} with {metric_count} metrics")

    print("✅ Output data validation passed")
    return True


def format_json_output(results: List[Dict[str, Any]]) -> str:
    """Format the results as a JSON string with proper indentation."""
    try:
        json_string = json.dumps(results, indent=2, ensure_ascii=False)
        print(f"✅ Successfully formatted JSON output ({len(json_string)} characters)")
        return json_string
    except Exception as e:
        print(f"❌ Error formatting JSON: {e}")
        raise


def create_output_directory(output_path: str) -> None:
    """Create output directory if it doesn't exist."""
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 Created output directory: {output_dir}")


def save_benchmark_summary_json(results: List[Dict[str, Any]], output_path: str) -> None:
    """Save benchmark summary to JSON file."""
    print(f"💾 Saving {len(results)} benchmark results to {output_path}")

    # Validate output data (PHASE 7 - IMPLEMENTED)
    if not validate_output_data(results):
        raise ValueError("Output data validation failed")

    # Create output directory if needed
    create_output_directory(output_path)

    # Format JSON output (PHASE 7 - IMPLEMENTED)
    json_string = format_json_output(results)

    # Save to file
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(json_string)
        print(f"✅ Successfully saved JSON to {output_path}")

        # Print file size
        file_size = os.path.getsize(output_path)
        print(f"📄 File size: {file_size} bytes")

    except Exception as e:
        print(f"❌ Error saving JSON file: {e}")
        raise


def main():
    """Main function to orchestrate the benchmark summary creation."""
    print("🚀 Starting benchmark summary JSON creation")
    print("-" * 50)

    # Configuration
    TEMP_DIR = "temp"
    CROSS_BENCHMARK_CSV = f"{TEMP_DIR}/cross_benchmark_summary.csv"
    OUTPUT_PATH = "benchmark_summary.json"
    MAX_BENCHMARKS = None  # Process all benchmarks

    print(f"📁 Temp directory: {TEMP_DIR}")
    print(f"📊 Cross-benchmark CSV: {CROSS_BENCHMARK_CSV}")
    print(f"💾 Output path: {OUTPUT_PATH}")
    print(f"🔢 Max benchmarks to process: {MAX_BENCHMARKS}")
    print("-" * 50)

    # Check if temp directory exists
    if not os.path.exists(TEMP_DIR):
        print(f"❌ Temp directory not found: {TEMP_DIR}")
        print("Please run run_cross_benchmark_analysis.py first")
        return

    # Check if cross-benchmark CSV exists
    if not os.path.exists(CROSS_BENCHMARK_CSV):
        print(f"❌ Cross-benchmark CSV not found: {CROSS_BENCHMARK_CSV}")
        print("Please run run_cross_benchmark_analysis.py first")
        return

    print("✅ Required files found")

    # Load cross-benchmark data (PHASE 2 - IMPLEMENTED)
    print("🔄 Loading cross-benchmark data...")
    try:
        summary_df = load_cross_benchmark_data(CROSS_BENCHMARK_CSV)
        print("📊 Loaded {len(summary_df)} benchmarks from CSV")

        # Show all benchmarks found
        if len(summary_df) > 0:
            print("📋 All benchmarks found:")
            for i, row in summary_df.iterrows():
                print(f"  {i + 1}. {row['benchmark']}: {row.get('overall_avg', 'N/A'):.4f}")
        else:
            print("⚠️  No benchmarks found in CSV")

    except Exception as e:
        print(f"❌ Error loading cross-benchmark data: {e}")
        return

    # Process benchmarks (PHASE 3, 4, 5 & 6 - IMPLEMENTED)
    print("🔄 Processing and formatting benchmarks with validation...")
    results = process_all_benchmarks(TEMP_DIR, MAX_BENCHMARKS)

    print(f"\n📊 Processed {len(results)} benchmarks")

    # Save results (PHASE 7 - IMPLEMENTED)
    print("💾 Saving results...")
    try:
        save_benchmark_summary_json(results, OUTPUT_PATH)
        print(f"\n🎉 Success! JSON saved to: {OUTPUT_PATH}")
        print("📄 You can now use the benchmark summary JSON file")
    except Exception as e:
        print(f"❌ Error saving results: {e}")
        return

    print("\n🎉 Benchmark summary creation completed!")


if __name__ == "__main__":
    main()
