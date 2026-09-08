#!/usr/bin/env python3
"""
Cross-benchmark analysis orchestration script.
Downloads qa-benchmarks volume and processes each benchmark folder.
"""

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
from analysis.analyze_single_benchmark import analyze_single_benchmark_folder
import logging

logger = logging.getLogger(__name__)


def download_modal_volume(volume_name: str, download_path: str) -> None:
    """Download entire modal volume to local directory."""
    print(f"📥 Downloading modal volume: {volume_name}")

    # Create download directory if it doesn't exist
    Path(download_path).mkdir(parents=True, exist_ok=True)

    original_dir = os.getcwd()
    os.chdir(download_path)

    try:
        cmd = ["modal", "volume", "get", volume_name, "/"]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print("✅ Successfully downloaded modal volume")
        else:
            print(f"❌ Error downloading volume: {result.stderr}")
            sys.exit(1)
    finally:
        os.chdir(original_dir)


def get_benchmark_folders(volume_path: str) -> list:
    """Get list of benchmark folders from downloaded volume."""
    volume_dir = Path(volume_path)

    if not volume_dir.exists():
        print(f"❌ Volume directory does not exist: {volume_path}")
        return []

    benchmark_folders = []
    for item in volume_dir.iterdir():
        if item.is_dir() and not item.name.startswith("."):
            benchmark_folders.append(item.name)

    print(f"📁 Found {len(benchmark_folders)} benchmark folders")
    return sorted(benchmark_folders)


def check_evaluated_folder_exists(benchmark_path: str) -> bool:
    """Check if evaluated folder exists and contains JSON files."""
    evaluated_path = Path(benchmark_path) / "evaluated"

    if not evaluated_path.exists():
        print(f"⚠️  No evaluated folder found: {evaluated_path}")
        return False

    json_files = list(evaluated_path.glob("*.json"))
    if not json_files:
        print(f"⚠️  No JSON files found in evaluated folder: {evaluated_path}")
        return False

    print(f"✅ Found {len(json_files)} JSON files in evaluated folder")
    return True


def check_analysis_files_exist(benchmark_path: str) -> bool:
    """Check if analysis files already exist for this benchmark."""
    analysis_path = Path(benchmark_path) / "analysis"

    if not analysis_path.exists():
        return False

    # Check for any CSV files in analysis folder
    csv_files = list(analysis_path.glob("*.csv"))
    return len(csv_files) > 0


def create_analysis_folder(benchmark_path: str) -> str:
    """Create analysis folder for benchmark if it doesn't exist."""
    analysis_path = Path(benchmark_path) / "analysis"
    analysis_path.mkdir(parents=True, exist_ok=True)
    return str(analysis_path)


def process_single_benchmark(benchmark_folder: str, volume_path: str) -> bool:
    """Process a single benchmark folder."""
    benchmark_path = Path(volume_path) / benchmark_folder

    print(f"\n🔄 Processing benchmark: {benchmark_folder}")

    # Check if evaluated folder exists
    if not check_evaluated_folder_exists(benchmark_path):
        return False

    # Check if analysis already exists
    if check_analysis_files_exist(benchmark_path):
        print(f"⏭️  Analysis files already exist, skipping: {benchmark_folder}")
        return False

    try:
        # Run analysis for this benchmark
        analyze_single_benchmark_folder(str(benchmark_path))
        print(f"✅ Successfully processed: {benchmark_folder}")
        return True
    except Exception as e:
        logger.debug("Ignoring exception in process_single_benchmark", exc_info=True)
        print(f"❌ Error processing {benchmark_folder}: {e}")
        return False


def print_progress_update(current: int, total: int) -> None:
    """Print progress update every 1/5 of total."""
    if current % max(1, total // 5) == 0:
        print(f"📊 Progress: {current}/{total} benchmarks processed")


def process_all_benchmarks(volume_path: str) -> dict:
    """Process all benchmark folders with progress tracking."""
    benchmark_folders = get_benchmark_folders(volume_path)

    results = {"processed": [], "skipped": [], "failed": []}

    if not benchmark_folders:
        print("❌ No benchmark folders found")
        return results

    print(f"\n🚀 Starting analysis of {len(benchmark_folders)} benchmarks")

    for i, folder in enumerate(benchmark_folders):
        print_progress_update(i, len(benchmark_folders))

        success = process_single_benchmark(folder, volume_path)

        if success:
            results["processed"].append(folder)
        else:
            results["skipped"].append(folder)

    return results


def create_cross_benchmark_summary(volume_path: str, results: dict) -> None:
    """Create a summary CSV with average metrics from all processed benchmarks."""
    print("\n📊 Creating cross-benchmark summary...")

    summary_data = []
    metrics = ["directllm_correctness", "deepeval_correctness", "EM", "f1"]

    for benchmark_folder in results["processed"]:
        benchmark_path = Path(volume_path) / benchmark_folder
        aggregate_csv_path = benchmark_path / "analysis" / "metrics_aggregate.csv"

        if aggregate_csv_path.exists():
            try:
                # Read the aggregate metrics CSV
                df = pd.read_csv(aggregate_csv_path, index_col=0)

                # Calculate average of averages for each metric
                benchmark_summary = {"benchmark": benchmark_folder, "questions_count": len(df)}

                for metric in metrics:
                    mean_col = f"{metric}_mean"
                    if mean_col in df.columns:
                        benchmark_summary[f"{metric}_avg"] = df[mean_col].mean()
                    else:
                        benchmark_summary[f"{metric}_avg"] = None

                summary_data.append(benchmark_summary)
                print(f"  ✅ Added {benchmark_folder}: {len(df)} questions")

            except Exception as e:
                logger.debug("Ignoring exception in create_cross_benchmark_summary", exc_info=True)
                print(f"  ❌ Error reading {benchmark_folder}: {e}")
        else:
            print(f"  ⚠️  No aggregate file found for {benchmark_folder}")

    if summary_data:
        # Create summary DataFrame
        summary_df = pd.DataFrame(summary_data)

        # Sort by overall performance (average of all metrics)
        metric_cols = [f"{metric}_avg" for metric in metrics]
        valid_metrics = [col for col in metric_cols if col in summary_df.columns]

        if valid_metrics:
            summary_df["overall_avg"] = summary_df[valid_metrics].mean(axis=1)
            summary_df = summary_df.sort_values("overall_avg", ascending=False)

        # Save summary CSV
        summary_path = Path(volume_path) / "cross_benchmark_summary.csv"
        summary_df.to_csv(summary_path, index=False)

        print(f"📈 Cross-benchmark summary saved to: {summary_path}")
        print(f"📊 Processed {len(summary_df)} benchmarks")

        # Print top performers
        print("\n🏆 Top 3 performers:")
        for i, row in summary_df.head(3).iterrows():
            print(f"  {i + 1}. {row['benchmark']}: {row.get('overall_avg', 'N/A'):.4f}")
    else:
        print("❌ No benchmark data found for summary")


def print_summary(results: dict) -> None:
    """Print summary of processing results."""
    print("\n" + "=" * 50)
    print("📊 PROCESSING SUMMARY")
    print("=" * 50)

    print(f"✅ Successfully processed: {len(results['processed'])}")
    print(f"⏭️  Skipped (already exists): {len(results['skipped'])}")
    print(f"❌ Failed: {len(results['failed'])}")

    if results["processed"]:
        print("\n📁 Processed benchmarks:")
        for folder in results["processed"]:
            print(f"  - {folder}")

    if results["skipped"]:
        print("\n⏭️  Skipped benchmarks:")
        for folder in results["skipped"]:
            print(f"  - {folder}")


def main():
    """Main orchestration function."""
    VOLUME_NAME = "qa-benchmarks"
    DOWNLOAD_PATH = "temp"

    print("🚀 Starting cross-benchmark analysis")
    print(f"📦 Modal volume: {VOLUME_NAME}")
    print(f"📁 Download path: {DOWNLOAD_PATH}")
    print("-" * 50)

    # Download modal volume
    download_modal_volume(VOLUME_NAME, DOWNLOAD_PATH)

    # Process all benchmarks
    results = process_all_benchmarks(DOWNLOAD_PATH)

    # Create cross-benchmark summary
    create_cross_benchmark_summary(DOWNLOAD_PATH, results)

    # Print summary
    print_summary(results)

    print("\n🎉 Cross-benchmark analysis completed!")


if __name__ == "__main__":
    main()
