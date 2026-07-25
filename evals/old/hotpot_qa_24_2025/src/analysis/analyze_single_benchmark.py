import os
import subprocess
import sys
from pathlib import Path
from analysis.process_results import (
    process_results,
    transform_results,
    validate_question_consistency,
    create_answers_df,
    create_all_metrics_df,
)
from analysis.analyze_results import create_aggregate_metrics_df, cumulative_all_metrics_analysis


def create_project_structure(project_dir: str) -> str:
    """Create project folder structure with an analysis subfolder."""
    project_path = Path(project_dir)
    analysis_path = project_path / "analysis"

    # Create directories
    analysis_path.mkdir(parents=True, exist_ok=True)

    print(f"Created project structure: {project_path}")
    print(f"  Analysis folder: {analysis_path}")

    return str(project_path)


def process_and_validate_data(data_folder: str) -> tuple:
    """Process and validate the downloaded data."""
    print("Processing and validating data...")

    # Process results
    results = process_results(data_folder)

    # Transform results
    transformed = transform_results(results)

    # Validate question consistency
    if not validate_question_consistency(transformed):
        raise ValueError("Question consistency validation failed")

    print("Data processing and validation completed successfully")
    return results, transformed


def create_all_dataframes(transformed: dict, analysis_folder: str) -> None:
    """Create all dataframes and save them to the analysis folder."""
    print("Creating dataframes...")

    # Create answers dataframe
    answers_df = create_answers_df(transformed, output_csv_path=f"{analysis_folder}/answers.csv")
    print(f"Created answers dataframe with {len(answers_df)} questions")

    # Create metrics dataframes
    metrics_dfs = create_all_metrics_df(
        transformed, save_folder=analysis_folder, save_prefix="metrics"
    )
    print(f"Created metrics dataframes: {list(metrics_dfs.keys())}")

    # Create aggregate metrics dataframe
    aggregate_df = create_aggregate_metrics_df(
        metrics_dfs,
        ["directllm_correctness", "deepeval_correctness", "EM", "f1"],
        save_folder=analysis_folder,
        save_prefix="metrics",
    )
    print(f"Created aggregate metrics dataframe with {len(aggregate_df.columns)} columns")

    # Create cumulative analysis
    cumulative_dfs = cumulative_all_metrics_analysis(
        aggregate_df,
        metrics=["directllm_correctness", "deepeval_correctness", "EM", "f1"],
        save_folder=analysis_folder,
        save_prefix="cumulative",
    )
    print(f"Created cumulative analysis for {len(cumulative_dfs)} metrics")

    # Print overall averages
    print("\nOverall averages:")
    for metric in ["directllm_correctness", "deepeval_correctness", "EM", "f1"]:
        mean_col = f"{metric}_mean"
        overall_avg = aggregate_df[mean_col].mean()
        print(f"  {metric}: {overall_avg:.4f}")


def analyze_single_benchmark_folder(
    benchmark_folder_path: str, volume_name: str = "qa-benchmarks"
) -> None:
    """Analyze a single benchmark folder from the modal volume."""
    print(f"Starting analysis for benchmark folder: {benchmark_folder_path}")
    print(f"Modal volume: {volume_name}")
    print("-" * 50)

    try:
        # Create project structure
        project_path = create_project_structure(benchmark_folder_path)
        analysis_folder = f"{project_path}/analysis"

        # Data is already downloaded locally, just process the evaluated folder
        evaluated_folder = f"{benchmark_folder_path}/evaluated"
        if not os.path.exists(evaluated_folder):
            raise FileNotFoundError(f"Evaluated folder not found: {evaluated_folder}")

        # Process and validate data from evaluated folder
        results, transformed = process_and_validate_data(evaluated_folder)

        # Create all dataframes
        create_all_dataframes(transformed, analysis_folder)

        print("\n" + "=" * 50)
        print("Analysis workflow completed successfully!")
        print(f"Results saved in: {analysis_folder}")

    except Exception as e:
        print(f"Error during analysis: {e}")
        raise


def main(project_dir: str, volume_name: str):
    """Main function to run the complete analysis workflow."""
    analyze_single_benchmark_folder(project_dir, volume_name)


if __name__ == "__main__":
    # Configuration variables - set these manually
    PROJECT_DIR = "temp/cognee_rag"
    MODAL_VOLUME_NAME = "qa-benchmarks"

    main(PROJECT_DIR, MODAL_VOLUME_NAME)
