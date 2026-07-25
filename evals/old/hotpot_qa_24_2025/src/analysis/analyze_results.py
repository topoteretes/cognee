import os
import pandas as pd


def create_aggregate_metrics_df(
    metrics_dfs: dict, metrics: list, save_folder: str = None, save_prefix: str = None
) -> pd.DataFrame:
    """Create aggregate dataframe with mean and std for each metric across files."""
    # Check that all requested metrics exist
    missing_metrics = [m for m in metrics if m not in metrics_dfs]
    if missing_metrics:
        raise ValueError(f"Metrics not found in metrics_dfs: {missing_metrics}")

    # Get questions from first metric dataframe
    questions = metrics_dfs[metrics[0]].index

    aggregate_df = pd.DataFrame(index=questions)

    for metric in metrics:
        df = metrics_dfs[metric]
        # Calculate mean and std across files (columns)
        aggregate_df[f"{metric}_mean"] = df.mean(axis=1)
        aggregate_df[f"{metric}_std"] = df.std(axis=1)

    # Save to CSV if folder exists
    if not save_folder:
        return aggregate_df
    if not os.path.exists(save_folder):
        print(f"Save folder '{save_folder}' does not exist, skipping save for aggregate metrics")
        return aggregate_df
    filename = f"{save_prefix}_aggregate.csv" if save_prefix else "aggregate_metrics.csv"
    csv_path = os.path.join(save_folder, filename)
    aggregate_df.to_csv(csv_path)
    print(f"Saved aggregate metrics dataframe to {csv_path}")

    return aggregate_df


def cumulative_single_metric_analysis(
    metric: str, aggregate_df: pd.DataFrame, save_folder: str = None, save_prefix: str = None
) -> pd.DataFrame:
    """Create cumulative analysis for a single metric, ordered by best results first."""
    # Get the mean column for the specified metric
    mean_col = f"{metric}_mean"
    if mean_col not in aggregate_df.columns:
        raise ValueError(f"Metric '{metric}' not found in aggregate_df columns")

    # Create a copy with just the metric mean, sorted descending
    analysis_df = aggregate_df[[mean_col]].copy()
    analysis_df = analysis_df.sort_values(by=mean_col, ascending=False)

    # Calculate cumulative average
    analysis_df["cumulative_avg"] = analysis_df[mean_col].expanding().mean()

    # Save to CSV if folder exists
    if not save_folder:
        return analysis_df
    if not os.path.exists(save_folder):
        print(
            f"Save folder '{save_folder}' does not exist, skipping save for {metric} cumulative analysis"
        )
        return analysis_df
    filename = (
        f"{save_prefix}_{metric}_cumulative.csv" if save_prefix else f"{metric}_cumulative.csv"
    )
    csv_path = os.path.join(save_folder, filename)
    analysis_df.to_csv(csv_path)
    print(f"Saved {metric} cumulative analysis to {csv_path}")

    return analysis_df


def cumulative_all_metrics_analysis(
    aggregate_df: pd.DataFrame,
    metrics: list = None,
    save_folder: str = None,
    save_prefix: str = None,
) -> dict:
    """Create cumulative analysis for all metrics, ordered by best results first."""
    if metrics is None:
        metrics = ["directllm_correctness", "deepeval_correctness", "EM", "f1"]

    analysis_dfs = {}

    for metric in metrics:
        analysis_dfs[metric] = cumulative_single_metric_analysis(
            metric, aggregate_df, save_folder, save_prefix
        )

    return analysis_dfs


if __name__ == "__main__":
    # Read the previously saved aggregate metrics CSV
    aggregate_csv_path = "./temp/metrics_aggregate.csv"

    if not os.path.exists(aggregate_csv_path):
        print(f"Aggregate metrics file not found: {aggregate_csv_path}")
        print("Please run process_results.py first to generate the aggregate metrics")
    else:
        # Read the aggregate DataFrame
        aggregate_df = pd.read_csv(aggregate_csv_path, index_col=0)
        print(
            f"Loaded aggregate metrics with {len(aggregate_df)} questions and {len(aggregate_df.columns)} columns"
        )

        # Generate cumulative analysis for all metrics
        cumulative_dfs = cumulative_all_metrics_analysis(
            aggregate_df,
            metrics=["directllm_correctness", "deepeval_correctness", "EM", "f1"],
            save_folder="./temp",
            save_prefix="cumulative",
        )

        print(f"Generated cumulative analysis for {len(cumulative_dfs)} metrics")

        # Print summary statistics
        print("\nCumulative analysis summary:")
        for metric, df in cumulative_dfs.items():
            final_avg = df["cumulative_avg"].iloc[-1]
            print(f"  {metric}: final cumulative average = {final_avg:.4f}")
