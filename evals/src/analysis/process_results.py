import os
import pandas as pd
from analysis.get_results import read_results, validate_folder_results
from analysis.analyze_results import create_aggregate_metrics_df


def process_results(dir_path: str) -> dict:
    """Read and validate results from the specified directory."""
    # Read results
    results = read_results(dir_path)

    # Validate results
    if not validate_folder_results(results):
        raise ValueError("Validation failed")

    print(f"Successfully processed {len(results)} files")
    return results


def transform_results(results: dict, metrics: list = None) -> dict:
    """Transform dictionary of lists into dictionary of dictionaries with questions as keys."""
    if metrics is None:
        metrics = ["directllm_correctness", "deepeval_correctness", "EM", "f1"]

    transformed = {}

    for filename, data_list in results.items():
        transformed[filename] = {}

        for item in data_list:
            question = item["question"]
            qa_dict = {"golden_answer": item["golden_answer"], "answer": item["answer"]}
            # Handle both dictionary format (with score key) and direct numeric format
            metrics_dict = {}
            for metric in metrics:
                metric_data = item["metrics"][metric]
                if isinstance(metric_data, dict):
                    metrics_dict[metric] = metric_data["score"]
                else:
                    metrics_dict[metric] = metric_data
            transformed[filename][question] = qa_dict | metrics_dict

    return transformed


def validate_question_consistency(transformed: dict) -> bool:
    """Validate that all files have the same question/golden_answer combinations."""
    if not transformed:
        print("No transformed data to validate")
        return False

    # Create sets of question/golden_answer combinations for each file
    file_combinations = {}
    for filename, questions in transformed.items():
        combinations = set()
        for question, data in questions.items():
            combo = f"{question}|{data['golden_answer']}"
            combinations.add(combo)
        file_combinations[filename] = combinations

    # Get the first file's combinations as reference
    first_file = next(iter(file_combinations.keys()))
    reference_combinations = file_combinations[first_file]

    # Check if all files have the same combinations
    for filename, combinations in file_combinations.items():
        if combinations != reference_combinations:
            missing = reference_combinations - combinations
            extra = combinations - reference_combinations
            print(f"Question/golden_answer mismatch in {filename}:")
            if missing:
                print(f"  Missing: {missing}")
            if extra:
                print(f"  Extra: {extra}")
            return False

    print(
        f"Validation passed: all files have same {len(reference_combinations)} question/golden_answer combinations"
    )
    return True


def create_answers_df(transformed: dict, output_csv_path: str = None) -> pd.DataFrame:
    """Create dataframe with questions as rows and answers as columns."""

    # Get all questions (they should be the same across files)
    first_file = next(iter(transformed.keys()))
    questions = list(transformed[first_file].keys())

    # Create dataframe
    df = pd.DataFrame(index=questions)

    # Add golden_answer column
    df["golden_answer"] = [transformed[first_file][q]["golden_answer"] for q in questions]

    # Add one column per file's answers
    for filename, questions_data in transformed.items():
        df[f"answer_{filename}"] = [questions_data[q]["answer"] for q in questions]

    # Save to CSV if path provided
    if output_csv_path:
        df.to_csv(output_csv_path)
        print(f"Saved answers dataframe to {output_csv_path}")

    return df


def create_single_metric_df(
    transformed: dict, metric: str, save_folder: str = None, save_prefix: str = None
) -> pd.DataFrame:
    """Create a single dataframe for one metric, with questions as rows and files as columns."""
    # Get all questions (they should be the same across files)
    first_file = next(iter(transformed.keys()))
    questions = list(transformed[first_file].keys())

    df = pd.DataFrame(index=questions)

    # Add one column per file's scores for this metric
    for filename, questions_data in transformed.items():
        df[filename] = [questions_data[q][metric] for q in questions]

    # Save to CSV if folder exists
    if not save_folder:
        return df
    if not os.path.exists(save_folder):
        print(f"Save folder '{save_folder}' does not exist, skipping save for {metric}")
        return df
    filename = f"{save_prefix}_{metric}.csv" if save_prefix else f"{metric}.csv"
    csv_path = os.path.join(save_folder, filename)
    df.to_csv(csv_path)
    print(f"Saved {metric} dataframe to {csv_path}")

    return df


def create_all_metrics_df(
    transformed: dict, metrics: list = None, save_folder: str = None, save_prefix: str = None
) -> dict:
    """Create dataframes for all metrics, with questions as rows and files as columns."""
    if metrics is None:
        metrics = ["directllm_correctness", "deepeval_correctness", "EM", "f1"]

    metrics_dfs = {}

    for metric in metrics:
        metrics_dfs[metric] = create_single_metric_df(transformed, metric, save_folder, save_prefix)

    return metrics_dfs


if __name__ == "__main__":
    results = process_results("./data/cognee_graphsearch")
    print(f"Processed {len(results)} files")

    transformed = transform_results(results)
    print(f"Transformed results for {len(transformed)} files")

    if not validate_question_consistency(transformed):
        raise ValueError("Question consistency validation failed")

    answers_df = create_answers_df(transformed, output_csv_path="./temp/answers.csv")
    print(
        f"Created answers dataframe with {len(answers_df)} questions and {len(answers_df.columns)} columns"
    )

    metrics_dfs = create_all_metrics_df(transformed, save_folder="./temp", save_prefix="metrics")
    print(f"Created metrics dataframes: {list(metrics_dfs.keys())}")

    aggregate_df = create_aggregate_metrics_df(
        metrics_dfs,
        ["directllm_correctness", "deepeval_correctness", "EM", "f1"],
        save_folder="./temp",
        save_prefix="metrics",
    )
    print(f"Created aggregate metrics dataframe with {len(aggregate_df.columns)} columns")

    # Print averages of averages
    print("\nOverall averages:")
    for metric in ["directllm_correctness", "deepeval_correctness", "EM", "f1"]:
        mean_col = f"{metric}_mean"
        overall_avg = aggregate_df[mean_col].mean()
        print(f"  {metric}: {overall_avg:.4f}")
