import asyncio
import itertools
from evals.eval_on_hotpot import eval_on_QA_dataset
import json
import pandas as pd
import matplotlib.pyplot as plt


def save_table_as_image(df, image_path):
    plt.figure(figsize=(10, 6))
    plt.axis("tight")
    plt.axis("off")
    plt.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, loc="center")
    plt.title(f"{df.index.name}")
    plt.savefig(image_path, bbox_inches="tight")
    plt.close()


parameters = {
    "dataset": ["hotpotqa"],  # "2wikimultihop"],
    "rag_option": ["no_rag", "cognee", "simple_rag", "brute_force"],
    "num_samples": [2],
    "metric_names": ["Correctness", "Comprehensiveness"],
}

# Generate the cross product of all parameter values
params_for_combos = {k: v for k, v in parameters.items() if k != "metric_name"}
keys, values = zip(*params_for_combos.items())
combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]


# Main async function to run all combinations concurrently
async def main():
    results = {}
    for params in combinations:
        dataset = params["dataset"]
        num_samples = params["num_samples"]
        rag_option = params["rag_option"]

        result = await eval_on_QA_dataset(
            dataset,
            rag_option,
            num_samples,
            parameters["metric_names"],
        )

        # Initialize nested structure if needed
        if dataset not in results:
            results[dataset] = {}
        if num_samples not in results[dataset]:
            results[dataset][num_samples] = {}

        # Update the nested dictionary
        results[dataset][num_samples][rag_option] = result

        # Save results as JSON
        json_path = "results.json"
        with open(json_path, "w") as file:
            json.dump(results, file, indent=1)

    # Convert to tables and save images
    for dataset, num_samples_data in results.items():
        for num_samples, table_data in num_samples_data.items():
            df = pd.DataFrame.from_dict(table_data, orient="index")
            df.index.name = f"Dataset: {dataset}, Num Samples: {num_samples}"
            image_path = f"table_{dataset}_{num_samples}.png"
            save_table_as_image(df, image_path)


if __name__ == "__main__":
    asyncio.run(main())
