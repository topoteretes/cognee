import asyncio
from evals.eval_on_hotpot import eval_on_QA_dataset, incremental_eval_on_QA_dataset
from evals.qa_eval_utils import get_combinations, save_results_as_image
import argparse
from pathlib import Path
import json


async def run_evals_on_paramset(paramset: dict, out_path: str):
    combinations = get_combinations(paramset)
    json_path = Path(out_path) / Path("results.json")
    results = {}
    for params in combinations:
        dataset = params["dataset"]
        num_samples = params["num_samples"]
        rag_option = params["rag_option"]

        if dataset not in results:
            results[dataset] = {}
        if num_samples not in results[dataset]:
            results[dataset][num_samples] = {}

        if rag_option == "cognee_incremental":
            result = await incremental_eval_on_QA_dataset(
                dataset,
                num_samples,
                paramset["metric_names"],
            )
            results[dataset][num_samples] |= result
        else:
            result = await eval_on_QA_dataset(
                dataset,
                rag_option,
                num_samples,
                paramset["metric_names"],
            )
            results[dataset][num_samples][rag_option] = result

        with open(json_path, "w") as file:
            json.dump(results, file, indent=1)

        save_results_as_image(results, out_path)

    return results


async def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--params_file", type=str, required=True, help="Which dataset to evaluate on"
    )
    parser.add_argument("--out_dir", type=str, help="Dir to save eval results")

    args = parser.parse_args()

    with open(args.params_file, "r") as file:
        parameters = json.load(file)

    await run_evals_on_paramset(parameters, args.out_dir)


if __name__ == "__main__":
    asyncio.run(main())
