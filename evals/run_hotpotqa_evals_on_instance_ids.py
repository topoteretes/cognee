import os
import asyncio
import json
import cognee
import signal


from evals.qa_context_provider_utils import qa_context_providers
from evals.qa_dataset_utils import load_hotpotqa_instances_by_ids
from evals.eval_on_hotpot import get_answers_for_instances, calculate_eval_metrics
from evals.qa_eval_utils import get_combinations
from evals.qa_metrics_utils import get_metrics


async def run_hotpotqa_evals_on_instance_ids(paramset: dict):
    """Runs evaluations using instance IDs from paramset, gets answers, and calculates evaluation metrics."""
    if "hotpotqa_instance_ids" not in paramset:
        raise ValueError("paramset must contain 'hotpotqa_instance_ids'.")

    instances = load_hotpotqa_instances_by_ids(paramset["hotpotqa_instance_ids"])
    combinations = get_combinations(paramset)
    results = {}

    for params in combinations:
        rag_option = params["rag_option"]
        metric_names = paramset["metric_names"]

        answers = await get_answers_for_instances(
            instances,
            context_provider=qa_context_providers[rag_option],
            answers_filename=None,
            contexts_filename=None,
            save_answers=False,
        )

        eval_metrics = get_metrics(metric_names)["deepeval_metrics"]
        result_metrics = await calculate_eval_metrics(instances, answers, eval_metrics)
        results[rag_option] = result_metrics | {"answers": answers}

    return {"instance_ids": paramset["hotpotqa_instance_ids"], "results": results}


async def main():
    paramset = {
        "dataset": ["hotpotqa"],
        "rag_option": ["cognee_25q1", "brute_force_25q1"],
        # "rag_option": ["cognee", "brute_force"],
        "metric_names": ["F1", "EM", "Correctness"],
    }

    instance_ids_path = "./evals/instance_ids.json"
    with open(instance_ids_path) as f:
        instance_ids = json.load(f)

    instance_ids = instance_ids[:1]

    paramsets = [paramset.copy() | dict(hotpotqa_instance_ids=[id]) for id in instance_ids]

    tasks = [run_hotpotqa_evals_on_instance_ids(p) for p in paramsets]

    results = await asyncio.gather(*tasks)

    print("\nFinal Results:")

    for result in results:
        print(result)

    # dump results to json
    with open("./evals/results.json", "w") as results_file:
        json.dump(results, results_file, indent=2)

    os.kill(os.getpid(), signal.SIGTERM)


if __name__ == "__main__":
    asyncio.run(main())
