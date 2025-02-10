import modal
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
from evals.run_hotpotqa_evals_on_instance_ids import run_hotpotqa_evals_on_instance_ids


app = modal.App("qa-eval")

image = (
    modal.Image.from_dockerfile(path="Dockerfile_modal", force_build=False)
    .copy_local_file("pyproject.toml", "pyproject.toml")
    .copy_local_file("poetry.lock", "poetry.lock")
    .env(
        {
            "ENV": os.getenv("ENV"),
            "LLM_API_KEY": os.getenv("LLM_API_KEY"),
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        }
    )
    .poetry_install_from_file(poetry_pyproject_toml="pyproject.toml")
    .pip_install("protobuf", "h2", "deepeval")
)


@app.function(image=image, concurrency_limit=50, timeout=1800, retries=3)
async def modal_run_hotpotqa_evals_on_instance_ids(paramset: dict):
    """Wrapper function that calls run_hotpotqa_evals_on_instance_ids with Modal decorator"""
    return await run_hotpotqa_evals_on_instance_ids(paramset)


@app.local_entrypoint()
async def main():
    import datetime

    paramset = {
        "dataset": ["hotpotqa"],
        # "rag_option": ["cognee_25q1", "brute_force_25q1"],
        # "rag_option": ["no_rag", "cognee", "brute_force"],
        "rag_option": ["no_rag", "cognee", "brute_force", "cognee_25q1", "brute_force_25q1"],
        "metric_names": ["F1", "EM", "Correctness"],
    }

    instance_ids_path = "./evals/instance_ids.json"
    with open(instance_ids_path) as f:
        instance_ids = json.load(f)

    # instance_ids = instance_ids[:2]

    paramsets = [paramset.copy() | dict(hotpotqa_instance_ids=[id]) for id in instance_ids]

    tasks = [modal_run_hotpotqa_evals_on_instance_ids.remote.aio(p) for p in paramsets]

    results = await asyncio.gather(*tasks)

    print("\nFinal Results:")

    for result in results:
        print(result)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    rag_options = "_".join(paramset["rag_option"])
    results_filename = f"./evals/results_{timestamp}_{rag_options}.json"

    with open(results_filename, "w") as results_file:
        json.dump(results, results_file, indent=2)

    os.kill(os.getpid(), signal.SIGTERM)
