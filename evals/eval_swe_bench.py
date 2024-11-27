import argparse
import json
import subprocess
import sys
from pathlib import Path

from datasets import Dataset
from swebench.harness.utils import load_swebench_dataset
from swebench.inference.make_datasets.create_instance import PATCH_EXAMPLE

import cognee

from cognee.shared.data_models import SummarizedContent
from cognee.shared.utils import render_graph
from cognee.tasks.repo_processor import (
    enrich_dependency_graph,
    expand_dependency_graph,
    get_repo_file_dependencies,
)
from cognee.tasks.storage import add_data_points
from cognee.tasks.summarization import summarize_code
from cognee.modules.pipelines import Task, run_tasks
from cognee.api.v1.cognify.code_graph_pipeline import code_graph_pipeline
from cognee.api.v1.search import SearchType
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.llm.get_llm_client import get_llm_client
from cognee.infrastructure.llm.prompts import read_query_prompt
from evals.eval_utils import download_instances


def check_install_package(package_name):
    """
    Check if a pip package is installed and install it if not.
    Returns True if package is/was installed successfully, False otherwise.
    """
    try:
        __import__(package_name)
        return True
    except ImportError:
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", package_name]
            )
            return True
        except subprocess.CalledProcessError:
            return False


<<<<<<< HEAD

async def generate_patch_with_cognee(instance, search_type=SearchType.CHUNKS):
=======
async def generate_patch_with_cognee(
    instance, search_type=SearchType.CHUNKS
):
>>>>>>> c4e3634 (Update eval_swe_bench)

    await cognee.prune.prune_data()
    await cognee.prune.prune_system()

    #dataset_name = "SWE_test_data"

    #await cognee.add('', dataset_name = dataset_name)

    # repo_path = download_github_repo(instance, '../RAW_GIT_REPOS')

    repo_path = '/Users/borisarzentar/Projects/graphrag'

    tasks = [
        Task(get_repo_file_dependencies),
        Task(add_data_points, task_config = { "batch_size": 50 }),
        Task(enrich_dependency_graph, task_config = { "batch_size": 50 }),
        Task(expand_dependency_graph, task_config = { "batch_size": 50 }),
        Task(add_data_points, task_config = { "batch_size": 50 }),
        # Task(summarize_code, summarization_model = SummarizedContent),
    ]

    pipeline = run_tasks(tasks, repo_path, "cognify_code_pipeline")

    async for result in pipeline:
        print(result)

    print('Here we have the repo under the repo_path')

    await render_graph(None, include_labels = True, include_nodes = True)

    problem_statement = instance['problem_statement']
    instructions = read_query_prompt("patch_gen_instructions.txt")

    graph_str = 'HERE WE SHOULD PASS THE TRIPLETS FROM GRAPHRAG'

    prompt = "\n".join(
        [
            instructions,
            "<patch>",
            PATCH_EXAMPLE,
            "</patch>",
            "This is the knowledge graph:",
            graph_str,
        ]
    )

    answer_prediction = await llm_client.acreate_structured_output(
        text_input=problem_statement,
        system_prompt=prompt,
        response_model=str,
    )

    return answer_prediction


async def generate_patch_without_cognee(instance, llm_client):
    problem_statement = instance['problem_statement']
    prompt = instance["text"]

    answer_prediction = await llm_client.acreate_structured_output(
        text_input=problem_statement,
        system_prompt=prompt,
        response_model=str,
    )
    return answer_prediction


async def get_preds(dataset, with_cognee=True):
    llm_client = get_llm_client()

    if with_cognee:
        model_name = "with_cognee"
        futures = [
            (instance["instance_id"], generate_patch_with_cognee(instance))
            for instance in dataset
        ]
    else:
        model_name = "without_cognee"
        futures = [
            (instance["instance_id"], generate_patch_without_cognee(instance, llm_client))
            for instance in dataset
        ]
    model_patches = await asyncio.gather(*[x[1] for x in futures])
<<<<<<< HEAD

=======
>>>>>>> c4e3634 (Update eval_swe_bench)
    preds = [
        {
            "instance_id": instance_id,
            "model_patch": model_patch,
            "model_name_or_path": model_name,
        }
        for (instance_id, _), model_patch in zip(futures, model_patches)
    ]

    return preds


async def main():
    parser = argparse.ArgumentParser(
        description="Run LLM predictions on SWE-bench dataset")
    parser.add_argument('--cognee_off', action='store_true')
    parser.add_argument("--max_workers", type=int, required=True)
    args = parser.parse_args()

    for dependency in ["transformers", "sentencepiece", "swebench"]:
        check_install_package(dependency)

    if args.cognee_off:
        dataset_name = 'princeton-nlp/SWE-bench_Lite_bm25_13K'
        dataset = load_swebench_dataset(dataset_name, split='test')
        predictions_path = "preds_nocognee.json"
        if not Path(predictions_path).exists():
            preds = await get_preds(dataset, with_cognee=False)
            with open(predictions_path, "w") as file:
                json.dump(preds, file)
    else:
        dataset_name = 'princeton-nlp/SWE-bench_Lite'
        swe_dataset = load_swebench_dataset(
            dataset_name, split='test')[:1]
        filepath = Path("SWE-bench_testsample")
        if filepath.exists():
            dataset = Dataset.load_from_disk(filepath)
        else:
            dataset = download_instances(swe_dataset, filepath)
        predictions_path = "preds.json"
        preds = await get_preds(dataset, with_cognee=not args.cognee_off)
        with open(predictions_path, "w") as file:
            json.dump(preds, file)

<<<<<<< HEAD

    subprocess.run(
        [
            "python",
            "-m",
            "swebench.harness.run_evaluation",
            "--dataset_name",
            dataset_name,
            "--split",
            "test",
            "--predictions_path",
            predictions_path,
            "--max_workers",
            str(args.max_workers),
            "--run_id",
            "test_run",
        ]
    )

=======
    subprocess.run(["python", "-m", "swebench.harness.run_evaluation",
                    "--dataset_name", dataset_name,
                    "--split", "test",
                    "--predictions_path",  predictions_path,
                    "--max_workers", "1",
                    "--run_id", "test_run"])
>>>>>>> c4e3634 (Update eval_swe_bench)

if __name__ == "__main__":
    import asyncio

    asyncio.run(main(), debug=True)
