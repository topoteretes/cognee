import argparse
import json
import subprocess
import sys
from pathlib import Path

from swebench.harness.utils import load_swebench_dataset
from swebench.inference.make_datasets.create_instance import PATCH_EXAMPLE

from cognee.api.v1.cognify.code_graph_pipeline import run_code_graph_pipeline
from cognee.api.v1.search import SearchType
from cognee.infrastructure.llm.get_llm_client import get_llm_client
from cognee.infrastructure.llm.prompts import read_query_prompt
from cognee.modules.retrieval.brute_force_triplet_search import \
    brute_force_triplet_search
from cognee.shared.utils import render_graph
from evals.eval_utils import download_github_repo, retrieved_edges_to_string


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


async def generate_patch_with_cognee(instance, llm_client, search_type=SearchType.CHUNKS):
    repo_path = download_github_repo(instance, '../RAW_GIT_REPOS')
    pipeline = await run_code_graph_pipeline(repo_path)

    async for result in pipeline:
        print(result)

    print('Here we have the repo under the repo_path')

    await render_graph(None, include_labels=True, include_nodes=True)

    problem_statement = instance['problem_statement']
    instructions = read_query_prompt("patch_gen_kg_instructions.txt")

    retrieved_edges = await brute_force_triplet_search(problem_statement, top_k=3,
                                                       collections=["data_point_source_code", "data_point_text"])

    retrieved_edges_str = retrieved_edges_to_string(retrieved_edges)

    prompt = "\n".join([
        problem_statement,
        "<patch>",
        PATCH_EXAMPLE,
        "</patch>",
        "These are the retrieved edges:",
        retrieved_edges_str
    ])

    llm_client = get_llm_client()
    answer_prediction = await llm_client.acreate_structured_output(
        text_input=prompt,
        system_prompt=instructions,
        response_model=str,
    )

    return answer_prediction


async def generate_patch_without_cognee(instance, llm_client):
    instructions = read_query_prompt("patch_gen_instructions.txt")

    answer_prediction = await llm_client.acreate_structured_output(
        text_input=instance["text"],
        system_prompt=instructions,
        response_model=str,
    )
    return answer_prediction


async def get_preds(dataset, with_cognee=True):
    llm_client = get_llm_client()

    if with_cognee:
        model_name = "with_cognee"
        pred_func = generate_patch_with_cognee
    else:
        model_name = "without_cognee"
        pred_func = generate_patch_without_cognee

    futures = [
        (instance["instance_id"], pred_func(instance, llm_client))
        for instance in dataset
    ]
    model_patches = await asyncio.gather(*[x[1] for x in futures])

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
        predictions_path = "preds.json"
        preds = await get_preds(swe_dataset, with_cognee=not args.cognee_off)
        with open(predictions_path, "w") as file:
            json.dump(preds, file)

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


if __name__ == "__main__":
    import asyncio

    asyncio.run(main(), debug=True)
