import argparse
import json
import subprocess
from pathlib import Path

from datasets import Dataset
from swebench.harness.utils import load_swebench_dataset
from swebench.inference.make_datasets.create_instance import PATCH_EXAMPLE

import cognee
from cognee.api.v1.cognify.code_graph_pipeline import code_graph_pipeline
from cognee.api.v1.search import SearchType
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.llm.get_llm_client import get_llm_client
from cognee.infrastructure.llm.prompts import read_query_prompt
from cognee.modules.pipelines import Task, run_tasks
from cognee.modules.retrieval.brute_force_triplet_search import \
    brute_force_triplet_search
from cognee.shared.data_models import SummarizedContent
from cognee.shared.utils import render_graph
from cognee.tasks.repo_processor import (enrich_dependency_graph,
                                         expand_dependency_graph,
                                         get_repo_file_dependencies)
from cognee.tasks.storage import add_data_points
from cognee.tasks.summarization import summarize_code
from evals.eval_utils import (delete_repo, download_github_repo,
                              download_instances, ingest_repos)


def node_to_string(node):
    text = node.attributes["text"]
    type = node.attributes["type"]
    return f"Node(id: {node.id}, type: {type}, description: {text})"
def retrieved_edges_to_string(retrieved_edges):
    edge_strings = []
    for edge in retrieved_edges:
        relationship_type = edge.attributes["relationship_type"]
        edge_str = f"{node_to_string(edge.node1)} {relationship_type} {node_to_string(edge.node2)}"
        edge_strings.append(edge_str)
    return "\n".join(edge_strings)  

async def generate_patch_with_cognee(instance):
    await cognee.prune.prune_data()
    await cognee.prune.prune_system()

    #dataset_name = "SWE_test_data"

    #await cognee.add('', dataset_name = dataset_name)

    # repo_path = download_github_repo(instance, '../RAW_GIT_REPOS')

    repo_path = '../minimal_repo'
    tasks = [
        Task(get_repo_file_dependencies),
        Task(add_data_points, task_config = { "batch_size": 50 }),
        Task(enrich_dependency_graph, task_config = { "batch_size": 50 }),
        Task(expand_dependency_graph, task_config = { "batch_size": 50 }),
        Task(add_data_points, task_config = { "batch_size": 50 }),
        Task(summarize_code, summarization_model = SummarizedContent),
    ]

    pipeline = run_tasks(tasks, repo_path, "cognify_code_pipeline")
        
    async for result in pipeline:
        print(result)

    print('Here we have the repo under the repo_path')

    await render_graph(None, include_labels = True, include_nodes = True)

    problem_statement = instance['problem_statement']
    instructions = read_query_prompt("patch_gen_instructions.txt")

    retrieved_edges = await brute_force_triplet_search(problem_statement, top_k = 3)
    
    retrieved_edges_str = retrieved_edges_to_string(retrieved_edges)

    prompt = "\n".join([
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


async def generate_patch_without_cognee(instance):
    problem_statement = instance['problem_statement']
    prompt = instance["text"]

    llm_client = get_llm_client()
    answer_prediction = await llm_client.acreate_structured_output(
        text_input=problem_statement,
        system_prompt=prompt,
        response_model=str,
    )
    return answer_prediction


async def get_preds(dataset, with_cognee=True):
    if with_cognee:
        model_name = "with_cognee"
        pred_func = generate_patch_with_cognee
    else:
        model_name = "without_cognee"
        pred_func = generate_patch_without_cognee


    for instance in dataset:
        await pred_func(instance)

    preds = [{"instance_id": instance["instance_id"],
              "model_patch": await pred_func(instance),
              "model_name_or_path": model_name} for instance in dataset]
  
    return preds


async def main():
    parser = argparse.ArgumentParser(
        description="Run LLM predictions on SWE-bench dataset")
    parser.add_argument('--cognee_off', action='store_true')
    args = parser.parse_args()

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

    subprocess.run(["python", "-m", "swebench.harness.run_evaluation",
                    "--dataset_name", dataset_name,
                    "--split", "test",
                    "--predictions_path",  predictions_path,
                    "--max_workers", "1",
                    "--run_id", "test_run"])

if __name__ == "__main__":
    import asyncio

    asyncio.run(main(), debug=True)
