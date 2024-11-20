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
from evals.eval_utils import download_instances


async def generate_patch_with_cognee(instance, search_type=SearchType.CHUNKS):

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    dataset_name = "SWE_test_data"
    code_text = instance["text"]
    await cognee.add([code_text], dataset_name)
    await code_graph_pipeline([dataset_name])
    graph_engine = await get_graph_engine()
    with open(graph_engine.filename, "r") as f:
        graph_str = f.read()

    problem_statement = instance['problem_statement']
    instructions = read_query_prompt("patch_gen_instructions.txt")

    prompt = "\n".join([
        instructions,
        "<patch>",
        PATCH_EXAMPLE,
        "</patch>",
        "This is the knowledge graph:",
        graph_str
    ])

    llm_client = get_llm_client()
    answer_prediction = await llm_client.acreate_structured_output(
        text_input=problem_statement,
        system_prompt=prompt,
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
