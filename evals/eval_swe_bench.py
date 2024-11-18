import json
import subprocess
from pathlib import Path

from swebench.harness.utils import load_swebench_dataset
from swebench.inference.make_datasets.create_instance import PATCH_EXAMPLE

import cognee
from cognee.api.v1.cognify.code_graph_pipeline import code_graph_pipeline
from cognee.api.v1.search import SearchType
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.llm.get_llm_client import get_llm_client
from evals.eval_utils import download_instances


async def cognee_and_llm(dataset, search_type=SearchType.CHUNKS):
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    dataset_name = "SWE_test_data"
    code_text = dataset[0]["text"]
    await cognee.add([code_text], dataset_name)
    await code_graph_pipeline([dataset_name])
    graph_engine = await get_graph_engine()
    with open(graph_engine.filename, "r") as f:
        graph_str = f.read()

    problem_statement = dataset[0]['problem_statement']
    instructions = (
        "I need you to solve this issue by looking at the provided knowledge graph and by "
        + "generating a single patch file that I can apply directly to this repository "
        + "using git apply. Please respond with a single patch "
        + "file in the following format."
    )

    prompt = "\n".join([
        instructions,
        "<patch>",
        PATCH_EXAMPLE,
        "</patch>",
        "This is the knowledge graph:",
        graph_str
    ])

    llm_client = get_llm_client()
    answer_prediction = llm_client.create_structured_output(
        text_input=problem_statement,
        system_prompt=prompt,
        response_model=str,
    )
    return answer_prediction


async def llm_on_preprocessed_data(dataset):
    problem_statement = dataset[0]['problem_statement']
    prompt = dataset[0]["text"]

    llm_client = get_llm_client()
    answer_prediction = llm_client.create_structured_output(
        text_input=problem_statement,
        system_prompt=prompt,
        response_model=str,
    )
    return answer_prediction


async def get_preds(dataset, with_cognee=True):
    if with_cognee:
        text_output = await cognee_and_llm(dataset)
        model_name = "with_cognee"
    else:
        text_output = await llm_on_preprocessed_data(dataset)
        model_name = "without_cognee"

    preds = [{"instance_id": dataset[0]["instance_id"],
              "model_patch": text_output,
              "model_name_or_path": model_name}]

    return preds


async def main():
    swe_dataset = load_swebench_dataset(
        'princeton-nlp/SWE-bench', split='test')
    swe_dataset_preprocessed = load_swebench_dataset(
        'princeton-nlp/SWE-bench_bm25_13K', split='test')
    test_data = swe_dataset[:1]
    test_data_preprocessed = swe_dataset_preprocessed[:1]
    assert test_data[0]["instance_id"] == test_data_preprocessed[0]["instance_id"]
    filepath = Path("SWE-bench_testsample")
    if filepath.exists():
        from datasets import Dataset
        dataset = Dataset.load_from_disk(filepath)
    else:
        dataset = download_instances(test_data, filepath)

    cognee_preds = await get_preds(dataset, with_cognee=True)
    # nocognee_preds = await get_preds(dataset, with_cognee=False)
    with open("withcognee.json", "w") as file:
        json.dump(cognee_preds, file)

    subprocess.run(["python", "-m", "swebench.harness.run_evaluation",
                    "--dataset_name", 'princeton-nlp/SWE-bench',
                    "--split", "test",
                    "--predictions_path",  "withcognee.json",
                    "--max_workers", "1",
                    "--instance_ids", test_data[0]["instance_id"],
                    "--run_id", "with_cognee"])

if __name__ == "__main__":
    import asyncio
    asyncio.run(main(), debug=True)
