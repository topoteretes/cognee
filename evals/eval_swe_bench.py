from swebench.harness.utils import load_swebench_dataset
from swebench.harness.run_evaluation import get_dataset_from_preds
from swebench.harness.run_evaluation import run_instances
from swebench.harness.test_spec import make_test_spec, TestSpec

import subprocess
from swebench.inference.make_datasets.create_instance import PATCH_EXAMPLE
from evals.eval_utils import download_instances
import cognee
from cognee.api.v1.cognify.code_graph_pipeline import code_graph_pipeline
from cognee.api.v1.search import SearchType
from pathlib import Path
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.llm.get_llm_client import get_llm_client

async def cognee_and_llm(dataset, search_type = SearchType.CHUNKS):
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata = True)

    dataset_name = "SWE_test_data"
    code_text = dataset[0]["text"][:100000]
    await cognee.add([code_text], dataset_name)
    await cognee.cognify([dataset_name])
    graph_engine = await get_graph_engine()
    with open(graph_engine.filename, "r") as f:
        graph_str  = f.read()
    
    problem_statement = dataset[0]['problem_statement']
    instructions = (
        f"I need you to solve this issue by looking at the provided knowledge graph and by "
        + f"generating a single patch file that I can apply directly to this repository "
        + f"using git apply. Please respond with a single patch "
        + f"file in the following format."  
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
                        text_input = problem_statement,
                        system_prompt = prompt,
                        response_model = str,
                        )
    return answer_prediction


async def llm_on_preprocessed_data(dataset):
    problem_statement = dataset[0]['problem_statement']
    prompt = dataset[0]["text"]
    
    llm_client = get_llm_client()
    answer_prediction = llm_client.create_structured_output(
                        text_input = problem_statement,
                        system_prompt = prompt, # TODO check if this is correct
                        response_model = str,
                        )
    return answer_prediction

async def get_preds(dataset, with_cognee):
    if with_cognee:
        text_output = await cognee_and_llm(dataset)
        model_name = "with_cognee"
    else:
        text_output = await llm_on_preprocessed_data(dataset)
        model_name = "without_cognee"
    
    preds = {dataset[0]["instance_id"]:
                {"instance_id": dataset[0]["instance_id"],
                "model_patch": text_output,
                "model_name_or_path": model_name}}
    
    dataset_name = 'princeton-nlp/SWE-bench' if with_cognee else 'princeton-nlp/SWE-bench_bm25_13K'
    preds_dataset = get_dataset_from_preds(dataset_name, 
                                            "test", 
                                            [dataset[0]["instance_id"]], 
                                            preds, 
                                            model_name)
    
    return preds, preds_dataset

async def evaluate(test_specs: list[TestSpec],
                    preds: dict,
                    ):
    for test_spec in test_specs:
        pred = preds[test_spec.instance_id]
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)

        patch_file = Path(log_dir / "patch.diff")
        patch_file.write_text(pred["model_patch"] or "")
        for command in test_spec.repo_script_list:
            if "/testbed" in command:
                command = command.replace("/testbed", "./testbed")
            result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
            print(result)
        
        subprocess.run("git apply --allow-empty -v logs/patch.diff", shell=True, capture_output=True, text=True)

        

async def main():
    swe_dataset = load_swebench_dataset('princeton-nlp/SWE-bench', split='test')
    swe_dataset_preprocessed = load_swebench_dataset('princeton-nlp/SWE-bench_bm25_13K', split='test')
    test_data = swe_dataset[:1] 
    test_data_preprocessed = swe_dataset_preprocessed[:1] 
    assert test_data[0]["instance_id"] == test_data_preprocessed[0]["instance_id"]
    filepath = Path("SWE-bench_testsample")
    if filepath.exists():
        from datasets import Dataset
        dataset = Dataset.load_from_disk(filepath)
    else:
        dataset = download_instances(test_data, filepath)
    
    cognee_preds, cognee_preds_dataset = await get_preds(dataset, with_cognee=True)
    # nocognee_preds = await get_preds(dataset, with_cognee=False)
    test_specs = list(map(make_test_spec, test_data))
    results = await evaluate(test_specs, cognee_preds)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main(), debug=True)
