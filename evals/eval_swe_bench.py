from swebench.harness.utils import load_swebench_dataset
from swebench.inference.make_datasets.create_instance import PATCH_EXAMPLE
from evals.eval_utils import download_instances
import cognee
from cognee.api.v1.cognify.code_graph_pipeline import code_graph_pipeline
from cognee.api.v1.search import SearchType
import os
from pathlib import Path
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.llm.get_llm_client import get_llm_client
from cognee.shared.data_models import Answer

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

def llm_on_preprocessed_data(dataset):
    problem_statement = dataset[0]['problem_statement']
    prompt = dataset[0]["text"]
    
    llm_client = get_llm_client()
    answer_prediction = llm_client.create_structured_output(
                        text_input = problem_statement,
                        system_prompt = prompt, # TODO check if this is correct
                        response_model = str,
                        )
    return answer_prediction


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
    
    llm_output_with_cognee = await cognee_and_llm(dataset)
    llm_output_without_cognee = llm_on_preprocessed_data(test_data_preprocessed)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main(), debug=True)
