import argparse
import asyncio
import json
import statistics
from pathlib import Path

import wget
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from tqdm import tqdm

import cognee
from cognee.api.v1.search import SearchType
from cognee.base_config import get_base_config
from cognee.infrastructure.llm.get_llm_client import get_llm_client
from cognee.infrastructure.llm.prompts import read_query_prompt, render_prompt


async def answer_without_cognee(instance):
    args = {
        "question": instance["question"],
        "context": instance["context"],
    }
    user_prompt = render_prompt("context_for_question.txt", args)
    system_prompt = read_query_prompt("answer_question.txt")

    llm_client = get_llm_client()
    answer_prediction = await llm_client.acreate_structured_output(
        text_input=user_prompt,
        system_prompt=system_prompt,
        response_model=str,
    )
    return answer_prediction

async def answer_with_cognee(instance):
    
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    for (title, sentences) in instance["context"]:
        await cognee.add("\n".join(sentences), dataset_name = "HotPotQA")
    await cognee.cognify("HotPotQA")

    search_results = await cognee.search(
        SearchType.INSIGHTS, query_text=instance["question"]
    )
   
    args = {
        "question": instance["question"],
        "context": search_results,
    }
    user_prompt = render_prompt("context_for_question.txt", args)
    system_prompt = read_query_prompt("answer_question_kg.txt")

    llm_client = get_llm_client()
    answer_prediction = await llm_client.acreate_structured_output(
        text_input=user_prompt,
        system_prompt=system_prompt,
        response_model=str,
    )
    return answer_prediction

correctness_metric = GEval(
        name="Correctness",
        model="gpt-4o-mini",
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT
        ],
        evaluation_steps=[
           "Determine whether the actual output is factually correct based on the expected output."    
        ]
    )


async def eval_correctness(with_cognee=True, num_samples=None):
    base_config = get_base_config()
    data_root_dir = base_config.data_root_directory
    filepath = data_root_dir / Path("hotpot_dev_fullwiki_v1.json")
    if not filepath.exists():
        url = 'http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json'
        wget.download(url, out=data_root_dir)
    with open(filepath, "r") as file:
        dataset = json.load(file)
    test_cases = []
    if not num_samples:
        num_samples = len(dataset)
    for instance in tqdm(dataset[:num_samples], desc="Evaluating correctness"):
        if with_cognee:
            answer = await answer_with_cognee(instance)
        else:
            answer = await answer_without_cognee(instance)
        test_case = LLMTestCase(
            input=instance["question"],
            actual_output=answer,
            expected_output=instance["answer"]
        )
        test_cases.append(test_case)
    evalset = EvaluationDataset(test_cases)
    evalresults = evalset.evaluate([correctness_metric])
    avg_correctness = statistics.mean([result.metrics_data[0].score for result in evalresults.test_results])
    return avg_correctness

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--with_cognee", action="store_true")
    parser.add_argument("--num_samples", type=int, default=500)
    args = parser.parse_args()
    avg_correctness = asyncio.run(eval_correctness(args.with_cognee, args.num_samples))
    print(f"Average correctness: {avg_correctness}")