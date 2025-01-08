import argparse
import asyncio
import json
import statistics
from pathlib import Path

import deepeval.metrics
import wget
from deepeval.dataset import EvaluationDataset
from deepeval.test_case import LLMTestCase
from jsonschema import ValidationError, validate
from tqdm import tqdm

import cognee
import evals.deepeval_metrics
from cognee.api.v1.search import SearchType
from cognee.infrastructure.llm.get_llm_client import get_llm_client
from cognee.infrastructure.llm.prompts import read_query_prompt, render_prompt
from cognee.root_dir import get_absolute_path

qa_datasets = {
    "hotpotqa": {
        "filename": "hotpot_dev_fullwiki_v1.json",
        "URL": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json"
    },
    "2wikimultihop": {
        "filename": "data/dev.json",
        "URL": "https://www.dropbox.com/scl/fi/heid2pkiswhfaqr5g0piw/data.zip?rlkey=ira57daau8lxfj022xvk1irju&e=1"
    }
}

qa_json_schema = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "question": {"type": "string"},
            "context": {"type": "array"},
        },
        "required": ["answer", "question", "context"], 
        "additionalProperties": True 
    } 
}


def download_qa_dataset(dataset_name: str, dir: str):
    
    if dataset_name not in qa_datasets:
        raise ValueError(f"{dataset_name} is not a supported dataset.")

    url = qa_datasets[dataset_name]["URL"]

    if dataset_name == "2wikimultihop":
        raise Exception("Please download 2wikimultihop dataset (data.zip) manually from \
                        https://www.dropbox.com/scl/fi/heid2pkiswhfaqr5g0piw/data.zip?rlkey=ira57daau8lxfj022xvk1irju&e=1 \
                        and unzip it.")

    wget.download(url, out=dir) 


def load_qa_dataset(filepath: Path):

    with open(filepath, "r") as file:
        dataset = json.load(file)

    try:
        validate(instance=dataset, schema=qa_json_schema)
    except ValidationError as e:
        print("File is not a valid QA dataset:", e.message)   

    return dataset

async def answer_without_cognee(instance):
    args = {
        "question": instance["question"],
        "context": instance["context"],
    }
    user_prompt = render_prompt("context_for_question.txt", args)
    system_prompt = read_query_prompt("answer_hotpot_question.txt")

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
        await cognee.add("\n".join(sentences), dataset_name = "QA")
    await cognee.cognify("QA")

    search_results = await cognee.search(
        SearchType.INSIGHTS, query_text=instance["question"]
    )
   
    args = {
        "question": instance["question"],
        "context": search_results,
    }
    user_prompt = render_prompt("context_for_question.txt", args)
    system_prompt = read_query_prompt("answer_hotpot_using_cognee_search.txt")

    llm_client = get_llm_client()
    answer_prediction = await llm_client.acreate_structured_output(
        text_input=user_prompt,
        system_prompt=system_prompt,
        response_model=str,
    )
    
    return answer_prediction


async def eval_answers(instances, answers, eval_metric):
    test_cases = []
    
    for instance, answer in zip(instances, answers):
        test_case = LLMTestCase(
            input=instance["question"],
            actual_output=answer,
            expected_output=instance["answer"]
        )
        test_cases.append(test_case)
    
    eval_set = EvaluationDataset(test_cases)
    eval_results = eval_set.evaluate([eval_metric])
    
    return eval_results

async def eval_on_QA_dataset(dataset_name: str, answer_provider, num_samples, eval_metric):
    
    data_root_dir = get_absolute_path("../.data")
    
    if not Path(data_root_dir).exists():
        Path(data_root_dir).mkdir()
    
    filename = qa_datasets[dataset_name]["filename"]
    filepath = data_root_dir / Path(filename)
    if not filepath.exists():
        download_qa_dataset(dataset_name, data_root_dir)
    
    dataset = load_qa_dataset(filepath)
    
    instances = dataset if not num_samples else dataset[:num_samples]
    answers = []
    for instance in tqdm(instances, desc="Getting answers"):
        answer = await answer_provider(instance)
        answers.append(answer)
    
    eval_results = await eval_answers(instances, answers, eval_metric)
    avg_score = statistics.mean([result.metrics_data[0].score for result in eval_results.test_results])
    
    return avg_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset", type=str, choices=list(qa_datasets.keys()), help="Which dataset to evaluate on")
    parser.add_argument("--with_cognee", action="store_true")
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--metric", type=str, default="correctness_metric",
                        help="Valid options are Deepeval metrics (e.g. AnswerRelevancyMetric) \
                              and metrics defined in evals/deepeval_metrics.py, e.g. f1_score_metric")
    
    args = parser.parse_args()

    try:
        metric_cls = getattr(deepeval.metrics, args.metric)
        metric = metric_cls()
    except AttributeError:
        metric = getattr(evals.deepeval_metrics, args.metric)
        if isinstance(metric, type):
            metric = metric()
    
    if args.with_cognee:
        answer_provider = answer_with_cognee
    else:
        answer_provider = answer_without_cognee
    
    avg_score = asyncio.run(eval_on_QA_dataset(args.dataset, answer_provider, args.num_samples, metric))
    print(f"Average {args.metric}: {avg_score}")