import argparse
import asyncio
import statistics
import deepeval.metrics
from deepeval.dataset import EvaluationDataset
from deepeval.test_case import LLMTestCase
from tqdm import tqdm

import cognee
import evals.deepeval_metrics
from cognee.api.v1.search import SearchType
from cognee.infrastructure.llm.get_llm_client import get_llm_client
from cognee.infrastructure.llm.prompts import read_query_prompt, render_prompt
from evals.qa_dataset_utils import load_qa_dataset
from evals.qa_metrics_utils import get_metric


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

    for title, sentences in instance["context"]:
        await cognee.add("\n".join(sentences), dataset_name="QA")
    await cognee.cognify("QA")

    search_results = await cognee.search(SearchType.INSIGHTS, query_text=instance["question"])
    search_results_second = await cognee.search(
        SearchType.SUMMARIES, query_text=instance["question"]
    )
    search_results = search_results + search_results_second

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
            input=instance["question"], actual_output=answer, expected_output=instance["answer"]
        )
        test_cases.append(test_case)

    eval_set = EvaluationDataset(test_cases)
    eval_results = eval_set.evaluate([eval_metric])

    return eval_results


async def eval_on_QA_dataset(
    dataset_name_or_filename: str, answer_provider, num_samples, eval_metric_name
):
    dataset = load_qa_dataset(dataset_name_or_filename)
    eval_metric = get_metric(eval_metric_name)

    instances = dataset if not num_samples else dataset[:num_samples]
    answers = []
    for instance in tqdm(instances, desc="Getting answers"):
        answer = await answer_provider(instance)
        answers.append(answer)

    eval_results = await eval_answers(instances, answers, eval_metric)
    avg_score = statistics.mean(
        [result.metrics_data[0].score for result in eval_results.test_results]
    )

    return avg_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, help="Which dataset to evaluate on")
    parser.add_argument("--with_cognee", action="store_true")
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--metric_name", type=str, default="Correctness")

    args = parser.parse_args()

    if args.with_cognee:
        answer_provider = answer_with_cognee
    else:
        answer_provider = answer_without_cognee

    avg_score = asyncio.run(
        eval_on_QA_dataset(args.dataset, answer_provider, args.num_samples, args.metric_name)
    )
    print(f"Average {args.metric_name}: {avg_score}")
