import argparse
import asyncio
import statistics
from deepeval.dataset import EvaluationDataset
from deepeval.test_case import LLMTestCase
from tqdm import tqdm
import logging
from cognee.infrastructure.llm.get_llm_client import get_llm_client
from cognee.infrastructure.llm.prompts import read_query_prompt, render_prompt
from evals.qa_dataset_utils import load_qa_dataset
from evals.qa_metrics_utils import get_metrics
from evals.qa_context_provider_utils import qa_context_providers, valid_pipeline_slices
import random
import os
import json
from pathlib import Path

logger = logging.getLogger(__name__)


async def get_context(instance, context_provider, contexts_filename=None):
    """Retrieves or generates context for a QA instance, optionally using a cache."""
    preloaded_contexts = {}

    if contexts_filename and os.path.exists(contexts_filename):
        with open(contexts_filename, "r") as file:
            preloaded_contexts = json.load(file)

    if instance["_id"] in preloaded_contexts:
        return preloaded_contexts[instance["_id"]]

    context = await context_provider(instance)
    preloaded_contexts[instance["_id"]] = context

    if contexts_filename:
        with open(contexts_filename, "w") as file:
            json.dump(preloaded_contexts, file)

    return context


async def answer_qa_instance(instance, context_provider, contexts_filename=None):
    """Answers a QA instance using a given context provider, optionally caching contexts."""
    context = await get_context(instance, context_provider, contexts_filename)

    args = {
        "question": instance["question"],
        "context": context,
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


async def deepeval_answers(instances, answers, eval_metrics):
    test_cases = []

    for instance, answer in zip(instances, answers):
        test_case = LLMTestCase(
            input=instance["question"], actual_output=answer, expected_output=instance["answer"]
        )
        test_cases.append(test_case)

    eval_set = EvaluationDataset(test_cases)
    eval_results = eval_set.evaluate(eval_metrics)

    return eval_results


async def get_answers_for_instances(
    instances, context_provider, answers_filename=None, contexts_filename=None, save_answers=False
):
    """Loads or generates answers for instances and optionally saves them."""
    preloaded_answers = {}

    if answers_filename and os.path.exists(answers_filename):
        with open(answers_filename, "r") as file:
            preloaded_answers = json.load(file)

    answers = []
    for instance in tqdm(instances, desc="Getting answers"):
        if instance["_id"] in preloaded_answers:
            answer = preloaded_answers[instance["_id"]]
        else:
            answer = await answer_qa_instance(instance, context_provider, contexts_filename)
            preloaded_answers[instance["_id"]] = answer
        answers.append(answer)

    if save_answers and answers_filename:
        with open(answers_filename, "w") as file:
            json.dump(preloaded_answers, file)

    return answers


async def calculate_eval_metrics(instances, answers, eval_metrics):
    """Evaluates answers and returns a dictionary of metric scores lists."""
    eval_results = await deepeval_answers(instances, answers, eval_metrics)

    score_lists_dict = {}
    for instance_result in eval_results.test_results:
        for metric_result in instance_result.metrics_data:
            score_lists_dict.setdefault(metric_result.name, []).append(metric_result.score)

    return score_lists_dict  # Returning raw lists of scores


async def deepeval_on_instances(
    instances,
    context_provider,
    eval_metrics,
    answers_filename=None,
    contexts_filename=None,
    save_answers=False,
):
    """Orchestrates answer generation and evaluation, including averaging of scores."""
    answers = await get_answers_for_instances(
        instances, context_provider, answers_filename, contexts_filename, save_answers
    )
    score_lists_dict = await calculate_eval_metrics(instances, answers, eval_metrics)

    avg_scores = {
        metric_name: statistics.mean(scores) for metric_name, scores in score_lists_dict.items()
    }

    return avg_scores


async def eval_on_QA_dataset(
    dataset_name_or_filename: str, context_provider_name, num_samples, metric_name_list, out_path
):
    dataset = load_qa_dataset(dataset_name_or_filename)
    context_provider = qa_context_providers[context_provider_name]
    eval_metrics = get_metrics(metric_name_list)

    out_path = Path(out_path)
    if not out_path.exists():
        out_path.mkdir(parents=True, exist_ok=True)

    random.seed(43)
    instances = dataset if not num_samples else random.sample(dataset, num_samples)

    # contexts_filename = out_path / Path(
    #     f"contexts_{dataset_name_or_filename.split('.')[0]}_{context_provider_name}.json"
    # )
    contexts_filename = None
    if "promptfoo_metrics" in eval_metrics:
        promptfoo_results = await eval_metrics["promptfoo_metrics"].measure(
            instances, context_provider, contexts_filename
        )
    else:
        promptfoo_results = {}

    # answers_filename = out_path / Path(
    #     f"answers_{dataset_name_or_filename.split('.')[0]}_{context_provider_name}.json"
    # )
    answers_filename = None
    deepeval_results = await deepeval_on_instances(
        instances,
        context_provider,
        eval_metrics["deepeval_metrics"],
        answers_filename,
        contexts_filename,
    )

    results = promptfoo_results | deepeval_results

    return results


async def incremental_eval_on_QA_dataset(
    dataset_name_or_filename: str, num_samples, metric_name_list, out_path
):
    pipeline_slice_names = valid_pipeline_slices.keys()

    incremental_results = {}
    for pipeline_slice_name in pipeline_slice_names:
        results = await eval_on_QA_dataset(
            dataset_name_or_filename, pipeline_slice_name, num_samples, metric_name_list, out_path
        )
        incremental_results[pipeline_slice_name] = results

    return incremental_results


async def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True, help="Which dataset to evaluate on")
    parser.add_argument(
        "--rag_option",
        type=str,
        choices=list(qa_context_providers.keys()) + ["cognee_incremental"],
        required=True,
        help="RAG option to use for providing context",
    )
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--metrics", type=str, nargs="+", default=["Correctness"])
    parser.add_argument("--out_dir", type=str, help="Dir to save eval results")

    args = parser.parse_args()

    if args.rag_option == "cognee_incremental":
        avg_scores = await incremental_eval_on_QA_dataset(
            args.dataset, args.num_samples, args.metrics, args.out_dir
        )

    else:
        avg_scores = await eval_on_QA_dataset(
            args.dataset, args.rag_option, args.num_samples, args.metrics, args.out_dir
        )

    logger.info(f"{avg_scores}")


if __name__ == "__main__":
    asyncio.run(main())
