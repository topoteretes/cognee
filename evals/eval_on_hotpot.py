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

logger = logging.getLogger(__name__)


async def answer_qa_instance(instance, context_provider):
    context = await context_provider(instance)

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


async def deepeval_on_instances(instances, context_provider, eval_metrics):
    answers = []
    for instance in tqdm(instances, desc="Getting answers"):
        answer = await answer_qa_instance(instance, context_provider)
        answers.append(answer)

    eval_results = await deepeval_answers(instances, answers, eval_metrics)
    score_lists_dict = {}
    for instance_result in eval_results.test_results:
        for metric_result in instance_result.metrics_data:
            if metric_result.name not in score_lists_dict:
                score_lists_dict[metric_result.name] = []
            score_lists_dict[metric_result.name].append(metric_result.score)

    avg_scores = {
        metric_name: statistics.mean(scorelist)
        for metric_name, scorelist in score_lists_dict.items()
    }

    return avg_scores


async def eval_on_QA_dataset(
    dataset_name_or_filename: str, context_provider_name, num_samples, metric_name_list
):
    dataset = load_qa_dataset(dataset_name_or_filename)
    context_provider = qa_context_providers[context_provider_name]
    eval_metrics = get_metrics(metric_name_list)
    instances = dataset if not num_samples else dataset[:num_samples]

    if "promptfoo_metrics" in eval_metrics:
        promptfoo_results = await eval_metrics["promptfoo_metrics"].measure(
            instances, context_provider
        )
    else:
        promptfoo_results = {}
    deepeval_results = await deepeval_on_instances(
        instances, context_provider, eval_metrics["deepeval_metrics"]
    )

    results = promptfoo_results | deepeval_results

    return results


async def incremental_eval_on_QA_dataset(
    dataset_name_or_filename: str, num_samples, metric_name_list
):
    pipeline_slice_names = valid_pipeline_slices.keys()

    incremental_results = {}
    for pipeline_slice_name in pipeline_slice_names:
        results = await eval_on_QA_dataset(
            dataset_name_or_filename, pipeline_slice_name, num_samples, metric_name_list
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

    args = parser.parse_args()

    if args.rag_option == "cognee_incremental":
        avg_scores = await incremental_eval_on_QA_dataset(
            args.dataset, args.num_samples, args.metrics
        )

    else:
        avg_scores = await eval_on_QA_dataset(
            args.dataset, args.rag_option, args.num_samples, args.metrics
        )

    logger.info(f"{avg_scores}")


if __name__ == "__main__":
    asyncio.run(main())
