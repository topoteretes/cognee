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
from evals.qa_metrics_utils import get_metric
from evals.qa_context_provider_utils import qa_context_providers

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


async def deepeval_answers(instances, answers, eval_metric):
    test_cases = []

    for instance, answer in zip(instances, answers):
        test_case = LLMTestCase(
            input=instance["question"], actual_output=answer, expected_output=instance["answer"]
        )
        test_cases.append(test_case)

    eval_set = EvaluationDataset(test_cases)
    eval_results = eval_set.evaluate([eval_metric])

    return eval_results


async def deepeval_on_instances(instances, context_provider, eval_metric):
    answers = []
    for instance in tqdm(instances, desc="Getting answers"):
        answer = await answer_qa_instance(instance, context_provider)
        answers.append(answer)

    eval_results = await deepeval_answers(instances, answers, eval_metric)
    avg_score = statistics.mean(
        [result.metrics_data[0].score for result in eval_results.test_results]
    )

    return avg_score


async def eval_on_QA_dataset(
    dataset_name_or_filename: str, context_provider_name, num_samples, eval_metric_name
):
    dataset = load_qa_dataset(dataset_name_or_filename)
    context_provider = qa_context_providers[context_provider_name]
    eval_metric = get_metric(eval_metric_name)
    instances = dataset if not num_samples else dataset[:num_samples]

    if eval_metric_name.startswith("promptfoo"):
        return await eval_metric.measure(instances, context_provider)
    else:
        return await deepeval_on_instances(instances, context_provider, eval_metric)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True, help="Which dataset to evaluate on")
    parser.add_argument(
        "--rag_option",
        type=str,
        choices=qa_context_providers.keys(),
        required=True,
        help="RAG option to use for providing context",
    )
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--metric_name", type=str, default="Correctness")

    args = parser.parse_args()

    avg_score = asyncio.run(
        eval_on_QA_dataset(args.dataset, args.rag_option, args.num_samples, args.metric_name)
    )
    logger.info(f"Average {args.metric_name}: {avg_score}")
