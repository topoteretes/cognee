import logging
import asyncio
import json

from evals.eval_framework.corpus_builder.corpus_builder_executor import CorpusBuilderExecutor
from evals.eval_framework.answer_generation.answer_generation_executor import (
    AnswerGeneratorExecutor,
)
from evals.eval_framework.evaluation.evaluation_executor import EvaluationExecutor
from evals.eval_framework.metrics_dashboard import generate_metrics_dashboard
from cognee.shared.utils import setup_logging


setup_logging(logging.INFO)

eval_params = {
    # Corpus builder params
    "building_corpus_from_scratch": True,
    "number_of_samples_in_corpus": 1,
    "benchmark": "Dummy",  # 'HotPotQA' or 'Dummy' or 'TwoWikiMultiHop'
    # Question answering params
    "answering_questions": True,
    "qa_engine": "cognee_completion",  # 'cognee_completion (simple RAG)' or 'cognee_graph_completion'
    # Evaluation params
    "evaluating_answers": True,
    "evaluation_engine": "DeepEval",
    "evaluation_metrics": ["correctness", "EM", "f1"],
    # Visualization
    "dashboard": True,
}

questions_file = "questions_output.json"
answers_file = "answers_output.json"
metrics_file = "metrics_output.json"


async def main():
    ################################ Step 1: Corpus builder module
    if eval_params["building_corpus_from_scratch"]:
        logging.info("Corpus Builder started...")

        corpus_builder = CorpusBuilderExecutor(benchmark=eval_params["benchmark"])
        questions = await corpus_builder.build_corpus(
            limit=eval_params["number_of_samples_in_corpus"]
        )

        with open(questions_file, "w", encoding="utf-8") as f:
            json.dump(questions, f, ensure_ascii=False, indent=4)

        logging.info("Corpus Builder End...")

    ################################ Step 2: Question answering module
    if eval_params["answering_questions"]:
        logging.info("Question answering started...")

        try:
            with open(questions_file, "r", encoding="utf-8") as f:
                questions = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find the file: {questions_file}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON from {questions_file}: {e}")

        print(f"Loaded {len(questions)} questions from {questions_file}")

        answer_generator = AnswerGeneratorExecutor()
        answers = await answer_generator.question_answering_non_parallel(
            questions=questions, qa_engine=eval_params["qa_engine"]
        )

        with open(answers_file, "w", encoding="utf-8") as f:
            json.dump(answers, f, ensure_ascii=False, indent=4)

        logging.info("Question answering end...")

    ################################ Step 3: Evaluation module
    if eval_params["evaluating_answers"]:
        logging.info("Evaluation started...")
        try:
            with open(answers_file, "r", encoding="utf-8") as f:
                answers = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find the file: {answers_file}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON from {answers_file}: {e}")

        print(f"Loaded {len(answers)} questions from {answers_file}")

        evaluator = EvaluationExecutor(evaluator_engine=eval_params["evaluation_engine"])
        metrics = await evaluator.execute(
            answers=answers,
            evaluator_metrics=eval_params["evaluation_metrics"],
        )

        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=4)

        logging.info("Question answering end...")

    if eval_params["dashboard"]:
        generate_metrics_dashboard(
            json_data=metrics_file, output_file="dashboard.html", benchmark=eval_params["benchmark"]
        )


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        print("Done")
