"""Run BEAM benchmark evaluation on a single conversation.

Usage:
    uv run python cognee/eval_framework/run_beam_eval.py
"""

import asyncio

from cognee.shared.logging_utils import get_logger
from cognee.eval_framework.eval_config import EvalConfig
from cognee.eval_framework.corpus_builder.run_corpus_builder import run_corpus_builder
from cognee.eval_framework.answer_generation.run_question_answering_module import (
    run_question_answering,
)
from cognee.eval_framework.evaluation.run_evaluation_module import run_evaluation
from cognee.eval_framework.metrics_dashboard import create_dashboard

logger = get_logger()


eval_params = EvalConfig(
    benchmark="BEAM",
    building_corpus_from_scratch=True,
    number_of_samples_in_corpus=1,
    qa_engine="beam_router",
    answering_questions=True,
    evaluating_answers=True,
    evaluating_contexts=False,
    evaluation_engine="DeepEval",
    evaluation_metrics=["rubric", "f1"],
    task_getter_type="Default",
    calculate_metrics=True,
    dashboard=True,
    questions_path="beam_questions.json",
    answers_path="beam_answers.json",
    metrics_path="beam_metrics.json",
    aggregate_metrics_path="beam_aggregate_metrics.json",
    dashboard_path="beam_dashboard.html",
).to_dict()

# Use max_batches=1 to truncate the conversation to ~1 session for faster local runs.
# Remove or increase this for full evaluation.
BEAM_MAX_BATCHES = 1


async def main():
    logger.info("=== BEAM Evaluation: 1 conversation, 100K split ===")

    # Step 1: Build corpus (ingest conversation into cognee)
    # Override the adapter to use max_batches for faster local runs
    logger.info("Step 1: Building corpus...")
    eval_params["_beam_max_batches"] = BEAM_MAX_BATCHES
    await run_corpus_builder(eval_params)

    # Step 2: Answer probing questions (routed by question type)
    logger.info("Step 2: Answering questions with BEAM router...")
    await run_question_answering(eval_params)

    # Step 3: Evaluate with rubric metric
    logger.info("Step 3: Evaluating answers...")
    await run_evaluation(eval_params)

    # Step 4: Dashboard
    if eval_params.get("dashboard"):
        logger.info("Step 4: Generating dashboard...")
        create_dashboard(
            metrics_path=eval_params["metrics_path"],
            aggregate_metrics_path=eval_params["aggregate_metrics_path"],
            output_file=eval_params["dashboard_path"],
            benchmark=eval_params["benchmark"],
        )

    logger.info("=== BEAM Evaluation complete ===")


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        print("Done")
