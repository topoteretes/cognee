from cognee.eval_framework.modal_run_eval import read_and_combine_metrics, image
from cognee.eval_framework.eval_config import EvalConfig
import modal
import logging
from cognee.eval_framework.corpus_builder.run_corpus_builder import run_corpus_builder
from cognee.eval_framework.answer_generation.run_question_answering_module import (
    run_question_answering,
)
from cognee.eval_framework.evaluation.run_evaluation_module import run_evaluation
import json

logger = logging.getLogger(__name__)

app = modal.App("cognee-regular-eval")


@app.function(image=image, concurrency_limit=2, timeout=1800, retries=1)
async def modal_run_eval(eval_params=None):
    """Runs evaluation pipeline and returns combined metrics results."""
    if eval_params is None:
        eval_params = EvalConfig().to_dict()

    logger.info(f"Running evaluation with params: {eval_params}")

    # Run the evaluation pipeline
    await run_corpus_builder(eval_params)
    await run_question_answering(eval_params)
    await run_evaluation(eval_params)

    # Early return if metrics calculation wasn't requested
    if not eval_params.get("evaluating_answers") or not eval_params.get("calculate_metrics"):
        logger.info(
            "Skipping metrics collection as either evaluating_answers or calculate_metrics is False"
        )
        return None

    return read_and_combine_metrics(eval_params)


@app.local_entrypoint()
async def main():
    config = EvalConfig(
        task_getter_type="Default",
        benchmark="HotPotQA",
        number_of_samples_in_corpus=50,
        building_corpus_from_scratch=True,
        answering_questions=True,
        qa_engine="cognee_graph_completion",
        evaluating_answers=True,
        calculate_metrics=True,
        dashboard=False,
    )

    results = await modal_run_eval.remote.aio(config.to_dict())
    output_file = "results.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Completed parallel evaluation runs. Results saved to {output_file}")
