from cognee.shared.logging_utils import get_logger
import asyncio
from cognee.eval_framework.eval_config import EvalConfig

from cognee.eval_framework.corpus_builder.run_corpus_builder import run_corpus_builder
from cognee.eval_framework.answer_generation.run_question_answering_module import (
    run_question_answering,
)
from cognee.eval_framework.evaluation.run_evaluation_module import run_evaluation
from cognee.eval_framework.metrics_dashboard import create_dashboard

# Configure logging(logging.INFO)
logger = get_logger()

# Define parameters and file paths.
eval_params = EvalConfig().to_dict()

questions_file = "questions_output.json"
answers_file = "answers_output.json"
metrics_file = "metrics_output.json"
dashboard_path = "dashboard.html"


async def main():
    # Corpus builder
    await run_corpus_builder(eval_params)

    # Question answering
    await run_question_answering(eval_params)

    # Metrics calculation + dashboard
    await run_evaluation(eval_params)

    if eval_params.get("dashboard"):
        logger.info("Generating dashboard...")
        create_dashboard(
            metrics_path=eval_params["metrics_path"],
            aggregate_metrics_path=eval_params["aggregate_metrics_path"],
            output_file=eval_params["dashboard_path"],
            benchmark=eval_params["benchmark"],
        )


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        print("Done")
