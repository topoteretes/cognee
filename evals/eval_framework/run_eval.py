import logging
import asyncio
from cognee.shared.utils import setup_logging
from evals.eval_framework.eval_config import EvalConfig

from evals.eval_framework.corpus_builder.run_corpus_builder import run_corpus_builder
from evals.eval_framework.answer_generation.run_question_answering_module import (
    run_question_answering,
)
from evals.eval_framework.evaluation.run_evaluation_module import run_evaluation

# Configure logging
setup_logging(logging.INFO)

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


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        print("Done")
