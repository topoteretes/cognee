import logging
import asyncio
import json
from cognee.shared.utils import setup_logging

from evals.eval_framework.corpus_builder.run_corpus_builder import run_corpus_builder
from evals.eval_framework.answer_generation.run_question_answering_module import (
    run_question_answering,
)
from evals.eval_framework.evaluation.run_evaluation_module import run_evaluation

# Configure logging
setup_logging(logging.INFO)

# Define parameters and file paths.
eval_params = {
    # Corpus builder params
    "building_corpus_from_scratch": True,
    "number_of_samples_in_corpus": 1,
    "benchmark": "Dummy",  # Options: 'HotPotQA', 'Dummy', 'TwoWikiMultiHop'
    # Question answering params
    "answering_questions": True,
    "qa_engine": "cognee_completion",  # Options: 'cognee_completion' or 'cognee_graph_completion'
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
    # Corpus builder
    await run_corpus_builder(eval_params, questions_file)

    # Question answering
    await run_question_answering(eval_params, questions_file, answers_file)

    # Metrics calculation + dashboard
    await run_evaluation(eval_params, answers_file, metrics_file)


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        print("Done")
