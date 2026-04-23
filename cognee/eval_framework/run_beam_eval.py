import asyncio
from pathlib import Path

from cognee.eval_framework.answer_generation.run_question_answering_module import (
    run_question_answering,
)
from cognee.eval_framework.beam.presets import BEAM_DEFAULT_SPLIT
from cognee.eval_framework.beam.runtime import build_beam_conversation_corpus
from cognee.eval_framework.evaluation.run_evaluation_module import run_evaluation
from cognee.eval_framework.reporting.aggregations import (
    average_aggregate_metrics,
    average_metrics_by_question_type,
)
from cognee.eval_framework.reporting.io import read_json, write_json
from cognee.shared.logging_utils import get_logger

logger = get_logger()

NUM_CONVERSATIONS = 20
BEAM_MAX_BATCHES = None  # None = use all sessions
OUTPUT_DIR = Path(".")
COMBINED_RESULTS_PATH = "beam_combined_results.json"


async def run_single_conversation(conversation_index: int) -> dict:
    logger.info(
        f"=== BEAM Evaluation: conversation {conversation_index} / {NUM_CONVERSATIONS - 1} ==="
    )

    params = await build_beam_conversation_corpus(
        conversation_index=conversation_index,
        output_dir=OUTPUT_DIR,
        split=BEAM_DEFAULT_SPLIT,
        max_batches=BEAM_MAX_BATCHES,
        answering_questions=True,
    )

    logger.info(f"[conv {conversation_index}] Answering questions...")
    await run_question_answering(params)

    logger.info(f"[conv {conversation_index}] Evaluating answers...")
    await run_evaluation(params)

    return {
        "aggregate": read_json(params["aggregate_metrics_path"]),
        "per_question": read_json(params["metrics_path"]),
    }


async def main():
    all_aggregate = []
    all_per_question = []

    for conv_idx in range(NUM_CONVERSATIONS):
        conversation_result = await run_single_conversation(conv_idx)
        all_aggregate.append(conversation_result["aggregate"])
        all_per_question.append(conversation_result["per_question"])

    avg_aggregate = average_aggregate_metrics(all_aggregate)
    avg_per_type = average_metrics_by_question_type(all_per_question)

    combined = {
        "split": BEAM_DEFAULT_SPLIT,
        "num_conversations": NUM_CONVERSATIONS,
        "avg_aggregate": avg_aggregate,
        "avg_per_type": avg_per_type,
        "per_conversation": all_aggregate,
    }
    write_json(COMBINED_RESULTS_PATH, combined)

    logger.info(f"\n=== BEAM Results (averaged over {NUM_CONVERSATIONS} conversations) ===")
    for metric, score in avg_aggregate.items():
        if score is not None:
            logger.info(f"  {metric}: {score:.3f}")

    logger.info("\nPer question type (beam_rubric):")
    for qtype, metrics in avg_per_type.items():
        beam_rubric = metrics.get("beam_rubric")
        kendall = metrics.get("kendall_tau")
        parts = []
        if beam_rubric is not None:
            parts.append(f"beam_rubric={beam_rubric:.3f}")
        if kendall is not None:
            parts.append(f"kendall_tau={kendall:.3f}")
        if parts:
            logger.info(f"  {qtype}: {', '.join(parts)}")

    logger.info("=== Done ===")


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        print("Done")
