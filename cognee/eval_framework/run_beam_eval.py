"""Run BEAM benchmark evaluation across multiple conversations.

Usage:
    uv run python cognee/eval_framework/run_beam_eval.py
"""

import asyncio
import json

from cognee.shared.logging_utils import get_logger
from cognee.eval_framework.eval_config import EvalConfig
from cognee.modules.chunking.ConversationChunker import ConversationChunker
from cognee.eval_framework.corpus_builder.run_corpus_builder import run_corpus_builder
from cognee.eval_framework.answer_generation.run_question_answering_module import (
    run_question_answering,
)
from cognee.eval_framework.evaluation.run_evaluation_module import run_evaluation
from cognee.eval_framework.metrics_dashboard import create_dashboard

logger = get_logger()

NUM_CONVERSATIONS = 20
BEAM_MAX_BATCHES = None  # None = use all sessions


def _make_eval_params(conversation_index: int) -> dict:
    """Create eval params for a single conversation run."""
    return EvalConfig(
        benchmark="BEAM",
        building_corpus_from_scratch=True,
        number_of_samples_in_corpus=20,
        qa_engine="beam_router",
        answering_questions=True,
        evaluating_answers=True,
        evaluating_contexts=False,
        evaluation_engine="DeepEval",
        evaluation_metrics=["beam_rubric", "kendall_tau"],
        task_getter_type="Default",
        calculate_metrics=True,
        dashboard=False,
        questions_path=f"beam_questions_conv{conversation_index}.json",
        answers_path=f"beam_answers_conv{conversation_index}.json",
        metrics_path=f"beam_metrics_conv{conversation_index}.json",
        aggregate_metrics_path=f"beam_aggregate_metrics_conv{conversation_index}.json",
        dashboard_path=f"beam_dashboard_conv{conversation_index}.html",
    ).to_dict()


async def run_single_conversation(conversation_index: int) -> dict:
    """Run the full eval pipeline for one conversation and return aggregate metrics."""
    logger.info(
        f"=== BEAM Evaluation: conversation {conversation_index} / {NUM_CONVERSATIONS - 1} ==="
    )

    params = _make_eval_params(conversation_index)
    params["_beam_max_batches"] = BEAM_MAX_BATCHES
    params["_beam_conversation_index"] = conversation_index
    params["chunker"] = ConversationChunker

    # Step 1: Build corpus
    logger.info(f"[conv {conversation_index}] Step 1: Building corpus...")
    await run_corpus_builder(params)

    # Step 2: Answer questions
    logger.info(f"[conv {conversation_index}] Step 2: Answering questions...")
    await run_question_answering(params)

    # Step 3: Evaluate
    logger.info(f"[conv {conversation_index}] Step 3: Evaluating answers...")
    await run_evaluation(params)

    # Load aggregate metrics for this conversation
    with open(params["aggregate_metrics_path"], "r") as f:
        return json.load(f)


def _average_aggregate_metrics(all_metrics: list) -> dict:
    """Average aggregate metrics across conversations."""
    if not all_metrics:
        return {}

    # Collect all metric keys from the first result
    avg = {}
    for key in all_metrics[0]:
        values = [m[key] for m in all_metrics if key in m and m[key] is not None]
        if values and all(isinstance(v, (int, float)) for v in values):
            avg[key] = sum(values) / len(values)
        else:
            avg[key] = None

    return avg


def _average_per_type_metrics(all_per_conv_metrics: list) -> dict:
    """Average per-question-type metrics across conversations."""
    from collections import defaultdict

    type_scores = defaultdict(lambda: defaultdict(list))

    for conv_metrics in all_per_conv_metrics:
        for entry in conv_metrics:
            qtype = entry.get("question_type", "unknown")
            metrics = entry.get("metrics", {})
            for metric_name, metric_data in metrics.items():
                score = metric_data.get("score")
                if score is not None:
                    type_scores[qtype][metric_name].append(score)

    result = {}
    for qtype, metrics in sorted(type_scores.items()):
        result[qtype] = {}
        for metric_name, scores in metrics.items():
            result[qtype][metric_name] = sum(scores) / len(scores) if scores else None

    return result


async def main():
    all_aggregate = []
    all_per_question = []

    for conv_idx in range(NUM_CONVERSATIONS):
        agg = await run_single_conversation(conv_idx)
        all_aggregate.append(agg)

        # Load per-question metrics
        metrics_path = f"beam_metrics_conv{conv_idx}.json"
        with open(metrics_path, "r") as f:
            all_per_question.append(json.load(f))

    # Average across conversations
    avg_aggregate = _average_aggregate_metrics(all_aggregate)
    avg_per_type = _average_per_type_metrics(all_per_question)

    # Save combined results
    combined = {
        "num_conversations": NUM_CONVERSATIONS,
        "avg_aggregate": avg_aggregate,
        "avg_per_type": avg_per_type,
        "per_conversation": all_aggregate,
    }
    with open("beam_combined_results.json", "w") as f:
        json.dump(combined, f, indent=2)

    # Print summary
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
