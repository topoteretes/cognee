"""Run BEAM 1M benchmark evaluation.

Usage:
    caffeinate -s uv run python run_beam_1m.py
"""

import asyncio
import json
import os

from cognee.shared.logging_utils import get_logger
from cognee.eval_framework.eval_config import EvalConfig
from cognee.eval_framework.answer_generation.beam_router import BEAMRouter
from cognee.modules.chunking.ConversationChunker import ConversationChunker
from cognee.eval_framework.corpus_builder.run_corpus_builder import run_corpus_builder
from cognee.eval_framework.evaluation.run_evaluation_module import run_evaluation

logger = get_logger()

NUM_CONVERSATIONS = 35  # 1M split has 35 conversations

# 1M-scale retrieval settings (larger graph needs wider retrieval)
_1M_TOP_K = {
    "summarization": 80,
    "DEFAULT": 30,
}
_1M_CONTEXT_EXTENSION_ROUNDS = 6
_1M_WIDE_SEARCH_TOP_K = 150
_1M_TRIPLET_DISTANCE_PENALTY = 5.0


def _make_eval_params(conversation_index: int) -> dict:
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
        questions_path=f"beam1m_questions_conv{conversation_index}.json",
        answers_path=f"beam1m_answers_conv{conversation_index}.json",
        metrics_path=f"beam1m_metrics_conv{conversation_index}.json",
        aggregate_metrics_path=f"beam1m_aggregate_metrics_conv{conversation_index}.json",
        dashboard_path=f"beam1m_dashboard_conv{conversation_index}.html",
    ).to_dict()


async def run_single_conversation(conversation_index: int) -> dict:
    logger.info(
        f"=== BEAM 1M: conversation {conversation_index} / {NUM_CONVERSATIONS - 1} ==="
    )

    params = _make_eval_params(conversation_index)
    params["_beam_split"] = "1M"
    params["_beam_max_batches"] = None
    params["_beam_conversation_index"] = conversation_index
    params["chunker"] = ConversationChunker

    await run_corpus_builder(params)

    # Answer questions with 1M-scale retrieval settings
    with open(params["questions_path"], "r") as f:
        questions = json.load(f)

    router = BEAMRouter(
        top_k_overrides=_1M_TOP_K,
        context_extension_rounds=_1M_CONTEXT_EXTENSION_ROUNDS,
        wide_search_top_k=_1M_WIDE_SEARCH_TOP_K,
        triplet_distance_penalty=_1M_TRIPLET_DISTANCE_PENALTY,
    )
    answers = await router.answer_questions(questions)

    with open(params["answers_path"], "w") as f:
        json.dump(answers, f, ensure_ascii=False, indent=4)

    params["answering_questions"] = False
    await run_evaluation(params)

    with open(params["aggregate_metrics_path"], "r") as f:
        return json.load(f)


async def main():
    from collections import defaultdict

    all_aggregate = []
    type_scores = defaultdict(lambda: defaultdict(list))

    # Conversations that OOM during cognify — skip them
    SKIP_CONVS = {23, 24, 33}  # 23: 4.7M, 24: 7.4M, 33: 7.3M — OOM during cognify

    start_from = 0
    # Resume: skip conversations that already have results
    for i in range(NUM_CONVERSATIONS):
        if os.path.exists(f"beam1m_aggregate_metrics_conv{i}.json"):
            start_from = i + 1
        elif i in SKIP_CONVS:
            start_from = i + 1
        else:
            break

    if start_from > 0:
        logger.info(f"Resuming from conversation {start_from} (skipping 0-{start_from - 1})")
        for i in range(start_from):
            if i in SKIP_CONVS:
                continue
            with open(f"beam1m_aggregate_metrics_conv{i}.json", "r") as f:
                all_aggregate.append(json.load(f))
            metrics_path = f"beam1m_metrics_conv{i}.json"
            with open(metrics_path, "r") as f:
                per_q = json.load(f)
            for entry in per_q:
                qtype = entry.get("question_type", "unknown")
                for metric_name, metric_data in entry.get("metrics", {}).items():
                    score = metric_data.get("score")
                    if score is not None:
                        type_scores[qtype][metric_name].append(score)

    for conv_idx in range(start_from, NUM_CONVERSATIONS):
        if conv_idx in SKIP_CONVS:
            logger.info(f"Skipping conversation {conv_idx} (in skip list)")
            continue
        agg = await run_single_conversation(conv_idx)
        all_aggregate.append(agg)

        metrics_path = f"beam1m_metrics_conv{conv_idx}.json"
        with open(metrics_path, "r") as f:
            per_q = json.load(f)
        for entry in per_q:
            qtype = entry.get("question_type", "unknown")
            for metric_name, metric_data in entry.get("metrics", {}).items():
                score = metric_data.get("score")
                if score is not None:
                    type_scores[qtype][metric_name].append(score)

    # Per-type averages
    avg_per_type = {}
    for qtype, metrics in sorted(type_scores.items()):
        avg_per_type[qtype] = {
            m: sum(s) / len(s) if s else None for m, s in metrics.items()
        }

    combined = {
        "dataset": "BEAM-1M",
        "num_conversations": NUM_CONVERSATIONS,
        "avg_per_type": avg_per_type,
        "per_conversation": all_aggregate,
    }
    with open("beam1m_combined_results.json", "w") as f:
        json.dump(combined, f, indent=2)

    print(f"\n=== BEAM 1M Results ({NUM_CONVERSATIONS} conversations) ===")
    for qtype, metrics in avg_per_type.items():
        beam_rubric = metrics.get("beam_rubric")
        kendall = metrics.get("kendall_tau")
        parts = []
        if beam_rubric is not None:
            parts.append(f"beam_rubric={beam_rubric:.3f}")
        if kendall is not None:
            parts.append(f"kendall_tau={kendall:.3f}")
        if parts:
            print(f"  {qtype}: {', '.join(parts)}")

    all_rubric = [
        m.get("beam_rubric") for m in avg_per_type.values() if m.get("beam_rubric") is not None
    ]
    if all_rubric:
        print(f"\n  OVERALL beam_rubric avg: {sum(all_rubric) / len(all_rubric):.3f}")


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        print("\nDone")
