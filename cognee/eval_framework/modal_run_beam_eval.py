"""Run BEAM benchmark evaluation on Modal.

Usage:
    # Run with defaults (100K split, 20 conversations)
    modal run cognee/eval_framework/modal_run_beam_eval.py

    # Run 1M split with 5 conversations
    modal run cognee/eval_framework/modal_run_beam_eval.py --split 1M --num-conversations 5

    # Run specific question types only
    modal run cognee/eval_framework/modal_run_beam_eval.py --question-types "summarization,knowledge_update"
"""

import asyncio
import datetime
import json
import os
import pathlib
from os import path
from typing import List, Optional

import modal
from modal import Image

vol = modal.Volume.from_name("beam_eval_results", create_if_missing=True)

app = modal.App("beam-benchmark-eval")

# 1M-scale retrieval settings (larger graph needs wider retrieval)
_1M_TOP_K = {
    "summarization": 80,
    "DEFAULT": 30,
}
_1M_CONTEXT_EXTENSION_ROUNDS = 6
_1M_WIDE_SEARCH_TOP_K = 150
_1M_TRIPLET_DISTANCE_PENALTY = 5.0  # softer than default 6.5 for sparser graphs

image = Image.from_dockerfile(
    path=pathlib.Path(path.join(path.dirname(__file__), "Dockerfile")).resolve(),
    force_build=False,
).add_local_python_source("cognee")


@app.function(
    image=image,
    timeout=86400,
    memory=16384,  # 16 GB — 1M conversations need headroom for cognify
    volumes={"/results": vol},
    secrets=[modal.Secret.from_name("eval_secrets")],
)
async def run_beam_conversation(
    conversation_index: int,
    split: str = "100K",
    question_types: Optional[List[str]] = None,
    beam_max_batches: Optional[int] = None,
):
    """Run BEAM eval for a single conversation on Modal."""
    from cognee.shared.logging_utils import get_logger
    from cognee.eval_framework.eval_config import EvalConfig
    from cognee.modules.chunking.ConversationChunker import ConversationChunker
    from cognee.eval_framework.corpus_builder.run_corpus_builder import run_corpus_builder
    from cognee.eval_framework.answer_generation.beam_router import BEAMRouter
    from cognee.eval_framework.evaluation.run_evaluation_module import run_evaluation

    logger = get_logger()
    logger.info(f"=== BEAM eval: conv {conversation_index}, split={split} ===")

    params = EvalConfig(
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
    params["_beam_max_batches"] = beam_max_batches
    params["_beam_conversation_index"] = conversation_index
    params["_beam_split"] = split
    params["chunker"] = ConversationChunker

    # Step 1: Build corpus
    logger.info(f"[conv {conversation_index}] Building corpus...")
    await run_corpus_builder(params)

    # Step 2: Load and optionally filter questions
    with open(params["questions_path"], "r") as f:
        all_questions = json.load(f)

    if question_types:
        target = set(question_types)
        questions = [q for q in all_questions if q.get("question_type") in target]
        logger.info(
            f"[conv {conversation_index}] {len(questions)} questions "
            f"(filtered from {len(all_questions)}) for types: {target}"
        )
    else:
        questions = all_questions
        logger.info(f"[conv {conversation_index}] {len(questions)} questions (all types)")

    if not questions:
        return None

    # Step 3: Answer questions with split-specific retrieval settings
    if split == "1M":
        router = BEAMRouter(
            top_k_overrides=_1M_TOP_K,
            context_extension_rounds=_1M_CONTEXT_EXTENSION_ROUNDS,
            wide_search_top_k=_1M_WIDE_SEARCH_TOP_K,
            triplet_distance_penalty=_1M_TRIPLET_DISTANCE_PENALTY,
        )
    else:
        router = BEAMRouter()
    answers = await router.answer_questions(questions)

    with open(params["answers_path"], "w") as f:
        json.dump(answers, f, ensure_ascii=False, indent=4)

    # Step 4: Evaluate
    params["answering_questions"] = False
    await run_evaluation(params)

    # Step 5: Save results to volume
    with open(params["aggregate_metrics_path"], "r") as f:
        aggregate = json.load(f)
    with open(params["metrics_path"], "r") as f:
        per_question = json.load(f)

    result = {
        "conversation_index": conversation_index,
        "split": split,
        "aggregate": aggregate,
        "per_question": per_question,
    }

    result_path = f"/results/conv_{conversation_index}.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    vol.commit()

    logger.info(f"[conv {conversation_index}] Done. Results saved to {result_path}")
    return result


@app.local_entrypoint()
async def main(
    split: str = "100K",
    num_conversations: int = 20,
    question_types: str = "",
    max_batches: int = 0,
):
    """Run BEAM eval across multiple conversations on Modal.

    Args:
        split: Dataset split ("100K", "500K", "1M")
        num_conversations: Number of conversations to evaluate
        question_types: Comma-separated question types to filter (empty = all)
        max_batches: Max session batches per conversation (0 = all)
    """
    from collections import defaultdict

    types_list = [t.strip() for t in question_types.split(",") if t.strip()] or None
    batches = max_batches if max_batches > 0 else None

    timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    print(f"BEAM eval started at {timestamp}")
    print(f"  split={split}, conversations={num_conversations}")
    print(f"  question_types={types_list or 'ALL'}")
    print(f"  max_batches={batches or 'ALL'}")
    if split == "1M":
        print(f"  top_k overrides: {_1M_TOP_K}")
        print(f"  context_extension_rounds: {_1M_CONTEXT_EXTENSION_ROUNDS}")
        print(f"  wide_search_top_k: {_1M_WIDE_SEARCH_TOP_K}")
        print(f"  triplet_distance_penalty: {_1M_TRIPLET_DISTANCE_PENALTY}")

    # Launch all conversations in parallel on Modal
    tasks = [
        run_beam_conversation.remote.aio(
            conversation_index=i,
            split=split,
            question_types=types_list,
            beam_max_batches=batches,
        )
        for i in range(num_conversations)
    ]
    results = await asyncio.gather(*tasks)

    # Aggregate results
    all_aggregate = []
    type_scores = defaultdict(lambda: defaultdict(list))

    for result in results:
        if result is None:
            continue
        all_aggregate.append(result["aggregate"])
        for entry in result["per_question"]:
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
        "split": split,
        "num_conversations": len(all_aggregate),
        "question_types": types_list,
        "timestamp": timestamp,
        "avg_per_type": avg_per_type,
        "per_conversation": all_aggregate,
    }

    # Save combined results to volume
    combined_path = f"/results/beam_combined_{split}_{timestamp}.json"
    # Also save locally
    local_path = f"beam_combined_{split}_{timestamp}.json"
    with open(local_path, "w") as f:
        json.dump(combined, f, indent=2)

    # Print summary
    print(f"\n=== BEAM Results ({split}, {len(all_aggregate)} conversations) ===")
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

    print(f"\nResults saved to: {local_path}")
