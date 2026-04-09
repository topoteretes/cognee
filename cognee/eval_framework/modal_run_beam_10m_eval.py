"""Run BEAM-10M benchmark evaluation on Modal.

Each conversation has 10 plans (~1M tokens each). To avoid OOM, we cognify
one plan at a time, then answer all 20 probing questions against the combined
knowledge graph.

Retrieval settings are scaled up vs. 1M:
  - top_k: 50 for most types, 150 for summarization (1M uses 30/80)
  - context_extension_rounds: 8 (vs. 6 for 1M)
  - wide_search_top_k: 300 (vs. 150 for 1M)
  - triplet_distance_penalty: 4.0 (vs. 5.0 for 1M)

Usage:
    # Run with defaults (all 10 conversations, all plans, per-plan ingestion)
    modal run cognee/eval_framework/modal_run_beam_10m_eval.py

    # Run 3 conversations only
    modal run cognee/eval_framework/modal_run_beam_10m_eval.py --num-conversations 3

    # Run specific question types only
    modal run cognee/eval_framework/modal_run_beam_10m_eval.py --question-types "summarization,knowledge_update"

    # Limit batches per plan (faster testing)
    modal run cognee/eval_framework/modal_run_beam_10m_eval.py --max-batches-per-plan 3

    # Run only specific plans (e.g., plan-1 and plan-2)
    modal run cognee/eval_framework/modal_run_beam_10m_eval.py --plans "plan-1,plan-2"
"""

import asyncio
import datetime
import json
import os
from typing import List, Optional

import modal
from modal import Image

vol = modal.Volume.from_name("beam_10m_eval_results", create_if_missing=True)

app = modal.App("beam-10m-benchmark-eval")

image = (
    Image.debian_slim(python_version="3.12")
    .apt_install("gcc", "libpq-dev", "git", "curl", "build-essential")
    .pip_install(
        "typing_extensions>=4.14",
        "pydantic>=2.0",
        "pydantic-core>=2.0",
        "pydantic-settings>=2.0",
    )
    .pip_install("cognee[distributed,evals,deepeval]", "datasets")
    .add_local_dir("cognee", remote_path="/root/cognee")
)

# 10M-scale retrieval settings (10x more chunks than 1M)
_10M_TOP_K = {
    "summarization": 150,
    "DEFAULT": 50,
}
_10M_CONTEXT_EXTENSION_ROUNDS = 8
_10M_WIDE_SEARCH_TOP_K = 300  # default is 100; larger graph needs wider candidate pool
_10M_TRIPLET_DISTANCE_PENALTY = 4.0  # default is 6.5; softer penalty for large sparse graphs
_10M_CHUNKS_PER_BATCH = 40  # default is 100; reduced to lower peak memory during cognify

# BEAM-10M has 10 plans per conversation
ALL_PLANS = [f"plan-{i}" for i in range(1, 11)]


@app.function(
    image=image,
    timeout=86400,
    memory=65536,  # 64 GB — graph accumulates across 10 plans; LanceDB indexing needs headroom
    volumes={"/results": vol},
    secrets=[
        modal.Secret.from_name("eval_secrets"),
        modal.Secret.from_dict({
            "COGNEE_SKIP_CONNECTION_TEST": "true",
            "LITELLM_LOG": "ERROR",
            "LLM_MAX_CONCURRENT": "40",
        }),
    ],
)
async def run_beam_10m_conversation(
    conversation_index: int,
    question_types: Optional[List[str]] = None,
    max_batches_per_plan: Optional[int] = None,
    plans: Optional[List[str]] = None,
):
    """Run BEAM-10M eval for a single conversation on Modal.

    Cognifies each plan separately to avoid OOM, then answers questions
    against the full accumulated knowledge graph.
    """
    from cognee.shared.logging_utils import get_logger
    from cognee.eval_framework.eval_config import EvalConfig
    from cognee.modules.chunking.ConversationChunker import ConversationChunker
    from cognee.eval_framework.corpus_builder.run_corpus_builder import run_corpus_builder
    from cognee.eval_framework.answer_generation.beam_router import BEAMRouter
    from cognee.eval_framework.evaluation.run_evaluation_module import run_evaluation

    logger = get_logger()

    selected_plans = plans or ALL_PLANS
    logger.info(
        f"=== BEAM-10M eval: conv {conversation_index}, "
        f"plans={selected_plans}, max_batches_per_plan={max_batches_per_plan or 'ALL'} ==="
    )

    # Helper to save progress checkpoints to the volume
    def _save_checkpoint(stage: str, data: dict):
        checkpoint_path = f"/results/conv_{conversation_index}_checkpoint.json"
        data["_stage"] = stage
        data["_conversation_index"] = conversation_index
        with open(checkpoint_path, "w") as f:
            json.dump(data, f, indent=2)
        vol.commit()
        logger.info(f"[conv {conversation_index}] Checkpoint saved: {stage}")

    # Resume from checkpoint if this is a retry — skip already-completed plans
    completed_plans = set()
    checkpoint_path = f"/results/conv_{conversation_index}_checkpoint.json"
    vol.reload()
    try:
        with open(checkpoint_path, "r") as f:
            checkpoint = json.load(f)
        completed_plans = set(checkpoint.get("plans_completed", []))
        if completed_plans:
            logger.info(
                f"[conv {conversation_index}] Resuming — "
                f"skipping already-completed plans: {sorted(completed_plans)}"
            )
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    # Step 1: Build corpus plan-by-plan to avoid OOM.
    # Each plan is ~1M tokens. We cognify them sequentially so the graph
    # accumulates across plans but memory stays bounded.
    for plan_idx, plan_name in enumerate(selected_plans):
        if plan_name in completed_plans:
            logger.info(
                f"[conv {conversation_index}] Skipping {plan_name} (already done)"
            )
            continue

        is_first = plan_idx == 0 and not completed_plans
        logger.info(
            f"[conv {conversation_index}] Cognifying {plan_name} "
            f"({plan_idx + 1}/{len(selected_plans)})..."
        )

        params = EvalConfig(
            benchmark="BEAM-10M",
            building_corpus_from_scratch=True,
            number_of_samples_in_corpus=20,
            qa_engine="beam_router",
            answering_questions=False,
            evaluating_answers=False,
            evaluating_contexts=False,
            evaluation_engine="DeepEval",
            evaluation_metrics=["beam_rubric", "kendall_tau"],
            task_getter_type="Default",
            calculate_metrics=False,
            dashboard=False,
            questions_path=f"beam10m_questions_conv{conversation_index}.json",
            answers_path=f"beam10m_answers_conv{conversation_index}.json",
            metrics_path=f"beam10m_metrics_conv{conversation_index}.json",
            aggregate_metrics_path=f"beam10m_aggregate_metrics_conv{conversation_index}.json",
            dashboard_path=f"beam10m_dashboard_conv{conversation_index}.html",
        ).to_dict()

        params["_beam_max_batches"] = max_batches_per_plan
        params["_beam_conversation_index"] = conversation_index
        params["_beam_plans"] = [plan_name]  # Single plan at a time
        params["dataset_name"] = f"beam10m_conv{conversation_index}_{plan_name}"
        params["chunker"] = ConversationChunker
        params["chunks_per_batch"] = _10M_CHUNKS_PER_BATCH
        # Only prune on first plan; subsequent plans accumulate into the graph
        params["_skip_prune"] = not is_first

        await run_corpus_builder(params)

        # Checkpoint after each plan so we know how far we got
        completed_plans.add(plan_name)
        all_done = [p for p in selected_plans if p in completed_plans]
        remaining = [p for p in selected_plans if p not in completed_plans]
        _save_checkpoint(f"cognified_{plan_name}", {
            "plans_completed": all_done,
            "plans_remaining": remaining,
        })

    # Step 2: Load and optionally filter questions
    questions_path = f"beam10m_questions_conv{conversation_index}.json"
    with open(questions_path, "r") as f:
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

    # Step 3: Answer questions with 10M-scale retrieval settings
    router = BEAMRouter(
        top_k_overrides=_10M_TOP_K,
        context_extension_rounds=_10M_CONTEXT_EXTENSION_ROUNDS,
        wide_search_top_k=_10M_WIDE_SEARCH_TOP_K,
        triplet_distance_penalty=_10M_TRIPLET_DISTANCE_PENALTY,
    )
    answers = await router.answer_questions(questions)

    answers_path = f"beam10m_answers_conv{conversation_index}.json"
    with open(answers_path, "w") as f:
        json.dump(answers, f, ensure_ascii=False, indent=4)

    # Checkpoint after answering — this is the expensive part
    _save_checkpoint("answered", {
        "plans_completed": selected_plans,
        "num_answers": len(answers),
    })

    # Step 4: Evaluate
    eval_params = EvalConfig(
        benchmark="BEAM-10M",
        building_corpus_from_scratch=False,
        number_of_samples_in_corpus=20,
        qa_engine="beam_router",
        answering_questions=False,
        evaluating_answers=True,
        evaluating_contexts=False,
        evaluation_engine="DeepEval",
        evaluation_metrics=["beam_rubric", "kendall_tau"],
        task_getter_type="Default",
        calculate_metrics=True,
        dashboard=False,
        questions_path=f"beam10m_questions_conv{conversation_index}.json",
        answers_path=f"beam10m_answers_conv{conversation_index}.json",
        metrics_path=f"beam10m_metrics_conv{conversation_index}.json",
        aggregate_metrics_path=f"beam10m_aggregate_metrics_conv{conversation_index}.json",
        dashboard_path=f"beam10m_dashboard_conv{conversation_index}.html",
    ).to_dict()
    await run_evaluation(eval_params)

    # Step 5: Save final results to volume
    with open(f"beam10m_aggregate_metrics_conv{conversation_index}.json", "r") as f:
        aggregate = json.load(f)
    with open(f"beam10m_metrics_conv{conversation_index}.json", "r") as f:
        per_question = json.load(f)

    result = {
        "conversation_index": conversation_index,
        "plans": selected_plans,
        "aggregate": aggregate,
        "per_question": per_question,
    }

    result_path = f"/results/conv_{conversation_index}.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    # Also save the raw answers to the volume for debugging
    vol_answers_path = f"/results/conv_{conversation_index}_answers.json"
    with open(vol_answers_path, "w") as f:
        json.dump(answers, f, ensure_ascii=False, indent=2)

    vol.commit()

    logger.info(f"[conv {conversation_index}] Done. Results saved to {result_path}")
    return result


@app.function(
    image=image,
    timeout=86400,  # 24h — with max_parallel=5, all batches fit within this
    volumes={"/results": vol},
    secrets=[modal.Secret.from_name("eval_secrets")],
)
async def orchestrate_beam_10m(
    num_conversations: int = 10,
    question_types: Optional[List[str]] = None,
    max_batches_per_plan: Optional[int] = None,
    plans: Optional[List[str]] = None,
    max_parallel: int = 5,
):
    """Orchestrator that runs entirely on Modal — survives laptop sleep.

    Spawns conversation workers in batches, collects results, and saves
    the combined report to the volume.
    """
    from collections import defaultdict
    from cognee.shared.logging_utils import get_logger

    logger = get_logger()

    timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    logger.info(
        f"BEAM-10M orchestrator started at {timestamp}: "
        f"conversations={num_conversations}, max_parallel={max_parallel}, "
        f"plans={plans or 'ALL'}, question_types={question_types or 'ALL'}, "
        f"max_batches_per_plan={max_batches_per_plan or 'ALL'}"
    )

    results = []
    failed = []

    for batch_start in range(0, num_conversations, max_parallel):
        batch_end = min(batch_start + max_parallel, num_conversations)
        batch_indices = list(range(batch_start, batch_end))
        logger.info(f"--- Batch: conversations {batch_indices} ---")

        function_calls = []
        for i in batch_indices:
            fc = await run_beam_10m_conversation.spawn.aio(
                conversation_index=i,
                question_types=question_types,
                max_batches_per_plan=max_batches_per_plan,
                plans=plans,
            )
            function_calls.append((i, fc))
            logger.info(f"  Spawned conv {i}: {fc.object_id}")

        for i, fc in function_calls:
            try:
                result = await fc.get.aio()
                results.append(result)
                logger.info(f"  [DONE] conv {i}")
            except Exception as e:
                logger.error(f"  [FAILED] conv {i}: {type(e).__name__}: {e}")
                failed.append(i)
                results.append(None)

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

    avg_per_type = {}
    for qtype, metrics in sorted(type_scores.items()):
        avg_per_type[qtype] = {
            m: sum(s) / len(s) if s else None for m, s in metrics.items()
        }

    combined = {
        "dataset": "BEAM-10M",
        "num_conversations": len(all_aggregate),
        "num_failed": len(failed),
        "failed_conversations": failed,
        "plans": plans,
        "question_types": question_types,
        "timestamp": timestamp,
        "retrieval_config": {
            "top_k_overrides": _10M_TOP_K,
            "context_extension_rounds": _10M_CONTEXT_EXTENSION_ROUNDS,
            "wide_search_top_k": _10M_WIDE_SEARCH_TOP_K,
            "triplet_distance_penalty": _10M_TRIPLET_DISTANCE_PENALTY,
            "chunks_per_batch": _10M_CHUNKS_PER_BATCH,
        },
        "avg_per_type": avg_per_type,
        "per_conversation": all_aggregate,
    }

    # Save combined results to volume
    result_path = f"/results/beam10m_combined_{timestamp}.json"
    with open(result_path, "w") as f:
        json.dump(combined, f, indent=2)
    vol.commit()

    logger.info(f"\n=== BEAM-10M Results ({len(all_aggregate)} conversations, {len(failed)} failed) ===")
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

    all_rubric = [
        m.get("beam_rubric") for m in avg_per_type.values() if m.get("beam_rubric") is not None
    ]
    if all_rubric:
        logger.info(f"  OVERALL beam_rubric avg: {sum(all_rubric) / len(all_rubric):.3f}")

    logger.info(f"Results saved to volume: {result_path}")
    return combined


@app.local_entrypoint()
async def main(
    num_conversations: int = 10,
    question_types: str = "",
    max_batches_per_plan: int = 0,
    plans: str = "",
    max_parallel: int = 5,
):
    """Trigger the DEPLOYED orchestrator on Modal and exit immediately.

    The orchestrator runs entirely on Modal — safe to close your laptop.

    Usage:
        # Deploy first (one-time):
        modal deploy cognee/eval_framework/modal_run_beam_10m_eval.py

        # Then trigger:
        modal run cognee/eval_framework/modal_run_beam_10m_eval.py

    Monitor progress:
        modal volume ls beam_10m_eval_results
    """
    types_list = [t.strip() for t in question_types.split(",") if t.strip()] or None
    batches = max_batches_per_plan if max_batches_per_plan > 0 else None
    plans_list = [p.strip() for p in plans.split(",") if p.strip()] or None

    print(f"Looking up deployed orchestrator...")

    # Look up the DEPLOYED function so it runs under the persistent app,
    # not this ephemeral `modal run` app.
    deployed_orchestrator = modal.Function.from_name(
        "beam-10m-benchmark-eval", "orchestrate_beam_10m"
    )

    print(f"Spawning orchestrator on deployed app...")
    print(f"  conversations={num_conversations}, max_parallel={max_parallel}")
    print(f"  plans={plans_list or 'ALL'}")
    print(f"  question_types={types_list or 'ALL'}")
    print(f"  max_batches_per_plan={batches or 'ALL'}")

    fc = await deployed_orchestrator.spawn.aio(
        num_conversations=num_conversations,
        question_types=types_list,
        max_batches_per_plan=batches,
        plans=plans_list,
        max_parallel=max_parallel,
    )
    print(f"\nOrchestrator spawned: {fc.object_id}")
    print("All work runs on the DEPLOYED app — safe to close your laptop.")
    print("\nMonitor progress:")
    print("  modal volume ls beam_10m_eval_results")
    print("  modal container list")
    print("\nWhen done, download results:")
    print("  modal volume get beam_10m_eval_results beam10m_combined_*.json .")
