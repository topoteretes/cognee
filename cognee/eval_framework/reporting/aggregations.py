import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Optional


def _safe_mean(values: list[float]) -> Optional[float]:
    return (sum(values) / len(values)) if values else None


def _safe_stdev(values: list[float]) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    return statistics.stdev(values)


def average_aggregate_metrics(all_metrics: list[dict[str, Any]]) -> dict[str, Optional[float]]:
    if not all_metrics:
        return {}

    averaged_metrics = {}
    for key in all_metrics[0]:
        values = [
            metrics[key] for metrics in all_metrics if key in metrics and metrics[key] is not None
        ]
        if values and all(isinstance(value, (int, float)) for value in values):
            averaged_metrics[key] = sum(values) / len(values)
        else:
            averaged_metrics[key] = None

    return averaged_metrics


def average_metrics_by_question_type(
    all_per_conversation_metrics: list[list[dict[str, Any]]],
) -> dict[str, dict[str, Optional[float]]]:
    type_scores = defaultdict(lambda: defaultdict(list))

    for conversation_metrics in all_per_conversation_metrics:
        for entry in conversation_metrics:
            question_type = entry.get("question_type", "unknown")
            metrics = entry.get("metrics", {})
            for metric_name, metric_data in metrics.items():
                score = metric_data.get("score")
                if score is not None:
                    type_scores[question_type][metric_name].append(score)

    result = {}
    for question_type, metrics in sorted(type_scores.items()):
        result[question_type] = {}
        for metric_name, scores in metrics.items():
            result[question_type][metric_name] = _safe_mean(scores)

    return result


def extract_metric_score(entry: dict[str, Any], metric_name: str) -> Optional[float]:
    metric_data = entry.get("metrics", {}).get(metric_name, {})
    score = metric_data.get("score")
    return score if isinstance(score, (int, float)) else None


def build_per_question_records(
    batch_results: list[dict[str, Any]],
    retriever_names: list[str],
    primary_metric_name: str,
) -> list[dict[str, Any]]:
    per_question: dict[tuple[Any, int], dict[str, Any]] = {}

    for batch in batch_results:
        retriever_name = batch["retriever_name"]
        run_idx = batch["run_idx"]

        for entry in batch["metrics"]:
            question_key = (entry.get("conversation_id", "unknown"), entry["question_idx"])
            question_record = per_question.setdefault(
                question_key,
                {
                    "conversation_id": entry.get("conversation_id", "unknown"),
                    "question_idx": entry["question_idx"],
                    "question": entry["question"],
                    "golden_answer": entry["golden_answer"],
                    "question_type": entry.get("question_type", "unknown"),
                    "difficulty": entry.get("difficulty"),
                    "retrievers": {},
                },
            )

            retriever_record = question_record["retrievers"].setdefault(
                retriever_name,
                {
                    "runs": [],
                    "scores_by_run": [],
                    "mean_score": None,
                    "std_score": None,
                },
            )

            primary_score = extract_metric_score(entry, primary_metric_name)
            retriever_record["runs"].append(
                {
                    "run_idx": run_idx,
                    "score": primary_score,
                    "answer": entry.get("answer", ""),
                    "retrieval_context": entry.get("retrieval_context", ""),
                    "metrics": entry.get("metrics", {}),
                }
            )

    sorted_question_records = []
    for question_key in sorted(per_question.keys(), key=lambda item: (str(item[0]), item[1])):
        question_record = per_question[question_key]

        for retriever_name in retriever_names:
            retriever_record = question_record["retrievers"].setdefault(
                retriever_name,
                {
                    "runs": [],
                    "scores_by_run": [],
                    "mean_score": None,
                    "std_score": None,
                },
            )
            retriever_record["runs"].sort(key=lambda item: item["run_idx"])

            scores = [
                run["score"]
                for run in retriever_record["runs"]
                if isinstance(run.get("score"), (int, float))
            ]
            retriever_record["scores_by_run"] = scores
            retriever_record["mean_score"] = _safe_mean(scores)
            retriever_record["std_score"] = _safe_stdev(scores)

        candidate_scores = {
            name: data["mean_score"]
            for name, data in question_record["retrievers"].items()
            if data["mean_score"] is not None
        }

        if candidate_scores:
            best_score = max(candidate_scores.values())
            winner_ties = [
                name
                for name, score in candidate_scores.items()
                if math.isclose(score, best_score, rel_tol=0.0, abs_tol=1e-12)
            ]
            question_record["winner_ties"] = winner_ties
            question_record["winner"] = winner_ties[0] if len(winner_ties) == 1 else None
            question_record["oracle_score"] = best_score
        else:
            question_record["winner_ties"] = []
            question_record["winner"] = None
            question_record["oracle_score"] = None

        sorted_question_records.append(question_record)

    return sorted_question_records


def build_group_aggregates(
    per_question_records: list[dict[str, Any]],
    retriever_names: list[str],
) -> dict[str, Any]:
    def aggregate_question_group(question_group: list[dict[str, Any]]) -> dict[str, Any]:
        question_count = len(question_group)
        aggregate = {}

        for retriever_name in retriever_names:
            mean_scores = []
            std_scores = []
            oracle_gaps = []
            win_count = 0
            tie_count = 0

            for question in question_group:
                retriever_record = question["retrievers"][retriever_name]
                mean_score = retriever_record["mean_score"]
                std_score = retriever_record["std_score"]

                if mean_score is not None:
                    mean_scores.append(mean_score)
                    if question["oracle_score"] is not None:
                        oracle_gaps.append(question["oracle_score"] - mean_score)
                if std_score is not None:
                    std_scores.append(std_score)
                if question.get("winner") == retriever_name:
                    win_count += 1
                if retriever_name in question.get("winner_ties", []):
                    tie_count += 1

            aggregate[retriever_name] = {
                "question_count": question_count,
                "scored_question_count": len(mean_scores),
                "mean_score": _safe_mean(mean_scores),
                "variance_summary": _safe_mean(std_scores),
                "win_rate": (win_count / question_count) if question_count else None,
                "tie_rate": (tie_count / question_count) if question_count else None,
                "oracle_gap": _safe_mean(oracle_gaps),
            }

        return aggregate

    grouped_records: dict[str, list[dict[str, Any]]] = {}
    for question_record in per_question_records:
        question_type = question_record.get("question_type", "unknown")
        grouped_records.setdefault(question_type, []).append(question_record)

    per_question_type = {
        question_type: aggregate_question_group(question_group)
        for question_type, question_group in sorted(grouped_records.items())
    }

    return {
        "per_question_type": per_question_type,
        "overall": aggregate_question_group(per_question_records),
    }


def build_empty_conversation_summary(
    conversation_index: int,
    settings: Any,
    retriever_names: list[str],
) -> dict[str, Any]:
    return {
        **settings.summary_tags,
        "conversation_index": conversation_index,
        "conversation_id": None,
        "num_runs": settings.num_runs,
        "num_questions": 0,
        "question_types_filter": settings.question_types,
        "retrievers": retriever_names,
        "output_dir": str(settings.output_dir),
        "batch_artifacts": [],
        "per_question": [],
        "per_question_type": {},
        "overall": {},
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "skipped": True,
    }


def build_conversation_summary(
    conversation_index: int,
    settings: Any,
    batch_results: list[dict[str, Any]],
    retriever_configs: list[dict[str, Any]],
) -> dict[str, Any]:
    retriever_names = [config["name"] for config in retriever_configs]
    per_question_records = build_per_question_records(
        batch_results, retriever_names, settings.primary_metric_name
    )

    if not per_question_records:
        return build_empty_conversation_summary(conversation_index, settings, retriever_names)

    aggregate_sections = build_group_aggregates(per_question_records, retriever_names)
    conversation_id = per_question_records[0].get("conversation_id")

    return {
        **settings.summary_tags,
        "conversation_index": conversation_index,
        "conversation_id": conversation_id,
        "num_runs": settings.num_runs,
        "num_questions": len(per_question_records),
        "question_types_filter": settings.question_types,
        "retrievers": retriever_names,
        "output_dir": str(settings.output_dir),
        "batch_artifacts": [
            {
                "retriever_name": batch["retriever_name"],
                "run_idx": batch["run_idx"],
                "answers_path": batch["answers_path"],
                "metrics_path": batch["metrics_path"],
                "aggregate_metrics_path": batch["aggregate_metrics_path"],
                "aggregate_metrics": batch["aggregate_metrics"],
            }
            for batch in batch_results
        ],
        "per_question": per_question_records,
        "per_question_type": aggregate_sections["per_question_type"],
        "overall": aggregate_sections["overall"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "skipped": False,
    }


def build_combined_summary(
    conversation_summaries: list[dict[str, Any]],
    settings: Any,
    retriever_configs: list[dict[str, Any]],
    num_requested_conversations: int,
) -> dict[str, Any]:
    retriever_names = [config["name"] for config in retriever_configs]
    usable_conversations = [
        summary for summary in conversation_summaries if not summary.get("skipped")
    ]

    all_per_question = []
    for conversation_summary in usable_conversations:
        all_per_question.extend(conversation_summary.get("per_question", []))

    aggregate_sections = build_group_aggregates(all_per_question, retriever_names)

    return {
        **settings.summary_tags,
        "num_requested_conversations": num_requested_conversations,
        "num_completed_conversations": len(conversation_summaries),
        "num_scored_conversations": len(usable_conversations),
        "num_runs": settings.num_runs,
        "question_types_filter": settings.question_types,
        "retrievers": retriever_names,
        "output_dir": str(settings.output_dir),
        "overall": aggregate_sections["overall"],
        "per_question_type": aggregate_sections["per_question_type"],
        "per_conversation": [
            {
                "conversation_index": summary["conversation_index"],
                "conversation_id": summary.get("conversation_id"),
                "num_questions": summary.get("num_questions", 0),
                "skipped": summary.get("skipped", False),
                "overall": summary.get("overall", {}),
            }
            for summary in conversation_summaries
        ],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def build_best_retriever_by_question_type_report(
    conversation_summaries: list[dict[str, Any]],
    settings: Any,
    retriever_configs: list[dict[str, Any]],
) -> dict[str, Any]:
    retriever_names = [config["name"] for config in retriever_configs]
    usable_conversations = [
        summary for summary in conversation_summaries if not summary.get("skipped")
    ]

    all_per_question = []
    for conversation_summary in usable_conversations:
        all_per_question.extend(conversation_summary.get("per_question", []))

    grouped_records: dict[str, list[dict[str, Any]]] = {}
    for question_record in all_per_question:
        question_type = question_record.get("question_type", "unknown")
        grouped_records.setdefault(question_type, []).append(question_record)

    selected_policy_scores = []
    selected_policy_std_scores = []
    per_question_type = {}

    for question_type, question_group in sorted(grouped_records.items()):
        candidate_scores = {}
        for retriever_name in retriever_names:
            scores = [
                question["retrievers"][retriever_name]["mean_score"]
                for question in question_group
                if question["retrievers"][retriever_name]["mean_score"] is not None
            ]
            candidate_scores[retriever_name] = _safe_mean(scores)

        scored_candidates = {
            name: score for name, score in candidate_scores.items() if score is not None
        }
        if not scored_candidates:
            per_question_type[question_type] = {
                "question_count": len(question_group),
                "scored_question_count": 0,
                "selected_retriever": None,
                "selected_retriever_ties": [],
                "selected_mean_score": None,
                "selected_variance_summary": None,
                "candidate_mean_scores": candidate_scores,
            }
            continue

        best_score = max(scored_candidates.values())
        best_retriever_ties = sorted(
            [
                name
                for name, score in scored_candidates.items()
                if math.isclose(score, best_score, rel_tol=0.0, abs_tol=1e-12)
            ]
        )
        selected_retriever = best_retriever_ties[0]

        question_scores = []
        question_std_scores = []
        for question in question_group:
            retriever_record = question["retrievers"][selected_retriever]
            if retriever_record["mean_score"] is not None:
                question_scores.append(retriever_record["mean_score"])
            if retriever_record["std_score"] is not None:
                question_std_scores.append(retriever_record["std_score"])

        selected_policy_scores.extend(question_scores)
        selected_policy_std_scores.extend(question_std_scores)

        per_question_type[question_type] = {
            "question_count": len(question_group),
            "scored_question_count": len(question_scores),
            "selected_retriever": selected_retriever,
            "selected_retriever_ties": best_retriever_ties,
            "selected_mean_score": _safe_mean(question_scores),
            "selected_variance_summary": _safe_mean(question_std_scores),
            "candidate_mean_scores": candidate_scores,
        }

    return {
        **settings.summary_tags,
        "policy": "best_retriever_by_question_type",
        "selection_metric": settings.primary_metric_name,
        "question_types_filter": settings.question_types,
        "retrievers": retriever_names,
        "output_dir": str(settings.output_dir),
        "overall": {
            "question_count": len(all_per_question),
            "scored_question_count": len(selected_policy_scores),
            "mean_score": _safe_mean(selected_policy_scores),
            "variance_summary": _safe_mean(selected_policy_std_scores),
        },
        "per_question_type": per_question_type,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
