"""Aggregate BEAM metrics across repeated runs for one retriever."""

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from cognee.eval_framework.analysis.metrics_calculator import bootstrap_ci

ARTIFACT_PREFIX = "beam_existing_ingestion"
METRICS = ("beam_rubric", "kendall_tau")


def slugify(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_") or "retriever"


def empty_metrics() -> dict[str, list[float]]:
    return {metric: [] for metric in METRICS}


def get_score(entry: dict, metric: str) -> float | None:
    score = entry.get("metrics", {}).get(metric, {}).get("score")
    return score if isinstance(score, (int, float)) else None


def scores_from_entry(entry: dict) -> tuple[str, list[tuple[str, float]]]:
    question_type = entry.get("question_type", "unknown")
    return question_type, [
        (metric, score) for metric in METRICS if (score := get_score(entry, metric)) is not None
    ]


def collect_run_scores(run: list[dict]) -> dict[str, dict[str, list[float]]]:
    by_type: dict[str, dict[str, list[float]]] = defaultdict(empty_metrics)
    for entry in run:
        question_type, scores = scores_from_entry(entry)
        for metric, score in scores:
            by_type[question_type][metric].append(score)
    return by_type


def overall_from_by_type(by_type: dict[str, dict[str, list[float]]]) -> dict[str, list[float]]:
    overall = empty_metrics()
    for metrics in by_type.values():
        for metric, scores in metrics.items():
            overall[metric].extend(scores)
    return overall


def collect_scores(runs: list[list[dict]]) -> dict[str, list[dict]]:
    type_runs: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        by_type = collect_run_scores(run)
        for question_type, metrics in by_type.items():
            type_runs[question_type].append(metrics)
    return type_runs


def validate_runs(runs: list[list[dict]]) -> list[int]:
    question_ids = [entry["question_idx"] for entry in runs[0]]
    if any([entry["question_idx"] for entry in run] != question_ids for run in runs[1:]):
        raise ValueError("All runs must contain the same questions in the same order")
    return question_ids


def stats(runs_scores: list[dict[str, list[float]]], metric: str) -> dict | None:
    all_scores = []
    run_means = []
    for run in runs_scores:
        scores = run[metric]
        all_scores.extend(scores)
        if scores:
            run_means.append(sum(scores) / len(scores))
    if not all_scores:
        return None
    mean, ci_lower, ci_upper = bootstrap_ci(all_scores)
    return {
        "mean": mean,
        "run_std": statistics.stdev(run_means) if len(run_means) >= 2 else None,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


def metric_stats(runs_scores: list[dict[str, list[float]]]) -> dict:
    return {metric: stats(runs_scores, metric) for metric in METRICS}


def build_summary(
    retriever: str,
    conversation_index: int,
    runs: list[list[dict]],
    type_runs: dict[str, list[dict]],
) -> dict:
    type_counts = Counter(entry.get("question_type", "unknown") for entry in runs[0])
    overall_runs = [
        overall_from_by_type(
            {question_type: type_runs[question_type][run_idx] for question_type in type_runs}
        )
        for run_idx in range(len(runs))
    ]
    return {
        "retriever_name": retriever,
        "conversation_index": conversation_index,
        "num_runs": len(runs),
        "question_count": len(runs[0]),
        "overall": metric_stats(overall_runs),
        "by_question_type": {
            question_type: {
                "question_count": type_counts[question_type],
                **metric_stats(type_runs[question_type]),
            }
            for question_type in sorted(type_counts)
        },
    }


def find_metrics_paths(
    output_dir: Path,
    retriever: str,
    conversation_index: int,
    artifact_prefix: str = ARTIFACT_PREFIX,
) -> list[Path]:
    pattern = f"{artifact_prefix}_metrics_conv{conversation_index}_{slugify(retriever)}_run*.json"
    paths = sorted(output_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No metrics files matched: {output_dir / pattern}")
    return paths


def load_runs(paths: list[Path]) -> list[list[dict]]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def aggregate_cross_run(retriever: str, conversation_index: int, paths: list[Path]) -> dict:
    runs = load_runs(paths)
    validate_runs(runs)
    type_runs = collect_scores(runs)
    return build_summary(retriever, conversation_index, runs, type_runs)


def run_aggregate_cross_run(
    output_dir: Path,
    retriever: str,
    conversation_index: int = 0,
    artifact_prefix: str = ARTIFACT_PREFIX,
    out: Path | None = None,
) -> Path:
    paths = find_metrics_paths(output_dir, retriever, conversation_index, artifact_prefix)
    summary = aggregate_cross_run(retriever, conversation_index, paths)
    out_path = out or output_dir / f"{retriever}_cross_run_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--retriever", required=True)
    parser.add_argument("--conversation-index", type=int, default=0)
    parser.add_argument("--artifact-prefix", default=ARTIFACT_PREFIX)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    print(
        run_aggregate_cross_run(
            args.output_dir,
            args.retriever,
            args.conversation_index,
            args.artifact_prefix,
            args.out,
        )
    )


if __name__ == "__main__":
    main()
