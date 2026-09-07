"""Aggregate LoCoMo metrics across conversations, retrievers and repeated runs.

Reads every ``qa/locomo_metrics_conv<i>_<retriever>_run<r>.json`` under a run directory and
produces, per retriever:

- overall and per-category mean F1 and LLM-judge accuracy (pooled over all conversations),
- a ``mem0_comparable`` slice (categories single_hop / multi_hop / temporal / open_domain,
  i.e. adversarial excluded, the subset the mem0 paper reports on),
- per-run means and their std (run-to-run variation of answering + judging),
- a descriptive pooled bootstrap CI.

Writes ``locomo_summary.json`` and ``locomo_summary.md`` into the run directory.
"""

from __future__ import annotations

import argparse
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from cognee.eval_framework.analysis.metrics_calculator import bootstrap_ci
from cognee.eval_framework.reporting.io import read_json, write_json

METRICS = ("f1", "llm_judge")
MEM0_COMPARABLE_TYPES = ("single_hop", "multi_hop", "temporal", "open_domain")
CATEGORY_ORDER = ("single_hop", "multi_hop", "temporal", "open_domain", "adversarial")
_FILE_RE = re.compile(r"locomo_metrics_conv(?P<conv>\d+)_(?P<retriever>.+)_run(?P<run>\d+)\.json$")


def find_metric_files(run_dir: Path) -> list[dict[str, Any]]:
    found = []
    for path in sorted(run_dir.rglob("locomo_metrics_conv*_run*.json")):
        match = _FILE_RE.search(path.name)
        if not match:
            continue
        found.append(
            {
                "path": path,
                "conversation_index": int(match.group("conv")),
                "retriever": match.group("retriever"),
                "run_idx": int(match.group("run")),
            }
        )
    return found


def _score(entry: dict[str, Any], metric: str) -> float | None:
    value = (entry.get("metrics") or {}).get(metric) or {}
    score = value.get("score")
    return float(score) if isinstance(score, (int, float)) else None


def _stats(scores: list[float]) -> dict[str, Any] | None:
    if not scores:
        return None
    mean = statistics.fmean(scores)
    result: dict[str, Any] = {"n": len(scores), "mean": mean}
    if len(scores) >= 2:
        _, low, high = bootstrap_ci(scores, num_samples=2000)
        result["ci_lower"] = low
        result["ci_upper"] = high
    return result


def aggregate_run_dir(run_dir: Path) -> dict[str, Any]:
    files = find_metric_files(run_dir)
    if not files:
        raise FileNotFoundError(f"No locomo_metrics_conv*_run*.json files under {run_dir}")

    # retriever -> metric -> question_type -> list of scores (pooled over convs and runs)
    pooled: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    # retriever -> metric -> run_idx -> list of scores (for run-to-run std)
    per_run: dict[str, dict[str, dict[int, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    # retriever -> conversation -> metric -> scores
    per_conv: dict[str, dict[int, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    unscored: dict[str, int] = defaultdict(int)
    errors: dict[str, int] = defaultdict(int)
    conversations = set()

    for info in files:
        entries = read_json(str(info["path"]))
        retriever = info["retriever"]
        conversations.add(info["conversation_index"])
        for entry in entries:
            question_type = entry.get("question_type", "unknown")
            if isinstance(entry.get("answer"), str) and entry["answer"].startswith("ERROR:"):
                errors[retriever] += 1
            for metric in METRICS:
                score = _score(entry, metric)
                if score is None:
                    unscored[f"{retriever}/{metric}"] += 1
                    continue
                pooled[retriever][metric][question_type].append(score)
                per_run[retriever][metric][info["run_idx"]].append(score)
                per_conv[retriever][info["conversation_index"]][metric].append(score)

    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "conversations": sorted(conversations),
        "metric_files": len(files),
        "retrievers": {},
    }

    for retriever in sorted(pooled):
        block: dict[str, Any] = {"metrics": {}, "answer_errors": errors.get(retriever, 0)}
        for metric in METRICS:
            by_type = pooled[retriever][metric]
            all_scores = [s for scores in by_type.values() for s in scores]
            comparable = [
                s for qt, scores in by_type.items() if qt in MEM0_COMPARABLE_TYPES for s in scores
            ]
            run_means = [
                statistics.fmean(scores) for scores in per_run[retriever][metric].values() if scores
            ]
            block["metrics"][metric] = {
                "overall": _stats(all_scores),
                "mem0_comparable": _stats(comparable),
                "by_question_type": {
                    qt: _stats(by_type[qt])
                    for qt in sorted(
                        by_type,
                        key=lambda t: CATEGORY_ORDER.index(t) if t in CATEGORY_ORDER else 99,
                    )
                },
                "run_means": run_means,
                "run_std": statistics.pstdev(run_means) if len(run_means) > 1 else 0.0,
                "unscored": unscored.get(f"{retriever}/{metric}", 0),
                "by_conversation": {
                    str(conv): _stats(per_conv[retriever][conv][metric])
                    for conv in sorted(per_conv[retriever])
                },
            }
        summary["retrievers"][retriever] = block

    return summary


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# LoCoMo results",
        "",
        f"Run dir: `{summary['run_dir']}`  ",
        f"Conversations: {summary['conversations']}  ",
        "",
    ]
    for retriever, block in summary["retrievers"].items():
        lines.append(f"## {retriever}")
        lines.append("")
        lines.append(
            "| metric | overall | mem0-comparable (no adversarial) | run std | unscored | answer errors |"
        )
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for metric, values in block["metrics"].items():
            overall = values["overall"] or {}
            comparable = values["mem0_comparable"] or {}
            lines.append(
                f"| {metric} | {overall.get('mean', float('nan')):.3f} (n={overall.get('n', 0)}) "
                f"| {comparable.get('mean', float('nan')):.3f} (n={comparable.get('n', 0)}) "
                f"| {values['run_std']:.3f} | {values['unscored']} | {block['answer_errors']} |"
            )
        lines.append("")
        types = sorted(
            {qt for values in block["metrics"].values() for qt in values["by_question_type"]},
            key=lambda t: CATEGORY_ORDER.index(t) if t in CATEGORY_ORDER else 99,
        )
        lines.append("| question type | n | " + " | ".join(block["metrics"]) + " |")
        lines.append("| --- | ---: | " + " | ".join("---:" for _ in block["metrics"]) + " |")
        for qt in types:
            cells = []
            n = 0
            for metric in block["metrics"]:
                stats = block["metrics"][metric]["by_question_type"].get(qt) or {}
                n = max(n, stats.get("n", 0))
                mean = stats.get("mean")
                cells.append(f"{mean:.3f}" if isinstance(mean, float) else "-")
            lines.append(f"| {qt} | {n} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def aggregate_and_write(run_dir: Path) -> dict[str, Any]:
    summary = aggregate_run_dir(run_dir)
    write_json(str(run_dir / "locomo_summary.json"), summary)
    (run_dir / "locomo_summary.md").write_text(render_markdown(summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    summary = aggregate_and_write(args.run_dir)
    print(render_markdown(summary))


if __name__ == "__main__":
    main()
