"""Aggregate LoCoMo LLM-judge accuracy across conversations and repeated runs.

Reads every ``qa/locomo_metrics_conv<i>_<retriever>_run<r>.json`` under a run directory and
produces, per retriever (pooled over all conversations):

- the total, which leaves out open-domain questions (they are not the focus of the evaluation and
  appear only per category),
- per-category and per-conversation means, per-run means and their std, and a descriptive pooled
  bootstrap CI,
- with ``--gold-exclusions``, the same sets and categories again without the questions whose gold
  answer an audit marked as corrupted (one block per variant of the exclusions file).

Writes ``locomo_summary.json`` and ``locomo_summary.md`` into the run directory. With gold
exclusions the Markdown reports only the numbers without the corrupted questions; the JSON keeps
the numbers on all questions too.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from cognee.eval_framework.analysis.metrics_calculator import bootstrap_ci
from cognee.eval_framework.reporting.io import read_json, write_json

METRICS = ("llm_judge",)
CATEGORY_ORDER = ("single_hop", "multi_hop", "temporal", "open_domain", "adversarial")
SLICES = {"all_except_open_domain": lambda question_type: question_type != "open_domain"}
SLICE_LABELS = {"all_except_open_domain": "total (open-domain left out)"}
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


def load_gold_exclusions(path: Path) -> dict[str, set[tuple[int, int]]]:
    """``{variant: {(conversation_index, question_idx), ...}}`` from a gold-exclusions file.

    The file lists question sets under ``sets.<name>.questions`` and, under ``variants``, which
    sets each variant removes together.
    """
    document = json.loads(path.read_text(encoding="utf-8"))
    sets = {
        name: {(int(q["conversation_index"]), int(q["question_idx"])) for q in block["questions"]}
        for name, block in document["sets"].items()
    }
    return {
        variant: set().union(*(sets[name] for name in names))
        for variant, names in document["variants"].items()
    }


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


def _type_order(question_type: str) -> int:
    return CATEGORY_ORDER.index(question_type) if question_type in CATEGORY_ORDER else 99


def _sets_and_types(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    scored = [(row["question_type"], row["scores"][metric]) for row in rows]
    scored = [(question_type, score) for question_type, score in scored if score is not None]
    types = sorted({question_type for question_type, _ in scored}, key=_type_order)
    return {
        "slices": {
            name: _stats([score for question_type, score in scored if keep(question_type)])
            for name, keep in SLICES.items()
        },
        "by_question_type": {
            question_type: _stats([score for qt, score in scored if qt == question_type])
            for question_type in types
        },
    }


def aggregate_run_dir(
    run_dir: Path, gold_exclusions: dict[str, set[tuple[int, int]]] | None = None
) -> dict[str, Any]:
    files = find_metric_files(run_dir)
    if not files:
        raise FileNotFoundError(f"No locomo_metrics_conv*_run*.json files under {run_dir}")

    rows_by_retriever: dict[str, list[dict[str, Any]]] = defaultdict(list)
    errors: dict[str, int] = defaultdict(int)
    conversations = set()
    for info in files:
        retriever = info["retriever"]
        conversations.add(info["conversation_index"])
        for entry in read_json(str(info["path"])):
            if isinstance(entry.get("answer"), str) and entry["answer"].startswith("ERROR:"):
                errors[retriever] += 1
            question_idx = entry.get("question_idx")
            rows_by_retriever[retriever].append(
                {
                    "conversation_index": info["conversation_index"],
                    "key": (
                        (info["conversation_index"], int(question_idx))
                        if question_idx is not None
                        else None
                    ),
                    "question_type": entry.get("question_type", "unknown"),
                    "run_idx": info["run_idx"],
                    "scores": {metric: _score(entry, metric) for metric in METRICS},
                }
            )

    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "conversations": sorted(conversations),
        "metric_files": len(files),
        "retrievers": {},
    }
    for retriever in sorted(rows_by_retriever):
        rows = rows_by_retriever[retriever]
        block: dict[str, Any] = {"metrics": {}, "answer_errors": errors.get(retriever, 0)}
        for metric in METRICS:
            per_run: dict[int, list[float]] = defaultdict(list)
            per_conv: dict[int, list[float]] = defaultdict(list)
            for row in rows:
                score = row["scores"][metric]
                if score is not None:
                    per_run[row["run_idx"]].append(score)
                    per_conv[row["conversation_index"]].append(score)
            run_means = [statistics.fmean(scores) for scores in per_run.values() if scores]
            block["metrics"][metric] = {
                **_sets_and_types(rows, metric),
                "run_means": run_means,
                "run_std": statistics.pstdev(run_means) if len(run_means) > 1 else 0.0,
                "unscored": sum(1 for row in rows if row["scores"][metric] is None),
                "by_conversation": {str(conv): _stats(per_conv[conv]) for conv in sorted(per_conv)},
            }
        if gold_exclusions:
            block["gold_variants"] = {}
            for variant, excluded in gold_exclusions.items():
                kept = [row for row in rows if row["key"] not in excluded]
                block["gold_variants"][variant] = {
                    "excluded_questions": len({row["key"] for row in rows} & excluded),
                    "metrics": {metric: _sets_and_types(kept, metric) for metric in METRICS},
                }
        summary["retrievers"][retriever] = block
    return summary


def _cell(stats: dict[str, Any] | None, with_n: bool = True) -> str:
    if not stats:
        return "-"
    return f"{stats['mean']:.3f}" + (f" (n={stats['n']})" if with_n else "")


def _render_tables(metrics: dict[str, Any]) -> list[str]:
    """LLM-judge accuracy per question set and per category."""
    judge = metrics["llm_judge"]
    lines = ["| questions | llm_judge |", "| --- | ---: |"]
    for name in SLICES:
        lines.append(f"| {SLICE_LABELS[name]} | {_cell(judge['slices'][name])} |")
    lines += ["", "| question type | llm_judge |", "| --- | ---: |"]
    for question_type in sorted(judge["by_question_type"], key=_type_order):
        lines.append(f"| {question_type} | {_cell(judge['by_question_type'][question_type])} |")
    return lines + [""]


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# LoCoMo results",
        "",
        f"Run dir: `{summary['run_dir']}`  ",
        f"Conversations: {summary['conversations']}  ",
        "",
    ]
    for retriever, block in summary["retrievers"].items():
        lines += [f"## {retriever}", ""]
        variants = block.get("gold_variants")
        if variants:
            # Report only the numbers without the questions whose gold answer is corrupted.
            for variant, values in variants.items():
                lines += [
                    (
                        f"Questions with a corrupted gold answer left out (`{variant}`): "
                        f"{values['excluded_questions']}."
                    ),
                    "",
                ]
                lines += _render_tables(values["metrics"])
        else:
            lines += _render_tables(block["metrics"])
        judge = block["metrics"]["llm_judge"]
        lines += [
            (
                f"Run std (llm_judge): {judge['run_std']:.3f}; unscored: {judge['unscored']}; "
                f"answer errors: {block['answer_errors']}."
            ),
            "",
        ]
    return "\n".join(lines)


def aggregate_and_write(run_dir: Path, gold_exclusions_path: Path | None = None) -> dict[str, Any]:
    exclusions = load_gold_exclusions(gold_exclusions_path) if gold_exclusions_path else None
    summary = aggregate_run_dir(run_dir, exclusions)
    if gold_exclusions_path:
        summary["gold_exclusions"] = str(gold_exclusions_path)
    write_json(str(run_dir / "locomo_summary.json"), summary)
    (run_dir / "locomo_summary.md").write_text(render_markdown(summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--gold-exclusions", type=Path, default=None)
    args = parser.parse_args()
    summary = aggregate_and_write(args.run_dir, args.gold_exclusions)
    print(render_markdown(summary))


if __name__ == "__main__":
    main()
