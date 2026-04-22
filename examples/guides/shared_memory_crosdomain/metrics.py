"""Aggregate ``results/*.json`` into summary tables.

Reports:
- per-arm × per-agent: correctness mean ± stddev
- per-arm × per-agent: LLM calls, memory citations, prior-context length (means)
- A3 unifies-domains rate per arm
- paired per-seed correctness deltas: SHARED − ISOLATED, SHARED − CONCAT

Usage::

    python metrics.py
    python metrics.py --results-dir path/to/results
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent
ARMS: tuple[str, ...] = ("shared", "isolated", "concat")


def _load(results_dir: Path) -> dict[str, dict[int, dict]]:
    out: dict[str, dict[int, dict]] = defaultdict(dict)
    for p in sorted(results_dir.glob("*.json")):
        arm, _, seed_str = p.stem.rpartition("_s")
        if arm not in ARMS or not seed_str.isdigit():
            continue
        out[arm][int(seed_str)] = json.loads(p.read_text())
    return dict(out)


def _stats(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), 0.0
    return statistics.fmean(values), statistics.stdev(values)


def _fmt(mean: float, stddev: float) -> str:
    return f"{mean:5.2f} ± {stddev:4.2f}"


def _correctness_table(runs: dict[str, dict[int, dict]]) -> str:
    lines = ["Correctness (0-8) — mean ± stddev across seeds",
             "  arm        A1             A2             A3            n"]
    for arm in ARMS:
        seeds = runs.get(arm, {})
        if not seeds:
            continue
        a = [[r["agents"][i]["correctness"] for r in seeds.values()] for i in range(3)]
        lines.append(
            f"  {arm:9}  {_fmt(*_stats(a[0]))}   {_fmt(*_stats(a[1]))}   "
            f"{_fmt(*_stats(a[2]))}   {len(seeds)}"
        )
    return "\n".join(lines)


def _secondary_table(runs: dict[str, dict[int, dict]]) -> str:
    lines = ["", "Secondary metrics — mean across seeds"]
    lines.append(f"  arm        {'metric':26}  A1       A2       A3")
    fields = [
        ("LLM calls", "n_llm_calls"),
        ("mem cites (findings)", "memory_citations_in_findings"),
        ("mem cites (final)", "memory_citations_in_final"),
        ("prior-context chars", "prior_context_chars"),
    ]
    for arm in ARMS:
        seeds = runs.get(arm, {})
        if not seeds:
            continue
        for label, fld in fields:
            col = [
                statistics.fmean(r["agents"][i][fld] for r in seeds.values())
                for i in range(3)
            ]
            lines.append(
                f"  {arm:9}  {label:26}  {col[0]:7.2f}  {col[1]:7.2f}  {col[2]:7.2f}"
            )
        lines.append("")
    return "\n".join(lines).rstrip()


def _abstraction_table(runs: dict[str, dict[int, dict]]) -> str:
    lines = ["", "A3 unifies-domains rate (final answer)"]
    for arm in ARMS:
        seeds = runs.get(arm, {})
        if not seeds:
            continue
        hits = sum(1 for r in seeds.values() if r["a3_unifies_domains"])
        lines.append(f"  {arm:9}  {hits}/{len(seeds)}  ({100 * hits / len(seeds):.0f}%)")
    return "\n".join(lines)


def _paired_table(runs: dict[str, dict[int, dict]]) -> str:
    shared = runs.get("shared", {})
    if not shared:
        return "\n(paired deltas: no SHARED runs)"
    lines = ["", "Paired correctness deltas — SHARED minus baseline (same seed)"]
    for other in ("isolated", "concat"):
        common = sorted(set(shared) & set(runs.get(other, {})))
        if not common:
            continue
        deltas = [[] for _ in range(3)]
        for s in common:
            for i in range(3):
                deltas[i].append(
                    shared[s]["agents"][i]["correctness"]
                    - runs[other][s]["agents"][i]["correctness"]
                )
        lines.append(
            f"  SHARED − {other.upper():8}  A1 {_fmt(*_stats(deltas[0]))}  "
            f"A2 {_fmt(*_stats(deltas[1]))}  A3 {_fmt(*_stats(deltas[2]))}  "
            f"(n={len(common)})"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=str(DEMO_DIR / "results"))
    args = parser.parse_args()

    runs = _load(Path(args.results_dir))
    if not runs:
        print(f"No result JSONs found under {args.results_dir}")
        return 1

    print(_correctness_table(runs))
    print(_secondary_table(runs))
    print(_abstraction_table(runs))
    print(_paired_table(runs))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
