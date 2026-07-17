import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

INPUT_COGNEE = Path("evals/benchmark_summary_cognee.json")
INPUT_COMPETITION = Path("evals/benchmark_summary_competition.json")

OUT_OPTIMISED = "evals/optimized_cognee_configurations.png"
OUT_COMP = "evals/comprehensive_metrics_comparison.png"

# Metric id ➜ bar colour (keep in same order for legend)
METRIC_KEYS = {
    "Human-like Correctness": "#4ade80",  # Green
    "DeepEval Correctness": "#818cf8",  # Indigo
    "DeepEval F1": "#c084fc",  # Light purple
    "DeepEval EM": "#6b7280",  # Grey
}

Y_LIM = (0.0, 1.05)  # applies to all charts


def _load(path: Path) -> List[Dict[str, Any]]:
    """Read JSON file that may be either a list or dict{'data': …}."""
    with path.open() as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], list):
        return obj["data"]
    raise ValueError(f"Unsupported format in {path}")


def _extract_matrix(records: List[Dict[str, Any]]):
    """
    Return:
        systems         -> list[str]
        means           -> dict[metric] = array(len(systems))
        error_minus     -> dict[metric] = array(len(systems))
        error_plus      -> dict[metric] = array(len(systems))
    Any missing value is filled with 0.
    """
    systems = [r["system"] for r in records]
    means, err_m, err_p = {}, {}, {}

    for metric in METRIC_KEYS:
        m, e_m, e_p = [], [], []
        for r in records:
            mean = r.get(metric, 0.0)
            low, high = r.get(f"{metric} Error", [mean, mean])
            m.append(mean)
            e_m.append(mean - low)
            e_p.append(high - mean)
        means[metric] = np.asarray(m)
        err_m[metric] = np.asarray(e_m)
        err_p[metric] = np.asarray(e_p)

    return systems, means, err_m, err_p


def _plot_grouped_bar(
    systems: List[str],
    means: Dict[str, np.ndarray],
    err_m: Dict[str, np.ndarray],
    err_p: Dict[str, np.ndarray],
    title: str,
    outfile: str,
    rotate_xticks: bool = False,
) -> None:
    n_metrics = len(METRIC_KEYS)
    ind = np.arange(len(systems))
    width = 0.8 / n_metrics

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    ax.set_ylim(*Y_LIM)
    ax.set_title(title, fontsize=16, fontweight="bold", pad=15)
    ax.set_ylabel("Score")
    ax.set_xticks(ind)
    ha = "right" if rotate_xticks else "center"
    ax.set_xticklabels(
        systems,
        rotation=15 if rotate_xticks else 0,
        ha=ha,
    )

    for i, (metric, colour) in enumerate(METRIC_KEYS.items()):
        offset = ind + (i - (n_metrics - 1) / 2) * width
        ax.bar(
            offset,
            means[metric],
            width,
            label=metric,
            color=colour,
            yerr=[err_m[metric], err_p[metric]],
            capsize=4,
            ecolor="#374151",
        )

        # value labels
        for x, y in zip(offset, means[metric]):
            if y > 0:
                ax.text(x, y + 0.02, f"{y:.2f}", ha="center", va="bottom", fontsize=8)

    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outfile)
    plt.close(fig)


def main() -> None:
    # Allow overriding the default locations via CLI arguments
    cognee_file = Path(sys.argv[1]) if len(sys.argv) > 1 else INPUT_COGNEE
    comp_file = Path(sys.argv[2]) if len(sys.argv) > 2 else INPUT_COMPETITION

    if not cognee_file.exists():
        raise FileNotFoundError(f"{cognee_file} not found")
    if not comp_file.exists():
        raise FileNotFoundError(f"{comp_file} not found")

    # Optimised Cognee configurations
    cfg_records = _load(cognee_file)
    systems, means, err_m, err_p = _extract_matrix(cfg_records)
    _plot_grouped_bar(
        systems,
        means,
        err_m,
        err_p,
        title="Optimized Cognee Configurations",
        outfile=OUT_OPTIMISED,
        rotate_xticks=True,
    )
    print(f"Wrote {OUT_OPTIMISED}")

    # Cognee vs. competition
    comp_records = _load(comp_file)
    for record in comp_records:
        if record.get("system") == "Graphiti":
            record["system"] = "Graphiti (Previous Results)"
    systems, means, err_m, err_p = _extract_matrix(comp_records)
    _plot_grouped_bar(
        systems,
        means,
        err_m,
        err_p,
        title="Comprehensive Metrics Comparison",
        outfile=OUT_COMP,
    )
    print(f"Wrote {OUT_COMP}")


if __name__ == "__main__":
    main()
