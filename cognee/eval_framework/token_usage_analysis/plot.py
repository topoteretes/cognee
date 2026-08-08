"""Optional plotting: one cumulative-cost cross-over figure per llm_model.

Matplotlib is imported lazily so a JSON-only run needs no extra dependency. The
curves are rebuilt from the four numbers already in the report, so this stays a
pure consumer of the report with no cost-model knowledge.
"""

from __future__ import annotations

from pathlib import Path


def write_plots(report: dict, out_dir: Path, name: str = "report") -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return [
        _plot_crossover(name, llm_model, result, out_dir, plt)
        for llm_model, result in report.items()
    ]


def _plot_crossover(name: str, llm_model: str, result: dict, out_dir: Path, plt) -> Path:
    full_per_query = result["full_context"]["per_query_tokens"]
    ingestion = result["cognee"]["ingestion_tokens"]
    cognee_per_query = result["cognee"]["per_query_tokens"]
    parity = result["reduction_milestones"].get("1")

    max_queries = int((parity or 20) * 2)
    queries = list(range(max_queries + 1))
    full_context = [q * full_per_query for q in queries]
    cognee = [ingestion + q * cognee_per_query for q in queries]

    figure, axes = plt.subplots(figsize=(7, 4.5))
    axes.plot(queries, full_context, label="full-context")
    axes.plot(queries, cognee, label="cognee memory")
    if parity is not None:
        axes.axvline(parity, linestyle="--", color="gray", linewidth=1)
        axes.annotate(
            f"parity ≈ {parity:g} queries",
            xy=(parity, parity * full_per_query),
            xytext=(6, 6),
            textcoords="offset points",
        )
    axes.set_xlabel("queries")
    axes.set_ylabel("cumulative tokens")
    axes.set_title(f"Full-context vs. cognee — {llm_model}")
    axes.legend()

    path = out_dir / f"{name}__{_safe(llm_model)}.png"
    figure.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(figure)
    return path


def _safe(llm_model: str) -> str:
    return llm_model.replace("/", "_").replace(":", "_")
