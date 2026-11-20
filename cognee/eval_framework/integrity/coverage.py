from typing import Any

from pydantic import BaseModel, computed_field


class MetricCoverage(BaseModel):
    """How many of the questions a metric actually scored.

    Means and confidence intervals are computed over scored entries only, so a metric with
    partial coverage describes a different population than the run's question set.
    """

    scored: int
    total: int

    @computed_field
    @property
    def unscored(self) -> int:
        return self.total - self.scored

    @computed_field
    @property
    def unscored_fraction(self) -> float:
        return self.unscored / self.total if self.total else 0.0


def compute_metric_coverage(entries: list[dict[str, Any]]) -> dict[str, MetricCoverage]:
    """Per-metric coverage over a metrics file.

    Every entry is part of every metric's population: a ``None`` score and an entry whose answer
    failed (it was never sent to the evaluator) are both unscored. 50 failed answers out of 100
    is ``scored 50 / total 100``, not 50/50.
    """
    names = {metric for entry in entries for metric in entry["metrics"]}
    return {
        metric: MetricCoverage(
            scored=sum(
                1
                for entry in entries
                if (entry["metrics"].get(metric) or {}).get("score") is not None
            ),
            total=len(entries),
        )
        for metric in names
    }
