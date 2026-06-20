"""Run retrieval-only benchmarks across Cognee search modes."""

from __future__ import annotations

from typing import Any, Optional

from cognee.eval_framework.retrieval_benchmark.metrics import (
    aggregate_metric,
    context_overlap_f1,
    context_recall,
    normalize_context_text,
)
from cognee.modules.search.types import SearchType


DEFAULT_SEARCH_TYPES = (
    SearchType.CHUNKS,
    SearchType.SUMMARIES,
    SearchType.GRAPH_COMPLETION,
)


class RetrievalBenchmarkRunner:
    """Evaluate retrieval quality without LLM answer generation."""

    def __init__(
        self,
        search_types: Optional[list[SearchType]] = None,
        top_k: int = 10,
    ):
        self.search_types = list(search_types or DEFAULT_SEARCH_TYPES)
        self.top_k = top_k

    async def run_instance(
        self,
        *,
        question: str,
        golden_context: str,
        search_fn,
        datasets: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        """Run retrieval metrics for one QA instance across configured search types."""
        results: list[dict[str, Any]] = []
        for search_type in self.search_types:
            payload = await search_fn(
                query_text=question,
                query_type=search_type,
                datasets=datasets,
                top_k=self.top_k,
                only_context=True,
            )
            retrieved_context = self._extract_context(payload)
            results.append(
                {
                    "question": question,
                    "search_type": search_type.value,
                    "context_recall": context_recall(retrieved_context, golden_context),
                    "context_f1": context_overlap_f1(retrieved_context, golden_context),
                    "retrieved_context": normalize_context_text(retrieved_context),
                    "golden_context": normalize_context_text(golden_context),
                }
            )
        return results

    async def run_instances(
        self,
        instances: list[dict[str, str]],
        search_fn,
        datasets: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Run retrieval metrics for many instances and aggregate per search type."""
        per_type: dict[str, list[dict[str, Any]]] = {
            search_type.value: [] for search_type in self.search_types
        }

        for instance in instances:
            question = instance["question"]
            golden_context = instance.get("golden_context", "")
            rows = await self.run_instance(
                question=question,
                golden_context=golden_context,
                search_fn=search_fn,
                datasets=datasets,
            )
            for row in rows:
                per_type[row["search_type"]].append(row)

        summary: dict[str, Any] = {"per_search_type": {}, "instances": []}
        for search_type, rows in per_type.items():
            summary["per_search_type"][search_type] = {
                "context_recall": aggregate_metric(row["context_recall"] for row in rows),
                "context_f1": aggregate_metric(row["context_f1"] for row in rows),
            }
            summary["instances"].extend(rows)
        return summary

    @staticmethod
    def _extract_context(payload: Any) -> Any:
        if payload is None:
            return ""
        if isinstance(payload, list) and payload:
            first = payload[0]
            if isinstance(first, dict):
                for key in ("context_result", "context", "search_result", "result"):
                    if key in first and first[key] is not None:
                        return first[key]
            return payload
        if isinstance(payload, dict):
            for key in ("context_result", "context", "search_result", "result"):
                if key in payload and payload[key] is not None:
                    return payload[key]
        return payload
