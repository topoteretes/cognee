from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel

from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge


class DecompositionMode(str, Enum):
    """Supported decomposition execution modes."""

    ANSWER_PER_SUBQUERY = "answer_per_subquery"
    COMBINED_TRIPLETS_CONTEXT = "combined_triplets_context"


class QueryDecomposition(BaseModel):
    """Structured decomposition output."""

    subqueries: list[str]


@dataclass
class SubqueryRunState:
    """State collected for one decomposed subquery."""

    query: str
    edges: list[Edge] = field(default_factory=list)
    context: str = ""
    answer: str = ""


@dataclass
class DecompositionRunState:
    """State collected for one decomposed retrieval run."""

    original_query: str
    subqueries: list[SubqueryRunState] = field(default_factory=list)
    merged_edges: list[Edge] = field(default_factory=list)
    final_context: str | None = None


def normalize_subqueries(original_query: str, subqueries: list[str] | None) -> list[str]:
    """Clean and bound decomposed subqueries."""

    normalized_queries: list[str] = []
    for subquery in subqueries or []:
        cleaned_query = subquery.strip()
        if not cleaned_query or cleaned_query in normalized_queries:
            continue
        normalized_queries.append(cleaned_query)
        if len(normalized_queries) >= 5:
            break

    if normalized_queries:
        return normalized_queries

    fallback_query = original_query.strip()
    return [fallback_query or original_query]


def merge_deduplicated_edges(edge_batches: list[list[Edge]]) -> list[Edge]:
    """Merge edge batches using identity-based deduplication."""

    merged_edges: list[Edge] = []
    seen_ids: set[int] = set()
    for edge_batch in edge_batches:
        for edge in edge_batch:
            edge_id = id(edge)
            if edge_id in seen_ids:
                continue
            merged_edges.append(edge)
            seen_ids.add(edge_id)
    return merged_edges


def build_subquery_answer_context(state: DecompositionRunState) -> str:
    """Build the final context from ordered subquery answers."""

    sections: list[str] = []
    for index, subquery_state in enumerate(state.subqueries, start=1):
        sections.append(
            f"Subquery {index}: {subquery_state.query}\n"
            f"Subquery {index} Answer:\n{subquery_state.answer}"
        )

    return "Question decomposition results:\n\n" + "\n\n".join(sections)
