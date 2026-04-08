from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel

from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge


class DecompositionMode(str, Enum):
    ANSWER_PER_SUBQUERY = "answer_per_subquery"
    COMBINED_TRIPLETS_CONTEXT = "combined_triplets_context"


class QueryDecomposition(BaseModel):
    subqueries: List[str]


@dataclass
class SubqueryRunState:
    query: str
    edges: List[Edge] = field(default_factory=list)
    context: str = ""
    answer: str = ""


@dataclass
class DecompositionRunState:
    original_query: str
    subqueries: List[SubqueryRunState] = field(default_factory=list)
    merged_edges: List[Edge] = field(default_factory=list)
    final_context: Optional[str] = None


def normalize_subqueries(original_query: str, subqueries: Optional[List[str]]) -> List[str]:
    normalized_queries: List[str] = []
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


def merge_deduplicated_edges(edge_batches: List[List[Edge]]) -> List[Edge]:
    merged_edges: List[Edge] = []
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
    sections: List[str] = []
    for index, subquery_state in enumerate(state.subqueries, start=1):
        sections.append(
            f"Subquery {index}: {subquery_state.query}\n"
            f"Subquery {index} Answer:\n{subquery_state.answer}"
        )

    return "Question decomposition results:\n\n" + "\n\n".join(sections)
