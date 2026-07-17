"""Contradiction detection for structural dedup (issue #3630, Approach D).

Two structurally-matched nodes may still disagree on an attribute value —
e.g. both "Apple" and "Apple Inc." connect to "iPhone" via the same
`founded_year` relationship, but with conflicting values (1976 vs 1977).
That is a contradiction, not a simple duplicate: we must record and resolve
it, never silently overwrite.

Resolution policy: most-recent `created_at` wins. The losing value is never
discarded — it's preserved in the merge audit record so the decision is
reversible and explainable.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class Contradiction:
    """A single conflicting-attribute finding between two candidate nodes."""

    field: str
    value_a: Any
    value_b: Any
    winner: Any
    winner_source: str  # "a" or "b"
    reason: str  # e.g. "recency"


def detect_contradictions(
    edges_a: List[Dict[str, Any]],
    edges_b: List[Dict[str, Any]],
) -> List[Contradiction]:
    """Detect contradicting edge attributes between two candidate nodes.

    Parameters
    ----------
    edges_a, edges_b : List[Dict[str, Any]]
        Each edge dict is expected to carry at minimum:
        ``relationship_name``, ``destination_node_id``, ``attributes``,
        and ``created_at`` (epoch ms int, matching DataPoint.created_at).

    Returns
    -------
    List[Contradiction]
        One entry per detected conflicting attribute. Empty if no
        contradictions are found (structural duplicate, no conflict).
    """
    contradictions: List[Contradiction] = []

    for edge_a in edges_a:
        for edge_b in edges_b:
            same_relationship = (
                edge_a.get("relationship_name") == edge_b.get("relationship_name")
            )
            same_target = (
                edge_a.get("destination_node_id") == edge_b.get("destination_node_id")
            )
            if not (same_relationship and same_target):
                continue

            attrs_a = edge_a.get("attributes") or {}
            attrs_b = edge_b.get("attributes") or {}

            if attrs_a == attrs_b:
                continue  # identical, not a contradiction

            # Determine recency winner. Missing created_at treated as 0 (oldest).
            created_a = edge_a.get("created_at", 0) or 0
            created_b = edge_b.get("created_at", 0) or 0

            if created_b >= created_a:
                winner_value, winner_source = attrs_b, "b"
            else:
                winner_value, winner_source = attrs_a, "a"

            contradictions.append(
                Contradiction(
                    field=edge_a.get("relationship_name", ""),
                    value_a=attrs_a,
                    value_b=attrs_b,
                    winner=winner_value,
                    winner_source=winner_source,
                    reason="recency",
                )
            )

    return contradictions