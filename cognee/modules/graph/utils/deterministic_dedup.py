"""Deterministic (rule-based) duplicate resolution — issue #3627, Approach A.

Two entities extracted under different surface forms — "USA", "U.S.A.",
"United States" — are the same real-world thing. The existing id-based dedup
(``deduplicate_nodes_and_edges``) only collapses names that normalise to the
*same* id string (case / spaces / apostrophes, via ``generate_node_id``); it
misses punctuation, unicode and alias variants. This module normalises each
entity name to a canonical key (NFKC + diacritics + lowercase + alias map +
punctuation/whitespace collapse + leading-article strip — see
``canonicalize_entity_name``) and links every same-key duplicate to a chosen
canonical node with a ``merged_into`` edge, so the duplication is recorded in
the graph itself — provenance-preserving (the edge says what merged into what
and the canonical key why), non-destructive (both nodes and all their edges are
kept), and reversible (drop the ``merged_into`` edge to undo).

Detection is deterministic — pure string normalisation, no LLM and no
embeddings — so it stays reproducible under the mocked-LLM CI harness (#3601)
and is the cheap baseline the sibling approaches (B embeddings, C LLM,
D graph-structure, E temporal) are measured against.

It extends the existing dedup path — called from ``add_data_points`` right after
``deduplicate_nodes_and_edges`` — rather than forking a new pipeline, and is
opt-in via ``add_data_points(..., deterministic_dedup=True)``. Consuming
``merged_into`` at query time (collapsing the duplicates during retrieval) is
deliberately left to a follow-up, once the sibling approaches (A–E) are compared
on a shared evaluation fixture.
"""

from collections import defaultdict
from typing import Any, Optional

from cognee.modules.graph.utils.canonicalization import canonicalize_entity_name, load_alias_map

# Graph edges flow through the pipeline as
# (source_node_id, target_node_id, relationship_name, properties) tuples.
Edge = tuple[Any, Any, str, dict]

# The synthetic relationship that records a deterministic merge.
MERGED_INTO_RELATIONSHIP = "merged_into"


def resolve_deterministic_duplicates(
    nodes: list[Any],
    *,
    alias_map: Optional[dict[str, str]] = None,
) -> list[Edge]:
    """Link name-variant duplicate nodes in a batch with ``merged_into`` edges.

    Same-type nodes whose names canonicalise to the same key are grouped; within
    each group the longest original name is kept canonical (ties broken by the
    smaller id) and every other node emits a ``merged_into`` edge to it.

    Args:
        nodes: The batch's nodes (``DataPoint``s), after identity dedup.
        alias_map: Alias table for canonicalisation; loaded from the default
            map when omitted.

    Returns:
        New ``merged_into`` edges (duplicate -> canonical), one per detected
        duplicate, each carrying ``resolution`` and ``canonical_key`` in its
        properties. Empty when no group has a duplicate. ``nodes`` is never
        mutated.
    """
    if alias_map is None:
        alias_map = load_alias_map()

    # Group candidate nodes by (type, canonical key). Only nodes carrying a
    # non-empty name and a non-empty key can be name-deduplicated.
    groups: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for node in nodes:
        name = getattr(node, "name", None)
        if not name:
            continue
        key = canonicalize_entity_name(name, alias_map=alias_map)
        if not key:
            continue
        groups[(_node_type(node), key)].append(node)

    merge_edges: list[Edge] = []
    for (_, key), members in groups.items():
        # Keep one node per distinct id — identical-id nodes are already
        # collapsed by deduplicate_nodes_and_edges upstream.
        by_id: dict[str, Any] = {}
        for node in members:
            by_id.setdefault(str(node.id), node)
        if len(by_id) < 2:
            continue

        canonical = _choose_canonical(by_id.values())
        for node in by_id.values():
            if node.id == canonical.id:
                continue
            merge_edges.append(
                (
                    node.id,
                    canonical.id,
                    MERGED_INTO_RELATIONSHIP,
                    {"resolution": "deterministic_key", "canonical_key": key},
                )
            )
    return merge_edges


def _node_type(node: Any) -> str:
    """The node's type label used for type-gating (Entity vs EntityType vs ...)."""
    return getattr(node, "type", None) or type(node).__name__


def _choose_canonical(nodes) -> Any:
    """Keep the longest original name (most informative); ties -> smaller id."""
    return sorted(nodes, key=lambda n: (-len(str(getattr(n, "name", ""))), str(n.id)))[0]
