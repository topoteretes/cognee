from typing import List, Tuple

from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge


def _triplet_key(triplet: Edge) -> Tuple:
    """Semantic identity of a triplet: (source node id, relationship, target node id).

    Every retrieval round re-projects the graph into fresh Edge instances, so
    the same logical triplet never keeps its object identity across rounds —
    identity must be derived from the node ids and the relationship instead.
    Objects that don't expose the triplet structure fall back to object
    identity so distinct objects stay distinct.
    """
    attributes = getattr(triplet, "attributes", None) or {}
    relationship = attributes.get("relationship_name") or attributes.get("relationship_type")
    source_id = getattr(getattr(triplet, "node1", None), "id", None)
    target_id = getattr(getattr(triplet, "node2", None), "id", None)

    if source_id is None and target_id is None and relationship is None:
        return ("__object__", id(triplet))
    return (source_id, relationship, target_id)


class QueryState:
    """Tracks per-query state across retrieval rounds."""

    def __init__(self, triplets: List[Edge] = None, context_text: str = ""):
        self.triplets = triplets or []
        self.context_text = context_text
        self.completion = None
        self.done = False

    def merge_triplets(self, new_triplets: List[Edge]):
        """Merge new triplets with existing ones, deduplicating by semantic identity."""
        seen_keys = {_triplet_key(triplet) for triplet in self.triplets}
        for triplet in new_triplets:
            key = _triplet_key(triplet)
            if key not in seen_keys:
                self.triplets.append(triplet)
                seen_keys.add(key)

    def check_convergence(self, prev_size: int):
        """Mark done if no new triplets were added."""
        if len(self.triplets) == prev_size:
            self.done = True
