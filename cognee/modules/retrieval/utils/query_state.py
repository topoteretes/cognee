from typing import List
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge
from cognee.modules.retrieval.utils.ids import triplet_key


class QueryState:
    """Tracks per-query state across retrieval rounds."""

    def __init__(self, triplets: List[Edge] = None, context_text: str = ""):
        self.triplets = triplets or []
        self.context_text = context_text
        self.completion = None
        self.done = False

    def merge_triplets(self, new_triplets: List[Edge]):
        """Merge new triplets with existing ones, deduplicating by triplet value."""
        seen_ids = {triplet_key(t) for t in self.triplets}
        for t in new_triplets:
            key = triplet_key(t)
            if key not in seen_ids:
                self.triplets.append(t)
                seen_ids.add(key)

    def check_convergence(self, prev_size: int):
        """Mark done if no new triplets were added."""
        if len(self.triplets) == prev_size:
            self.done = True
