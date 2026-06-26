TRUTH_CENTROID_COLLECTION = "TruthCentroid_vector"
TRUTH_NODE_SET = ["session_learnings"]
DEFAULT_K = 8


def truth_session_node_set(session_id: str) -> str:
    return f"session_learnings:{session_id}"
