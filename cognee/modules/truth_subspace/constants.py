from cognee.modules.improve.constants import SESSION_LEARNINGS_NODE_SET

TRUTH_CENTROID_COLLECTION = "TruthCentroid_vector"
TRUTH_NODE_SET = [SESSION_LEARNINGS_NODE_SET]
DEFAULT_K = 8


def truth_session_node_set(session_id: str) -> str:
    return f"{SESSION_LEARNINGS_NODE_SET}:{session_id}"
