"""Prepare edges for graph storage by ensuring each has a stable edge_object_id."""

from typing import Any, Dict, List, Tuple

from cognee.modules.engine.utils import generate_edge_object_id


<<<<<<< feature/cog-4069-memify-pipeline-apply-feedback-weights-to-graph-elements
def ensure_default_edge_properties(
    edges: List[Tuple[str, str, str, Dict[str, Any]]],
) -> List[Tuple[str, str, str, Dict[str, Any]]]:
    """
    Ensure each edge has all the default properties (edge_object_id, feedback_weight).
=======
def ensure_edge_object_ids(
    edges: List[Tuple[str, str, str, Dict[str, Any]]],
) -> List[Tuple[str, str, str, Dict[str, Any]]]:
    """
    Ensure each edge has edge_object_id in its properties dict.
>>>>>>> dev
    Returns a new list of edges; does not mutate the input.
    """
    result = []
    for source_id, target_id, relationship_name, properties in edges:
        props = dict(properties) if properties else {}
        if "edge_object_id" not in props:
            props["edge_object_id"] = generate_edge_object_id(
                source_id, target_id, relationship_name
            )
<<<<<<< feature/cog-4069-memify-pipeline-apply-feedback-weights-to-graph-elements
        if "feedback_weight" not in props:
            props["feedback_weight"] = 0.5
=======
>>>>>>> dev
        result.append((source_id, target_id, relationship_name, props))
    return result
