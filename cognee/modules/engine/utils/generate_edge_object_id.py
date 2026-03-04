from uuid import NAMESPACE_OID, uuid5


def generate_edge_object_id(
    source_node_id: str, target_node_id: str, relationship_name: str
) -> str:
    """
    Deterministic, stable id for a single edge (source, target, relationship).
    Used for feedback mapping and stored on edges in the graph DB as edge_object_id.
    """
    edge_specific_identifier = (
        (str(source_node_id) + relationship_name + str(target_node_id))
        .lower()
        .replace(" ", "_")
        .replace("'", "")
    )
    return str(uuid5(NAMESPACE_OID, edge_specific_identifier))
