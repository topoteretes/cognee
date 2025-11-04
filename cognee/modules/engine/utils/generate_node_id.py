from uuid import NAMESPACE_OID, UUID, uuid5


def generate_node_id(node_id: str) -> UUID:
    return uuid5(NAMESPACE_OID, node_id.lower().replace(" ", "_").replace("'", ""))
