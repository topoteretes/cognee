from uuid import NAMESPACE_OID, uuid5


def generate_edge_id(edge_id: str) -> str:
    return uuid5(NAMESPACE_OID, edge_id.lower().replace(" ", "_").replace("'", ""))
