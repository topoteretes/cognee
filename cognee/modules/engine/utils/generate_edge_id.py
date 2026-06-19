from uuid import NAMESPACE_OID, UUID, uuid5


def generate_edge_id(edge_id: str) -> UUID:
    """Stable dedup slug for relational-ledger EDGE rows (upsert_edges only).

    NOT an EdgeType point id — those are EdgeType.id_for(text) (namespaced by
    the class, like every identity-bearing DataPoint). This bare derivation
    stays only because ledger rows store no edge text, so existing slugs can
    never be re-derived/migrated; it is an opaque key compared against nothing
    else.
    """
    return uuid5(NAMESPACE_OID, edge_id.lower().replace(" ", "_").replace("'", ""))
