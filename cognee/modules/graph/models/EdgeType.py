from cognee.infrastructure.engine import DataPoint


class EdgeType(DataPoint):
    relationship_name: str
    number_of_edges: int

    # identity_fields makes the id deterministic and namespaced by class
    # (uuid5 of "EdgeType:<normalized relationship_name>") — same mechanism as
    # Entity/EntityType. EdgeType.id_for(text) is the single way to compute a
    # point id for lookups (retrieval joins, delete flows, adapters).
    metadata: dict = {
        "index_fields": ["relationship_name"],
        "identity_fields": ["relationship_name"],
    }
