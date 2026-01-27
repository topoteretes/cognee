def get_contains_edge_text(entity_name: str, entity_description: str) -> str:
    edge_text = "; ".join(
        [
            "relationship_name: contains",
            f"entity_name: {entity_name}",
            f"entity_description: {entity_description}",
        ]
    )
    return edge_text
