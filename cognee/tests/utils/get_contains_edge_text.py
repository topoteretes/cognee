def get_contains_edge_text(entity_name: str, entity_description: str) -> str:
    description = (entity_description or "").strip()
    if description:
        return f"Document chunk mentions {entity_name}: {description}"

    return f"Document chunk mentions {entity_name}."
