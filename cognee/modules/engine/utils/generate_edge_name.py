def generate_edge_name(name: str) -> str:
    return name.lower().replace(" ", "_").replace("'", "")
