def generate_node_name(name: str) -> str:
    return name.lower().replace(" ", "_").replace("'", "")

def generate_node_id(node_id: str) -> str:
    return node_id.lower().replace(" ", "_").replace("'", "")
