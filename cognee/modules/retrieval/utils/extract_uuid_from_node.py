from typing import Any, Optional
from uuid import UUID


def extract_uuid_from_node(node: Any) -> Optional[UUID]:
    """
    Try to pull a UUID string out of node.id or node.properties['id'],
    then return a UUID instance (or None if neither exists).
    """
    id_str = None
    if not id_str:
        id_str = getattr(node, "id", None)

    if hasattr(node, "attributes") and not id_str:
        id_str = node.attributes.get("id", None)

    id = UUID(id_str) if isinstance(id_str, str) else None
    return id
