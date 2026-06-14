from typing import Any, Optional
from uuid import UUID


def payload(result: Any) -> dict:
    if isinstance(result, dict):
        return result
    result_payload = getattr(result, "payload", None)
    return result_payload if isinstance(result_payload, dict) else {}


def display_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool, UUID)):
        text = str(value).strip()
        return text or None
    return None


def result_id(result: Any) -> Optional[str]:
    result_payload = payload(result)
    return display_value(result_payload.get("id")) or display_value(getattr(result, "id", None))


def scored_payload(item: Any) -> tuple[Any, float]:
    if not isinstance(item, (list, tuple)) or len(item) != 2:
        return item, 0.0
    item_payload, score = item
    if not isinstance(score, (int, float)):
        return item_payload, 0.0
    return item_payload, float(score)


def payload_matches_node_filter(
    result_payload: dict,
    node_name: Optional[list[str]],
    node_name_filter_operator: str,
) -> bool:
    if not node_name:
        return True

    belongs_to_set = result_payload.get("belongs_to_set")
    if not isinstance(belongs_to_set, list):
        return False

    payload_sets = {str(name) for name in belongs_to_set}
    requested_sets = {str(name) for name in node_name}
    if node_name_filter_operator == "AND":
        return requested_sets.issubset(payload_sets)
    return bool(payload_sets & requested_sets)


def first_display_value(*values: Any) -> Optional[str]:
    for value in values:
        text = display_value(value)
        if text:
            return text
    return None
