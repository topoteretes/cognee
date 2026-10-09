"""Result shapes specific to hybrid retrieval.

The readers shared with every other retriever live in
``cognee.modules.retrieval.utils.results``.
"""


def empty_hybrid_result() -> dict:
    return {"chunks": [], "chunk_summaries": {}, "entities": [], "facts": []}


def payload_matches_node_filter(
    result_payload: dict,
    node_name: list[str] | None,
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
