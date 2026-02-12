from cognee.modules.retrieval.exceptions.exceptions import QueryValidationError


def validate_queries(query, query_batch) -> tuple[bool, str]:
    if query is not None and query_batch is not None:
        return False, "Cannot provide both 'query' and 'query_batch'; use exactly one."
    if query is None and query_batch is None:
        return False, "Must provide either 'query' or 'query_batch'."
    if query is not None and (not query or not isinstance(query, str)):
        return False, "The query must be a non-empty string."
    if query_batch is not None:
        if not isinstance(query_batch, list) or not query_batch:
            return False, "query_batch must be a non-empty list of strings."
        if not all(isinstance(q, str) and q for q in query_batch):
            return False, "All items in query_batch must be non-empty strings."

    return True, ""


def validate_retriever_input(query, query_batch, session_cache=False):
    """Validates query inputs and session cache compatibility."""
    if query_batch and session_cache:
        raise QueryValidationError(
            message="You cannot use batch queries with session cache currently."
        )
    is_valid, msg = validate_queries(query, query_batch)
    if not is_valid:
        raise QueryValidationError(message=msg)
