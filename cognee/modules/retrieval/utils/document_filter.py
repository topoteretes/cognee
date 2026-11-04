"""The document-name filter a chunk search passes to the vector store."""


def document_filter(document_name: list[str] | None) -> dict:
    """Keyword arguments for ``VectorDBInterface.search``: ``document_name`` only when set.

    Passed only when a filter is asked for, so community vector adapters written
    against the interface before the parameter existed keep serving unfiltered
    chunk searches. A filtered search on such an adapter fails loudly instead.
    """
    return {"document_name": document_name} if document_name else {}
