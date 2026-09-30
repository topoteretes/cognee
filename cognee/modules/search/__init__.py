"""Search dispatch: from a ``SearchType`` to results across datasets.

``methods/search.py`` authorizes datasets, fans the query out per dataset
under the right database context, calls ``get_retriever_output`` and logs
search history. ``methods/get_search_type_retriever_instance.py`` is the
``SearchType`` -> retriever registry. ``types/`` holds ``SearchType``,
and ``SearchResult``; ``operations/`` the FEELING_LUCKY
type selector and history logging. Retriever classes live in
``cognee.modules.retrieval``; ``cognee.api.v1.search`` is the public wrapper.
"""
