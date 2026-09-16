"""Pipeline task implementations, grouped by stage.

A task is an async function (or generator) wrapped in
``cognee.modules.pipelines.Task``. ``cognify()`` chains
``documents.classify_documents`` -> ``documents.extract_chunks_from_documents``
-> ``graph.extract_graph_from_data`` -> ``summarization.summarize_text`` ->
``storage.add_data_points``; ``improve()``/``memify()`` chain the tasks in
``memify/``. See ``README.md`` in this folder for the full index and the
runner semantics (batching, ``enriches``, ``ctx``) in
``cognee.modules.pipelines``.
"""
