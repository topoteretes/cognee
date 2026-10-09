"""Adapters to the outside world: databases, LLMs, embeddings, files, storage.

Nothing here knows about cognee's domain flow; it implements interfaces the
``cognee.modules`` layer calls.

* ``databases/`` -- graph, vector, relational and cache backends behind
  ``GraphDBInterface`` / ``VectorDBInterface`` / ``CacheDBInterface``, plus the
  per-dataset database handlers that give multi-tenant isolation. Get engines
  via ``get_graph_engine()``, ``get_vector_engine_async()``,
  ``get_relational_engine()`` -- never instantiate adapters directly.
* ``llm/`` -- ``LLMGateway`` (structured output, transcription, prompts) over
  litellm; ``STRUCTURED_OUTPUT_FRAMEWORK`` selects litellm-native (default),
  instructor (legacy) or BAML.
* ``engine/`` -- the ``DataPoint`` base class and ``Edge`` model every graph
  node derives from.
* ``loaders/`` -- file-type loaders behind ``LoaderInterface``; register new
  ones in ``supported_loaders.py``.
* ``files/`` -- local and S3 storage, file utilities; ``data/chunking`` --
  chunking config.
* ``session/`` -- ``SessionManager`` over the cache backend (session memory).
* ``locks/`` -- per-dataset and per-session locks; ``context/`` and
  ``entities/`` -- the ``BaseContextProvider`` / ``BaseEntityExtractor`` base
  classes retrievers and extractors plug into; ``utils/`` -- shared helpers.

Configuration for each area is a pydantic-settings class named ``*Config`` in
the area's ``config.py`` (``get_llm_config()``, ``get_graph_config()``, ...),
read from environment variables.
"""
