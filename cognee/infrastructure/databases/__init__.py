"""Database backends and the interfaces they implement.

* ``graph/`` -- ``GraphDBInterface`` and adapters: ``ladybug`` (Kuzu, default,
  embedded), ``neo4j_driver``, ``neptune_driver``, ``turso``, ``postgres_demo``
  (demo only). Entry point ``get_graph_engine()``.
* ``vector/`` -- ``VectorDBInterface`` and adapters: ``lancedb`` (default),
  ``pgvector``, ``neptune_analytics``, ``turso``; embedding engines under
  ``vector/embeddings``. Entry point ``get_vector_engine_async()``.
* ``relational/`` -- SQLAlchemy over SQLite (default) or Postgres; holds users,
  ACLs, datasets, pipeline runs, search history. Always shared, never
  isolated per dataset. Entry point ``get_relational_engine()``.
* ``cache/`` -- ``CacheDBInterface`` session-cache backends: ``sql`` (SQLite
  default / Postgres), ``redis``, ``fs``, ``tapes``.
* ``dataset_database_handler/`` -- per-user+dataset database provisioning for
  ``ENABLE_BACKEND_ACCESS_CONTROL``; ``supported_dataset_database_handlers.py``
  is the support matrix. ``dataset_queue/`` caps concurrent embedded engines.
* ``hybrid/``, ``unified/`` -- backends that serve more than one role;
  ``provenance/`` -- graph source-ref bookkeeping; ``utils/`` -- engine caches.

Community adapters (ChromaDB, Qdrant, ...) register through
``use_vector_adapter`` / ``use_graph_adapter`` from the cognee-community repo.
"""
