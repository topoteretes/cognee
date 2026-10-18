"""Domain logic: what cognee does, independent of which backend does it.

Each subpackage owns one concept. The ones you will reach for first:

* ``pipelines`` -- the task runner (``Task``, ``run_pipeline``); see its docstring.
* ``retrieval`` -- one retriever class per ``SearchType``; ``search`` -- the
  dispatcher that picks the retriever, fans out over datasets and logs history.
* ``graph`` -- graph-side helpers (``CogneeGraph`` in-memory graph, node/edge
  utilities, deletion of chunk-owned nodes); ``engine`` -- the built-in
  ``DataPoint`` models (Entity, EntityType, NodeSet, Skill, ...).
* ``chunking`` -- text chunkers and content-derived chunk ids; ``data`` --
  Dataset/Data relational models and their CRUD; ``ingestion`` -- dedup and
  classification helpers used by ``add()``.
* ``users`` -- users, tenants, roles, ACLs and permission checks; ``settings``
  -- runtime configuration read/write.
* ``memify`` and ``run_custom_pipeline`` -- the public enrichment/custom-run
  entry points (``improve()`` wraps ``memify``).
* ``session_lifecycle``, ``session_distillation``, ``agent_memory``, ``recall``
  -- session memory, its distillation into the graph, the ``@agent_memory``
  decorator, and recall's query router.
* ``ontology`` -- OWL/RDF resolvers for grounding extraction; ``provenance``
  -- edge evidence and the audit ledger; ``observability`` -- tracing spans.

Convention: a ``methods/`` folder holds one function per file, named after
the function; ``models/`` holds SQLAlchemy or pydantic models; ``operations/``
holds multi-step procedures. Backend adapters live in ``cognee.infrastructure``,
pipeline task implementations in ``cognee.tasks``, the public API in
``cognee.api``.
"""
