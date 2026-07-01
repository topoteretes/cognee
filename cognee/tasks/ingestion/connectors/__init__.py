"""SaaS source connectors built on the shared DLT ingestion subsystem.

Each connector under this package exposes a ``create_<name>_source(...)``
factory returning a dlt resource/source that plugs directly into
``cognee.remember()``/``cognee.add()`` via ``resolve_dlt_sources`` — no
connector-specific wiring is needed in the core pipeline. Connectors that
ingest content-bearing items (documents, messages, pages) should pass
``dlt_content_column`` so rows get normal chunking and LLM graph extraction
instead of the relational row treatment (see ``resolve_dlt_sources.py``).
"""
