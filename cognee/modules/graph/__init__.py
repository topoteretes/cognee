"""Graph-side helpers that sit above the graph database adapter.

``cognee_graph/`` -- ``CogneeGraph``, the in-memory graph (``Node``/``Edge``)
retrievers project database results into; ``methods/`` -- graph reads and
deletes (e.g. chunk-scoped orphan deletion for incremental update);
``utils/`` -- converting ``DataPoint`` trees to nodes/edges and back,
transparent-node unwrapping, deduplication; ``models/`` -- pydantic graph
models; ``rdf/`` -- RDF export of the graph; ``legacy/`` -- the pre-source-ref
relationship ledger, read only so old data can still be deleted.
"""
