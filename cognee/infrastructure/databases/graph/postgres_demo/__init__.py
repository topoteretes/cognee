"""Postgres demo graph backend (graph-as-tables).

Selected with ``GRAPH_DATABASE_PROVIDER=postgres_demo``. The older value
``postgres`` resolves here too, so existing deployments keep working.

DEMO: Using Postgres as a graph store is currently a demo feature and is not
production-ready. Use it to demo keeping relational metadata, PGVector, and graph
state in a single Postgres service, but rely on a graph-native backend such as Kuzu or Neo4j
for production workloads.

Interested in further development or production use of Postgres as a graph database? Write to
us at social@cognee.ai to explore the options.
"""
