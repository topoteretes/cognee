"""Postgres graph backend (graph-as-tables).

EXPERIMENTAL: Using Postgres as a graph store is currently experimental and is not
production-ready. Use it to experiment with keeping relational metadata, PGVector, and graph
state in a single Postgres service, but rely on a graph-native backend such as Kuzu or Neo4j
for production workloads.

Interested in further development or production use of Postgres as a graph database? Write to
us at social@cognee.ai to explore the options.
"""
