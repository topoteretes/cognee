"""Concrete data migrations — one module per migration.

The chain registry (registry.py) wires these into ONE ordered revision
chain; a module here may touch every store (graph, vector, relational
ledger).
"""
