"""Minimal subprocess-side machinery for running native DB clients (kuzu,
lancedb) in a dedicated child process without importing cognee.

This package must stay free of cognee imports so that a spawned worker has
a small memory footprint (just the native DB library + stdlib).
"""
