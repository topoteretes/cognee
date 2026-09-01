"""Minimal subprocess-side machinery for running native DB clients (kuzu,
lancedb) in a dedicated child process without importing cognee.

This package must stay free of cognee imports so that a spawned worker has
a small memory footprint (just the native DB library + stdlib).
"""

from ._windows_openssl import ensure_ladybug_openssl

# Runs before anything in this package imports ``ladybug``, in the parent
# process and in every spawned worker alike: ``multiprocessing`` spawn
# re-imports this package in the child, and a child gets its own DLL search
# path. No-op off Windows.
ensure_ladybug_openssl()
