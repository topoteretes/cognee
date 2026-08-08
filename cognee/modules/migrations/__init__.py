"""Version-keyed migration framework for Cognee graph and vector databases.

Migrations are tracked as an Alembic-style revision chain (see ``migration.py``)
rather than by comparing Cognee version strings, which is fragile across
dev/pre-release/local builds. Each database records the last-applied revision;
the runner walks the chain forward to head on startup.
"""
