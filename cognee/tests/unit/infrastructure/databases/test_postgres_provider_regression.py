"""Regression net: the unification must not shift the non-Postgres defaults.

Proves the four DB-unification commits left the field-level defaults intact:
relational -> sqlite, vector -> lancedb (local path), graph -> ladybug.
"""

import os
from unittest.mock import patch

from cognee.infrastructure.databases.relational.config import RelationalConfig
from cognee.infrastructure.databases.vector.config import VectorConfig
from cognee.infrastructure.databases.graph.config import GraphConfig


def test_non_postgres_defaults_unchanged():
    """With no DB configuration at all, each store keeps its original default."""
    # Isolate from os.environ and the repo .env so the pydantic field defaults
    # (not the repo's kuzu/lancedb/sqlite .env values) are what gets asserted.
    with patch.dict(os.environ, {}, clear=True):
        assert RelationalConfig(_env_file=None).db_provider == "sqlite"

        vector_config = VectorConfig(_env_file=None)
        assert vector_config.vector_db_provider == "lancedb"
        assert vector_config.vector_db_url.endswith("cognee.lancedb")

        assert GraphConfig(_env_file=None).graph_database_provider == "ladybug"
