"""Test for vector database configuration validation."""

import os
from unittest.mock import patch

import pytest

from cognee.api.v1.config.config import config
from cognee.api.v1.exceptions.exceptions import InvalidConfigAttributeError
from cognee.infrastructure.databases.vector.config import VectorConfig


def test_set_vector_db_config_invalid_attribute_raises():
    """Ensure invalid vector DB config attributes raise an error."""
    with pytest.raises(InvalidConfigAttributeError):
        config.set_vector_db_config({"invalid_attribute": "should_fail"})


def test_vector_db_provider_postgres_aliases_to_pgvector():
    """VECTOR_DB_PROVIDER=postgres is a user-facing alias of pgvector."""
    with patch.dict(os.environ, {"VECTOR_DB_PROVIDER": "postgres"}):
        cfg = VectorConfig()

    assert cfg.vector_db_provider == "pgvector"
    assert cfg.vector_dataset_database_handler == "pgvector"
