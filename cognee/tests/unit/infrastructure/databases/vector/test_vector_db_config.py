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


_RESOLVER_LOGGER = "cognee.infrastructure.databases.utils.resolve_postgres_connection.logger"


def test_unknown_vector_db_env_var_warns_only_on_vector_namespace(tmp_path):
    """A typo'd VECTOR_DB_* var warns; DB_*/unrelated keys do not trip it."""
    envf = tmp_path / ".env"
    envf.write_text("VECTOR_DB_XYZ=x\nDB_NAME=n\nLLM_API_KEY=k\n")

    with patch(_RESOLVER_LOGGER) as mock_logger:
        VectorConfig(_env_file=str(envf))

    warned = [call.args[1] for call in mock_logger.warning.call_args_list]
    assert warned == ["vector_db_xyz"]
