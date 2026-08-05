"""Tests for GraphConfig env-var handling."""

from unittest.mock import patch

from cognee.infrastructure.databases.graph.config import GraphConfig


_RESOLVER_LOGGER = "cognee.infrastructure.databases.utils.resolve_postgres_connection.logger"


def test_unknown_graph_database_env_var_warns_only_on_graph_namespace(tmp_path):
    """A typo'd GRAPH_DATABASE_* var warns; VECTOR_DB_*/unrelated keys do not trip it."""
    envf = tmp_path / ".env"
    envf.write_text("GRAPH_DATABASE_XYZ=x\nVECTOR_DB_HOST=v\nLLM_API_KEY=k\n")

    with patch(_RESOLVER_LOGGER) as mock_logger:
        GraphConfig(_env_file=str(envf))

    warned = [call.args[1] for call in mock_logger.warning.call_args_list]
    assert warned == ["graph_database_xyz"]
