"""Test for the config.set method to verify Issue #2047 fix."""

import pytest
import inspect
from cognee.api.v1.config.config import config
from cognee.api.v1.exceptions.exceptions import InvalidConfigAttributeError


class TestConfigSetMethod:
    """Test the generic set method with @staticmethod decorator."""

    def test_set_method_exists(self):
        """Verify that the set method exists on the config class."""
        assert hasattr(config, "set")
        assert callable(config.set)

    def test_set_method_is_static(self):
        """Verify that the set method is a static method."""
        assert isinstance(inspect.getattr_static(config, "set"), staticmethod)

    def test_set_llm_provider(self):
        """Test setting LLM provider through generic set method."""
        config.set("llm_provider", "anthropic")

    def test_set_llm_model(self):
        """Test setting LLM model through generic set method."""
        config.set("llm_model", "gpt-4")

    def test_set_vector_db_provider(self):
        """Test setting vector DB provider through generic set method."""
        config.set("vector_db_provider", "chromadb")

    def test_set_graph_database_subprocess_enabled(self):
        """Test enabling graph subprocess adapters through generic set method."""
        from cognee.infrastructure.databases.graph.config import get_graph_config

        original = get_graph_config().graph_database_subprocess_enabled
        try:
            config.set("graph_database_subprocess_enabled", True)
            assert get_graph_config().graph_database_subprocess_enabled is True
        finally:
            get_graph_config().graph_database_subprocess_enabled = original

    def test_set_vector_db_subprocess_enabled(self):
        """Test enabling vector subprocess adapters through generic set method."""
        from cognee.infrastructure.databases.vector.config import get_vectordb_config

        original = get_vectordb_config().vector_db_subprocess_enabled
        try:
            config.set("vector_db_subprocess_enabled", True)
            assert get_vectordb_config().vector_db_subprocess_enabled is True
        finally:
            get_vectordb_config().vector_db_subprocess_enabled = original

    def test_set_chunk_size(self):
        """Test setting chunk size through generic set method."""
        config.set("chunk_size", 2048)

    def test_set_invalid_key_raises_error(self):
        """Test that setting an invalid key raises InvalidConfigAttributeError."""
        with pytest.raises(InvalidConfigAttributeError):
            config.set("invalid_key_that_does_not_exist", "some_value")

    def test_set_method_signature(self):
        """Verify the set method has the correct signature."""
        sig = inspect.signature(config.set)
        params = list(sig.parameters.keys())
        assert "key" in params
        assert "value" in params
        assert len(params) == 2

    def test_set_can_be_called_without_instance(self):
        """Test that set can be called as a static method without instantiation."""
        try:
            config.set("llm_provider", "openai")
        except TypeError as e:
            if "missing" in str(e).lower() or "self" in str(e).lower():
                pytest.fail("set method is not properly decorated with @staticmethod")
            raise

    def test_set_with_multiple_keys(self):
        """Test setting multiple different keys."""
        test_cases = [
            ("llm_provider", "openai"),
            ("llm_model", "gpt-5-mini"),
            ("vector_db_provider", "lancedb"),
            ("chunk_size", 1500),
            ("chunk_overlap", 100),
        ]

        for key, value in test_cases:
            config.set(key, value)

    def test_set_graph_db_config_coerces_string_values(self):
        """Bulk ``set_graph_db_config`` must coerce string payloads to the
        field's declared type — e.g. ``{"graph_database_subprocess_enabled":
        "true"}`` should land as ``bool True``, not the literal string.
        Without coercion, downstream ``if subprocess_enabled:`` evaluates
        a non-empty string as truthy regardless of its content.

        Snapshots and restores the mutated singleton fields so the test
        doesn't leak state into other tests in the same session.
        """
        from cognee.infrastructure.databases.graph.config import get_graph_config

        cfg = get_graph_config()
        original = {
            "graph_database_subprocess_enabled": cfg.graph_database_subprocess_enabled,
            "kuzu_num_threads": cfg.kuzu_num_threads,
        }
        try:
            config.set_graph_db_config(
                {
                    "graph_database_subprocess_enabled": "true",
                    "kuzu_num_threads": "4",
                }
            )
            cfg = get_graph_config()
            assert cfg.graph_database_subprocess_enabled is True
            assert cfg.kuzu_num_threads == 4

            # Native types pass through unchanged.
            config.set_graph_db_config(
                {
                    "graph_database_subprocess_enabled": False,
                    "kuzu_num_threads": 8,
                }
            )
            cfg = get_graph_config()
            assert cfg.graph_database_subprocess_enabled is False
            assert cfg.kuzu_num_threads == 8
        finally:
            config.set_graph_db_config(original)
