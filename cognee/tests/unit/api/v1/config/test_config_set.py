"""
Tests for the config.set() method to ensure CLI config commands work correctly.
"""

import pytest
from unittest.mock import patch, MagicMock
from cognee.api.v1.config.config import config
from cognee.api.v1.exceptions.exceptions import InvalidConfigAttributeError


class TestConfigSet:
    """Test the config.set() method for various configuration keys."""

    def test_set_llm_api_key(self):
        """Test setting LLM API key"""
        test_key = "sk-test-key-123"

        with patch("cognee.api.v1.config.config.get_llm_config") as mock_get_llm_config:
            mock_llm_config = MagicMock()
            mock_get_llm_config.return_value = mock_llm_config

            config.set("llm_api_key", test_key)

            assert mock_llm_config.llm_api_key == test_key

    def test_set_llm_provider(self):
        """Test setting LLM provider"""
        test_provider = "anthropic"

        with patch("cognee.api.v1.config.config.get_llm_config") as mock_get_llm_config:
            mock_llm_config = MagicMock()
            mock_get_llm_config.return_value = mock_llm_config

            config.set("llm_provider", test_provider)

            assert mock_llm_config.llm_provider == test_provider

    def test_set_llm_model(self):
        """Test setting LLM model"""
        test_model = "gpt-4o"

        with patch("cognee.api.v1.config.config.get_llm_config") as mock_get_llm_config:
            mock_llm_config = MagicMock()
            mock_get_llm_config.return_value = mock_llm_config

            config.set("llm_model", test_model)

            assert mock_llm_config.llm_model == test_model

    def test_set_llm_endpoint(self):
        """Test setting LLM endpoint"""
        test_endpoint = "https://api.example.com"

        with patch("cognee.api.v1.config.config.get_llm_config") as mock_get_llm_config:
            mock_llm_config = MagicMock()
            mock_get_llm_config.return_value = mock_llm_config

            config.set("llm_endpoint", test_endpoint)

            assert mock_llm_config.llm_endpoint == test_endpoint

    def test_set_graph_database_provider(self):
        """Test setting graph database provider"""
        test_provider = "neo4j"

        with patch("cognee.api.v1.config.config.get_graph_config") as mock_get_graph_config:
            mock_graph_config = MagicMock()
            mock_get_graph_config.return_value = mock_graph_config

            config.set("graph_database_provider", test_provider)

            assert mock_graph_config.graph_database_provider == test_provider

    def test_set_vector_db_provider(self):
        """Test setting vector database provider"""
        test_provider = "chromadb"

        with patch("cognee.api.v1.config.config.get_vectordb_config") as mock_get_vectordb_config:
            mock_vector_config = MagicMock()
            mock_get_vectordb_config.return_value = mock_vector_config

            config.set("vector_db_provider", test_provider)

            assert mock_vector_config.vector_db_provider == test_provider

    def test_set_vector_db_url(self):
        """Test setting vector database URL"""
        test_url = "http://localhost:8000"

        with patch("cognee.api.v1.config.config.get_vectordb_config") as mock_get_vectordb_config:
            mock_vector_config = MagicMock()
            mock_get_vectordb_config.return_value = mock_vector_config

            config.set("vector_db_url", test_url)

            assert mock_vector_config.vector_db_url == test_url

    def test_set_vector_db_key(self):
        """Test setting vector database key"""
        test_key = "test-key-123"

        with patch("cognee.api.v1.config.config.get_vectordb_config") as mock_get_vectordb_config:
            mock_vector_config = MagicMock()
            mock_get_vectordb_config.return_value = mock_vector_config

            config.set("vector_db_key", test_key)

            assert mock_vector_config.vector_db_key == test_key

    def test_set_chunk_size(self):
        """Test setting chunk size"""
        test_size = 2000

        with patch("cognee.api.v1.config.config.get_chunk_config") as mock_get_chunk_config:
            mock_chunk_config = MagicMock()
            mock_get_chunk_config.return_value = mock_chunk_config

            config.set("chunk_size", test_size)

            assert mock_chunk_config.chunk_size == test_size

    def test_set_chunk_overlap(self):
        """Test setting chunk overlap"""
        test_overlap = 20

        with patch("cognee.api.v1.config.config.get_chunk_config") as mock_get_chunk_config:
            mock_chunk_config = MagicMock()
            mock_get_chunk_config.return_value = mock_chunk_config

            config.set("chunk_overlap", test_overlap)

            assert mock_chunk_config.chunk_overlap == test_overlap

    def test_set_invalid_key(self):
        """Test that setting an invalid key raises InvalidConfigAttributeError"""
        with pytest.raises(InvalidConfigAttributeError):
            config.set("invalid_key", "some_value")

    def test_set_multiple_keys(self):
        """Test setting multiple configuration keys in sequence"""
        with patch("cognee.api.v1.config.config.get_llm_config") as mock_get_llm_config:
            mock_llm_config = MagicMock()
            mock_get_llm_config.return_value = mock_llm_config

            # Set multiple keys
            config.set("llm_api_key", "test-key")
            config.set("llm_provider", "openai")
            config.set("llm_model", "gpt-4o")

            # Verify all were set
            assert mock_llm_config.llm_api_key == "test-key"
            assert mock_llm_config.llm_provider == "openai"
            assert mock_llm_config.llm_model == "gpt-4o"

    def test_set_system_root_directory(self):
        """Test setting system root directory"""
        test_dir = "/tmp/test"

        with patch("cognee.api.v1.config.config.get_base_config") as mock_get_base_config, \
             patch("cognee.api.v1.config.config.get_relational_config") as mock_get_relational_config, \
             patch("cognee.api.v1.config.config.get_graph_config") as mock_get_graph_config, \
             patch("cognee.api.v1.config.config.get_vectordb_config") as mock_get_vectordb_config:

            mock_base_config = MagicMock()
            mock_base_config.system_root_directory = ""
            mock_get_base_config.return_value = mock_base_config

            mock_relational_config = MagicMock()
            mock_get_relational_config.return_value = mock_relational_config

            mock_graph_config = MagicMock()
            mock_graph_config.graph_filename = "cognee.db"
            mock_get_graph_config.return_value = mock_graph_config

            mock_vector_config = MagicMock()
            mock_vector_config.vector_db_provider = "lancedb"
            mock_get_vectordb_config.return_value = mock_vector_config

            config.set("system_root_directory", test_dir)

            assert mock_base_config.system_root_directory == test_dir

    def test_set_data_root_directory(self):
        """Test setting data root directory"""
        test_dir = "/tmp/data"

        with patch("cognee.api.v1.config.config.get_base_config") as mock_get_base_config:
            mock_base_config = MagicMock()
            mock_get_base_config.return_value = mock_base_config

            config.set("data_root_directory", test_dir)

            assert mock_base_config.data_root_directory == test_dir
