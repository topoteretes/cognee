"""Example-test configuration.

Sets lightweight in-process backends before cognee is imported, so no external
databases, vector stores or LLM keys are required.  All env-vars use
``os.environ.setdefault`` so they can still be overridden from the shell when
you *want* to run against a real provider.
"""

import os
import shutil
import tempfile
import pytest

@pytest.fixture(autouse=True)
def isolated_cognee_env(monkeypatch):
    """Create a unique tmp_path-based directory per test and set env vars."""
    temp_dir = tempfile.mkdtemp()
    
    # Core directories
    monkeypatch.setenv("COGNEE_DATA_PATH", temp_dir)
    monkeypatch.setenv("SYSTEM_ROOT_DIRECTORY", temp_dir)
    monkeypatch.setenv("DATA_ROOT_DIRECTORY", temp_dir)
    monkeypatch.setenv("CACHE_ROOT_DIRECTORY", temp_dir)
    monkeypatch.setenv("COGNEE_LOGS_DIR", temp_dir)
    
    # Providers
    monkeypatch.setenv("GRAPH_DATABASE_PROVIDER", "ladybug")
    monkeypatch.setenv("VECTOR_DB_PROVIDER", "lancedb")
    monkeypatch.setenv("DB_PROVIDER", "sqlite")
    
    # Settings
    monkeypatch.setenv("ENABLE_BACKEND_ACCESS_CONTROL", "false")
    monkeypatch.setenv("LLM_API_KEY", "mock-key-for-testing")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("LOG_LEVEL", "ERROR")
    monkeypatch.setenv("COGNEE_LOG_FILE", "false")
    monkeypatch.setenv("COGNEE_SKIP_CONNECTION_TEST", "true")
    monkeypatch.setenv("VECTOR_DB_SUBPROCESS_ENABLED", "false")
    monkeypatch.setenv("GRAPH_DATABASE_SUBPROCESS_ENABLED", "false")
    
    # Invalidate cache so cognee picks up the new paths
    from cognee.base_config import get_base_config
    get_base_config.cache_clear()
    
    yield temp_dir
    
    shutil.rmtree(temp_dir, ignore_errors=True)

from cognee.tests.utils.mock_llm import (  # noqa: F401, E402
    clean_cognee_state,
    mock_cognee_embeddings,
    mock_cognee_llm,
)
