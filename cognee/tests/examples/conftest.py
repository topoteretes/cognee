"""Example-test configuration.

Sets lightweight in-process backends before cognee is imported, so no external
databases, vector stores or LLM keys are required.  All env-vars use
``os.environ.setdefault`` so they can still be overridden from the shell when
you *want* to run against a real provider.
"""

import os

# ---------------------------------------------------------------------------
# Force lightweight, zero-dependency backends
# ---------------------------------------------------------------------------
os.environ.setdefault("VECTOR_DB_PROVIDER", "lancedb")
os.environ.setdefault("GRAPH_DATABASE_PROVIDER", "networkx")
os.environ.setdefault("DB_PROVIDER", "sqlite")

# Satisfy API-key validation without touching any real service
os.environ.setdefault("LLM_API_KEY", "mock-key-for-testing")
os.environ.setdefault("LLM_PROVIDER", "openai")
os.environ.setdefault("LLM_MODEL", "gpt-4o-mini")

# Silence noisy startup logs in CI
os.environ.setdefault("LOG_LEVEL", "ERROR")
os.environ.setdefault("COGNEE_LOG_FILE", "false")

# ---------------------------------------------------------------------------
# Re-export shared fixtures so all tests in this subtree inherit them
# without needing explicit imports.
# ---------------------------------------------------------------------------
from cognee.tests.utils.mock_llm import (  # noqa: F401, E402
    clean_cognee_state,
    mock_cognee_embeddings,
    mock_cognee_llm,
)
