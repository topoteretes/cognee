"""
Configuration and settings for Cognee MCP server.

This module handles all configuration, environment variables, and settings
that control the behavior of the MCP server.
"""

import os
import re
from typing import Optional, List

from cognee.shared.logging_utils import get_logger

# ExceptionGroup backport for Python 3.10 compatibility
# ExceptionGroup is built-in in Python 3.11+
try:
    from exceptiongroup import ExceptionGroup
except ImportError:
    from builtins import ExceptionGroup  # Python 3.11+


def validate_cors_origins(origins: List[str]) -> List[str]:
    """
    Validate CORS origins to prevent injection attacks.

    Args:
        origins: List of origin URLs

    Returns:
        Validated origins

    Raises:
        ValueError: If invalid origin detected
    """
    validated = []
    # Pattern matches: http(s)://domain(:port)
    origin_pattern = re.compile(
        r"^https?://[a-zA-Z0-9\-\.]+(:[0-9]{1,5})?$|^https?://[a-zA-Z0-9\-\.]+$"
    )

    for origin in origins:
        origin = origin.strip()
        if not origin:
            continue

        if not origin_pattern.match(origin):
            raise ValueError(
                f"Invalid CORS origin: {origin}. Must be in format: http(s)://domain(:port)"
            )

        validated.append(origin)

    return validated


class Settings:
    """
    Application settings loaded from environment variables.

    Attributes:
        BACKEND_API_TOKEN: Internal API token for Cognee backend
        CORS_ALLOWED_ORIGINS: List of allowed CORS origins
        ENABLE_BACKEND_ACCESS_CONTROL: Enable KB isolation
        ALLOW_HTTP_REQUESTS: Allow HTTP requests
        ALLOW_CYPHER_QUERY: Allow Cypher queries
        VECTOR_DB_PROVIDER: Vector database provider
        GRAPH_DATABASE_PROVIDER: Graph database provider
    """

    # Backend API Configuration
    BACKEND_API_TOKEN: Optional[str] = os.getenv("BACKEND_API_TOKEN")

    # CORS Configuration
    CORS_ALLOWED_ORIGINS: List[str] = validate_cors_origins(
        [
            origin.strip()
            for origin in os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:3000").split(",")
        ]
    )

    # Security Configuration
    ENABLE_BACKEND_ACCESS_CONTROL: bool = (
        os.getenv("ENABLE_BACKEND_ACCESS_CONTROL", "false").lower() == "true"
    )
    ALLOW_HTTP_REQUESTS: bool = os.getenv("ALLOW_HTTP_REQUESTS", "false").lower() == "true"
    ALLOW_CYPHER_QUERY: bool = os.getenv("ALLOW_CYPHER_QUERY", "false").lower() == "true"

    # Database Configuration
    VECTOR_DB_PROVIDER: str = os.getenv("VECTOR_DB_PROVIDER", "lancedb")
    GRAPH_DATABASE_PROVIDER: str = os.getenv("GRAPH_DATABASE_PROVIDER", "kuzu")


# Server Instructions for MCP
SERVER_INSTRUCTIONS = """Cognee Knowledge Search (Read Only)

Use this server to explore knowledge bases that were created in the Cognee UI.
It only exposes two tools:

1. list_datasets – discover every available knowledge base and surface IDs you
   can later reference in search queries.
2. search – ask natural language questions against one or more datasets. The
   Cognee backend performs ranking and returns evidence for LibreChat to use
   when drafting the final answer.

Best practices for LibreChat agents:
- Call list_datasets once per conversation to understand the KB catalog.
- Always pass dataset IDs when you need a specific KB. Searching without the
  datasets parameter queries everything the backend exposes to this MCP server.
- Adjust top_k (1-50) to balance recall vs. latency. 6-10 results usually give
  the LLM enough evidence while keeping responses snappy.
- search_type accepts GRAPH_COMPLETION (default, rich answers), CHUNKS (raw
  passages), or SUMMARIES (high level notes). Pick the output that best fits
  the user’s follow-up question.
- system_prompt lets you steer answer tone (e.g., “reply as a clinical
  researcher summarizing findings”). Leave it empty for the default behavior.

This MCP server never mutates data. It relies on the Cognee API that runs in a
separate container, so make sure `api-url` and `api-token` (if required) are
configured before launching the server. Backend access control rules still
apply: when ENABLE_BACKEND_ACCESS_CONTROL=true you may only search datasets you
have permission to view. When it is false the backend behaves as a single
global knowledge base and filtering by dataset is ignored."""


def validate_settings() -> bool:
    """
    Validate that all required settings are properly configured.

    Returns:
        bool: True if settings are valid, False otherwise

    Raises:
        ValueError: If critical settings are missing
    """
    errors = []
    warnings = []

    # Check critical settings
    if Settings.ENABLE_BACKEND_ACCESS_CONTROL:
        logger.info("✓ Backend access control enabled (KB isolation enforced)")
    else:
        warnings.append("Backend access control disabled")

    # Check CORS configuration
    if not Settings.CORS_ALLOWED_ORIGINS:
        warnings.append("CORS_ALLOWED_ORIGINS not set - will block cross-origin requests")

    # Log warnings (non-fatal)
    if warnings:
        logger.warning("Configuration warnings:")
        for warning in warnings:
            logger.warning(f"  - {warning}")

    # Fail on errors (critical)
    if errors:
        logger.error("Configuration validation FAILED:")
        for error in errors:
            logger.error(f"  - {error}")
        raise ValueError("Invalid configuration - see errors above")

    logger.info("✓ Configuration validation passed")
    return True


logger = get_logger()
