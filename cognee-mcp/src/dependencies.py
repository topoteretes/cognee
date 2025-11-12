"""
Dependency injection container for MCP server.

This module provides a centralized container for managing shared dependencies
across the MCP server, replacing global variables with proper dependency injection.
"""

from typing import Optional
import logging

from cognee_client import CogneeClient
from config import Settings


class DependencyContainer:
    """
    Dependency injection container for MCP server.

    Manages lifecycle of shared resources:
    - HTTP client
    - Configuration
    - Logger instances
    """

    def __init__(self, api_url: str, api_token: Optional[str] = None):
        if not api_url:
            raise ValueError("Cognee API URL must be provided")

        self._cognee_client: Optional[CogneeClient] = None
        self._settings = Settings()
        self._api_url = api_url
        self._api_token = api_token
        self._logger = logging.getLogger(__name__)

    @property
    def cognee_client(self) -> CogneeClient:
        """Get or create Cognee client (lazy initialization)."""
        if self._cognee_client is None:
            token = self._api_token or self._settings.BACKEND_API_TOKEN
            self._cognee_client = CogneeClient(api_url=self._api_url, api_token=token)
        return self._cognee_client

    @property
    def settings(self) -> Settings:
        """Get settings instance."""
        return self._settings

    @property
    def logger(self) -> logging.Logger:
        """Get logger instance."""
        return self._logger

    async def cleanup(self):
        """Cleanup resources on shutdown."""
        if self._cognee_client:
            await self._cognee_client.close()
            self._cognee_client = None
        self._logger.info("Dependency container cleaned up")
