"""Configuration for Cognee FastMCP module."""

from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Cognee FastMCP Server Configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_prefix=""
    )

    MCP_SERVER_NAME: str = Field(
        default="cognee-mcp-server",
        description="Name of the MCP server"
    )
    MCP_TRANSPORT: str = Field(
        default="stdio",
        description="MCP transport: 'stdio' or 'sse'"
    )
    MCP_HOST: str = Field(
        default="127.0.0.1",
        description="Host for SSE transport"
    )
    MCP_PORT: int = Field(
        default=8000,
        description="Port for SSE transport"
    )

    COGNEE_API_URL: Optional[str] = Field(
        default=None,
        description="Optional URL of a remote Cognee FastAPI server. If unset, embedded Cognee is used."
    )
    COGNEE_API_KEY: Optional[str] = Field(
        default=None,
        description="Optional API token for remote Cognee API"
    )
    DEFAULT_DATASET: str = Field(
        default="default_dataset",
        description="Default dataset namespace for memory ingestion"
    )
    AUTO_COGNIFY: bool = Field(
        default=True,
        description="Whether to automatically run cognify() graph extraction after add()"
    )

    OPENAI_API_KEY: Optional[str] = Field(
        default=None,
        description="OpenAI API Key for Cognee embeddings and cognify pipelines"
    )
    ANTHROPIC_API_KEY: Optional[str] = Field(
        default=None,
        description="Anthropic API Key if configured in Cognee"
    )

    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level: DEBUG, INFO, WARNING, ERROR"
    )


settings = Settings()
