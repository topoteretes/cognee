"""Configuration for recall tool calls (external database access).

Everything here is gated behind ``tool_calls_enabled`` (env
``TOOL_CALLS_ENABLED``), which defaults to **off**: recall never reaches an
external database unless a deployment explicitly opts in.
"""

import json
from functools import lru_cache
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict


class ToolsConfig(BaseSettings):
    # Master gate for the recall "tools" scope. Off by default: executing
    # LLM-generated SQL against external databases is opt-in per deployment.
    tool_calls_enabled: bool = False

    # Separate gate for WRITE-BACK (correction proposals that UPDATE the
    # source database). Requires tool_calls_enabled too. Writes are always
    # approval-gated: proposals never execute without an explicit apply call.
    tool_write_calls_enabled: bool = False

    # Deployment-level SQL connections as a JSON object mapping a connection
    # name to either a connection-string value or an object:
    #   TOOL_SQL_CONNECTIONS='{"analytics": {"connection_string": "postgresql://...",
    #                                        "allowed_tables": ["orders"], "max_rows": 50}}'
    # These are visible to EVERY authenticated caller — suitable for
    # single-tenant deployments only. Multi-tenant deployments should use
    # per-user registration (cognee.tools.register_sql_connection) instead.
    tool_sql_connections: str = "{}"

    # User-registered targets are denied by default. Configure an approved
    # SQLite root and/or comma-separated PostgreSQL host allowlist before
    # allowing per-user connections in a deployment.
    tool_sqlite_allowed_root: str = ""
    tool_sql_allowed_hosts: str = ""

    text_to_sql_max_rows: int = 100
    text_to_sql_max_attempts: int = 3
    text_to_sql_statement_timeout_ms: int = 5000
    text_to_sql_max_schema_tables: int = 50

    # Hard cap on rows one applied write proposal may touch; the apply step
    # rolls back and marks the proposal failed when exceeded.
    text_to_sql_max_affected_rows: int = 50

    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    def sql_connections(self) -> dict[str, dict[str, Any]]:
        """Parse ``tool_sql_connections`` into name → options dicts.

        String values are shorthand for ``{"connection_string": <value>}``.
        Malformed JSON raises so a misconfigured deployment fails loudly
        instead of silently exposing zero (or the wrong) connections.
        """
        parsed = json.loads(self.tool_sql_connections)
        if not isinstance(parsed, dict):
            raise ValueError("TOOL_SQL_CONNECTIONS must be a JSON object")
        connections: dict[str, dict[str, Any]] = {}
        for name, value in parsed.items():
            if isinstance(value, str):
                connections[name] = {"connection_string": value}
            elif isinstance(value, dict) and "connection_string" in value:
                connections[name] = value
            else:
                raise ValueError(
                    f"TOOL_SQL_CONNECTIONS entry '{name}' must be a connection string "
                    "or an object with a 'connection_string' key"
                )
        return connections

    def sql_allowed_hosts(self) -> set[str]:
        """Return normalized PostgreSQL hosts approved for user connections."""
        return {
            host.strip().lower().rstrip(".")
            for host in self.tool_sql_allowed_hosts.split(",")
            if host.strip()
        }

    def to_dict(self) -> dict:
        return {
            "tool_calls_enabled": self.tool_calls_enabled,
            "tool_write_calls_enabled": self.tool_write_calls_enabled,
            "text_to_sql_max_rows": self.text_to_sql_max_rows,
            "text_to_sql_max_attempts": self.text_to_sql_max_attempts,
            "text_to_sql_statement_timeout_ms": self.text_to_sql_statement_timeout_ms,
            "text_to_sql_max_schema_tables": self.text_to_sql_max_schema_tables,
            "text_to_sql_max_affected_rows": self.text_to_sql_max_affected_rows,
        }


@lru_cache
def get_tools_config() -> ToolsConfig:
    return ToolsConfig()
