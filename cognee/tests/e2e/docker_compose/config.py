"""Configuration for the docker-compose full-stack end-to-end test.

Everything is driven by environment variables so the same test runs unchanged
in CI (where the workflow brings the stack up) and on a developer machine
(where you bring the stack up yourself with ``docker compose up``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class E2EConfig:
    """Resolved settings for the full-stack e2e run."""

    # Public API of the main `cognee` service (compose port 8000).
    api_url: str = field(
        default_factory=lambda: os.getenv("COGNEE_API_URL", "http://localhost:8000")
    )
    # Public endpoint of the `cognee-mcp` service (compose port 8001 -> 8000).
    mcp_url: str = field(
        default_factory=lambda: os.getenv("COGNEE_MCP_URL", "http://localhost:8001")
    )

    # Seeded default account created on first boot.
    username: str = field(
        default_factory=lambda: os.getenv("COGNEE_DEFAULT_USER", "default_user@example.com")
    )
    password: str = field(
        default_factory=lambda: os.getenv("COGNEE_DEFAULT_PASSWORD", "default_password")
    )

    # How long to wait for a freshly-started service to report healthy.
    startup_timeout: float = field(
        default_factory=lambda: float(os.getenv("COGNEE_E2E_STARTUP_TIMEOUT", "300"))
    )
    poll_interval: float = field(
        default_factory=lambda: float(os.getenv("COGNEE_E2E_POLL_INTERVAL", "3"))
    )

    # Run the LLM-dependent leg of the golden flow (cognify + search).
    # Off by default -> the PR-blocking run never touches a real LLM ("mock LLM
    # by default"). Turn on locally with COGNEE_E2E_RUN_LLM=1 and a real key.
    run_llm: bool = field(default_factory=lambda: _env_bool("COGNEE_E2E_RUN_LLM", False))

    # Whether this process is allowed to drive `docker compose` (restart a
    # service for the persistence check, read logs for the traceback check).
    # The CI workflow sets this to 1; a developer hitting an already-running
    # stack can leave it off to skip the compose-driving tests.
    manage_compose: bool = field(
        default_factory=lambda: _env_bool("COGNEE_E2E_MANAGE_COMPOSE", False)
    )

    compose_file: str = field(
        default_factory=lambda: os.getenv("COGNEE_E2E_COMPOSE_FILE", "docker-compose.yml")
    )
    compose_profiles: List[str] = field(
        default_factory=lambda: [
            p.strip()
            for p in os.getenv("COGNEE_E2E_COMPOSE_PROFILES", "postgres,mcp").split(",")
            if p.strip()
        ]
    )

    @property
    def health_url(self) -> str:
        return f"{self.api_url.rstrip('/')}/health"

    @property
    def mcp_health_url(self) -> str:
        return f"{self.mcp_url.rstrip('/')}/health"

    @property
    def mcp_sse_url(self) -> str:
        return f"{self.mcp_url.rstrip('/')}/sse"


CONFIG = E2EConfig()
