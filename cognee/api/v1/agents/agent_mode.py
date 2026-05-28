"""Agent mode: automatic server shutdown when no agents are active.

When COGNEE_AGENT_MODE=true, the server tracks how many agents have
registered via the /register endpoint. A background watchdog thread
starts on the first registration and checks every 60 seconds whether
any agents remain. If the count drops to zero the watchdog sends
SIGTERM, shutting the server down gracefully.

This is designed for ephemeral deployments where an external
orchestrator spins up a Cognee server for one or more agents. Each
agent calls /register on connect and /unregister when done. Once all
agents finish, the server tears itself down automatically instead of
idling.

The watchdog does NOT start until at least one agent registers, so a
server launched with COGNEE_AGENT_MODE=true will stay alive
indefinitely while waiting for its first agent.
"""

from __future__ import annotations

import os
import signal
import threading
from typing import TYPE_CHECKING

from cognee.shared.logging_utils import get_logger

if TYPE_CHECKING:
    from cognee.modules.agents.models import AgentConnection, RegisterAgentRequest
    from cognee.modules.users.models.User import User

logger = get_logger(__name__)


def set_agent_mode(enabled: bool) -> None:
    os.environ["COGNEE_AGENT_MODE"] = str(enabled).lower()


def is_agent_mode_enabled() -> bool:
    return os.getenv("COGNEE_AGENT_MODE", "false").lower() == "true"


_lock = threading.Lock()
_active_count = 0
_watchdog_started = False


def _shutdown_server():
    logger.info("No active agents remaining — shutting down server")
    os.kill(os.getpid(), signal.SIGTERM)


def _watchdog():
    global _active_count
    timer = threading.Timer(60.0, _watchdog)
    timer.daemon = True
    timer.start()

    with _lock:
        count = _active_count

    if count <= 0:
        _shutdown_server()


async def register_agent(user: User, request: RegisterAgentRequest) -> AgentConnection:
    global _active_count, _watchdog_started

    from cognee.modules.agents.operations import register_agent_from_request

    connection = await register_agent_from_request(user, request)

    with _lock:
        _active_count += 1
        count = _active_count

        if is_agent_mode_enabled() and not _watchdog_started:
            _watchdog_started = True
            timer = threading.Timer(60.0, _watchdog)
            timer.daemon = True
            timer.start()
            logger.info("Agent mode watchdog started")

    logger.info("Agent registered (active: %d)", count)
    return connection


def unregister_agent() -> int:
    global _active_count

    with _lock:
        _active_count = max(0, _active_count - 1)
        count = _active_count

    logger.info("Agent unregistered (active: %d)", count)
    return count


def get_active_count() -> int:
    with _lock:
        return _active_count
