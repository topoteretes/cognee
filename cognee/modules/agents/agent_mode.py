"""Agent mode: automatic server shutdown when no agents are active.

When COGNEE_AGENT_MODE=true, the server tracks how many agent connections
have registered via the /register endpoint. A background watchdog thread
starts on the first registration and checks every 60 seconds whether
any connections remain. If the count drops to zero the watchdog sends
SIGTERM, shutting the server down gracefully.

This is designed for ephemeral deployments where an external
orchestrator spins up a Cognee server for one or more agents. Each
agent calls /register on connect and /unregister when done. Once all
connections finish, the server tears itself down automatically instead of
idling.

The watchdog does NOT start until at least one connection registers, so a
server launched with COGNEE_AGENT_MODE=true will stay alive
indefinitely while waiting for its first connection.
"""

from __future__ import annotations

import os
import signal
import threading
from typing import TYPE_CHECKING

from cognee.shared.logging_utils import get_logger

if TYPE_CHECKING:
    from cognee.modules.agents.models import (
        AgentConnection,
        RegisterAgentRequest,
        UnregisterAgentRequest,
    )
    from cognee.modules.users.models.User import User

logger = get_logger(__name__)


def set_agent_mode(enabled: bool) -> None:
    os.environ["COGNEE_AGENT_MODE"] = str(enabled).lower()


def is_agent_mode_enabled() -> bool:
    return os.getenv("COGNEE_AGENT_MODE", "false").lower() == "true"


_lock = threading.Lock()
_active_count = 0
_active_connection_ids: set[str] = set()
_watchdog_started = False


def _shutdown_server():
    logger.info("No active agent connections remaining — shutting down server")
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
        if connection.id in _active_connection_ids:
            return connection

        _active_connection_ids.add(connection.id)
        _active_count += 1
        count = _active_count

        if is_agent_mode_enabled() and not _watchdog_started:
            _watchdog_started = True
            timer = threading.Timer(60.0, _watchdog)
            timer.daemon = True
            timer.start()
            logger.info("Agent mode watchdog started")

    logger.info("Agent connection registered (active: %d)", count)
    return connection


async def unregister_agent(user: User, request: UnregisterAgentRequest) -> int:
    global _active_count

    from cognee.modules.agents.registry import (
        build_agent_connection_id,
        deactivate_agent_connection,
    )

    connection_id = build_agent_connection_id(
        agent_session_name=request.agent_session_name,
        user_id=str(user.id) if user.id is not None else None,
    )

    await deactivate_agent_connection(user.id, connection_id)

    with _lock:
        if connection_id in _active_connection_ids:
            _active_connection_ids.discard(connection_id)
            _active_count = max(0, _active_count - 1)
        count = _active_count

    logger.info("Agent connection unregistered (active: %d)", count)
    return count


def get_active_count() -> int:
    with _lock:
        return _active_count
