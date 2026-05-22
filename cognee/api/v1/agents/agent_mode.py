import os
import signal
import threading
from cognee.shared.logging_utils import get_logger

logger = get_logger(__name__)


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


def register_agent() -> int:
    global _active_count, _watchdog_started

    with _lock:
        _active_count += 1
        count = _active_count

        if not _watchdog_started:
            _watchdog_started = True
            timer = threading.Timer(60.0, _watchdog)
            timer.daemon = True
            timer.start()
            logger.info("Agent mode watchdog started")

    logger.info("Agent registered (active: %d)", count)
    return count


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
