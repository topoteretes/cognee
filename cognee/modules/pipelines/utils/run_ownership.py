"""Who owns a pipeline run, and is that owner still alive.

A heartbeat can only ever answer "it was alive N seconds ago", which forces
whoever consumes it to pick a threshold and accept being wrong in both
directions: too eager deletes a live run's work, too lax leaves a dead run's
dataset blocked. Ownership replaces that guess with a question the operating
system can answer exactly, for the case that covers most deployments.

The identity is deliberately hostname-scoped. Processes sharing a hostname
share a process table, so a recorded pid is only meaningful to a reader whose
own hostname matches. Containers get distinct hostnames and therefore never
interpret each other's pids, falling through to the heartbeat instead. Set
``COGNEE_NODE_ID`` where the hostname is not stable per host, or where several
hosts could report the same one.
"""

import os
import socket

from cognee.shared.logging_utils import get_logger

logger = get_logger("run_ownership")


def get_node_id() -> str:
    """Identity of the machine whose process table ``owner_pid`` refers to."""
    configured = os.getenv("COGNEE_NODE_ID")
    if configured:
        return configured

    try:
        return socket.gethostname()
    except Exception:
        # Without a stable identity, claim one that can never match a reader,
        # so ownership checks fall through to the heartbeat rather than
        # comparing pids across unrelated machines.
        return f"unknown-{os.getpid()}"


def get_owner_pid() -> int:
    return os.getpid()


def is_owner_process_alive(node_id, pid) -> bool | None:
    """Is the process that owns a run still running?

    Returns True (alive), False (definitively gone) or None (cannot tell from
    here, so the caller must fall back to the heartbeat).

    None is returned whenever ownership was never recorded or the run belongs
    to a different node, because a pid from another machine's process table
    says nothing about this one.
    """
    if not node_id or not pid:
        return None

    if node_id != get_node_id():
        return None

    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return None

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Alive, owned by another user. Signalling is denied but existence is
        # confirmed, which is all this asks.
        return True
    except Exception as error:
        logger.debug("Could not probe owner pid %s on node %s: %s", pid, node_id, error)
        return None

    return True
