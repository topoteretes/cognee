"""Module-level singleton for the remote cloud client.

When set, V2 functions (remember/recall/improve/forget) route to the
cloud instead of executing locally.
"""

from typing import TYPE_CHECKING, Optional

from cognee.shared.logging_utils import get_logger

if TYPE_CHECKING:
    from cognee.api.v1.serve.cloud_client import CloudClient

logger = get_logger("serve.state")

_remote_client: Optional["CloudClient"] = None

# The client is per-process while credentials persist on disk, so "connected
# yesterday, silently local today" is a common surprise. Warn once per process
# when operations run locally although saved credentials exist.
_local_execution_noted = False


def get_remote_client() -> Optional["CloudClient"]:
    global _local_execution_noted
    if _remote_client is None and not _local_execution_noted:
        _local_execution_noted = True
        _note_local_execution()
    return _remote_client


def _note_local_execution() -> None:
    try:
        from cognee.api.v1.serve.credentials import load_credentials

        creds = load_credentials()
    except Exception:
        return
    if creds and creds.service_url and creds.api_key:
        logger.warning(
            "Executing memory operations locally. Saved connection credentials for %s "
            "exist, but this process has not called cognee.serve() — call it to route "
            "operations to that instance, or ignore this if local execution is intended.",
            creds.service_url,
        )


def set_remote_client(client: Optional["CloudClient"]) -> None:
    global _remote_client
    _remote_client = client


def is_remote_mode() -> bool:
    return _remote_client is not None
