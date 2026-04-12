"""Module-level singleton for the remote cloud client.

When set, V2 functions (remember/recall/improve/forget) route to the
cloud instead of executing locally.
"""

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from cognee.api.v1.serve.cloud_client import CloudClient

_remote_client: Optional["CloudClient"] = None


def get_remote_client() -> Optional["CloudClient"]:
    return _remote_client


def set_remote_client(client: Optional["CloudClient"]) -> None:
    global _remote_client
    _remote_client = client


def is_remote_mode() -> bool:
    return _remote_client is not None
