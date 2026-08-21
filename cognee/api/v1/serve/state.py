"""Module-level singleton for the remote cloud client.

When set, V2 functions (remember/recall/improve/forget) route to the
cloud instead of executing locally.
"""

from typing import TYPE_CHECKING, Optional

from cognee.shared.logging_utils import get_logger

if TYPE_CHECKING:
    from cognee.api.v1.serve.cloud_client import CloudClient

logger = get_logger("serve")


class _Unset:
    """Sentinel distinguishing "argument not given" from an explicit ``None``.

    ``recall(search_type=None)`` must send ``"search_type": null`` on the
    wire (the server's opt-in for auto-routing and session-scope reads),
    while omitting the argument must omit the key (server default,
    GRAPH_COMPLETION). A plain ``None`` default cannot express both.
    """

    def __repr__(self) -> str:
        return "UNSET"

    def __bool__(self) -> bool:
        return False


UNSET = _Unset()

_remote_client: Optional["CloudClient"] = None


def get_remote_client() -> Optional["CloudClient"]:
    return _remote_client


def set_remote_client(client: Optional["CloudClient"]) -> None:
    global _remote_client
    _remote_client = client


def is_remote_mode() -> bool:
    return _remote_client is not None


def warn_unsupported_remote_params(operation: str, **params) -> None:
    """Warn about arguments the remote HTTP path cannot forward.

    Silent dropping is what taught integration authors to bypass the
    client — any param that cannot cross the HTTP boundary must be
    surfaced. Pass each candidate as ``name=value``; only values that
    are actually set (not ``None``) are reported.
    """
    dropped = sorted(name for name, value in params.items() if value is not None)
    if dropped:
        logger.warning(
            "%s(): ignoring %s while connected to a remote Cognee instance — "
            "these parameters cannot be forwarded over HTTP. "
            "Call cognee.disconnect() to apply them locally.",
            operation,
            ", ".join(dropped),
        )
