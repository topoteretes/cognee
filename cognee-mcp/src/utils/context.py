from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.clients.cognee_client import CogneeClient

cognee_client: Optional["CogneeClient"] = None


def set_cognee_client(client: "CogneeClient") -> None:
    """Set the global cognee client instance."""
    global cognee_client
    cognee_client = client
