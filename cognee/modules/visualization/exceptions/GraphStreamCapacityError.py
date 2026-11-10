from fastapi import status

from cognee.exceptions.exceptions import CogneeTransientError


class GraphStreamCapacityError(CogneeTransientError):
    """Raised when this process already holds as many graph streams as it allows.

    Each streamed graph read keeps its compact graph in memory until the client
    has taken it, so the number in flight at once is capped. Retrying shortly
    succeeds once one of them finishes.
    """

    def __init__(
        self,
        message: str = "Too many graph streams are open on this server. Retry shortly.",
        name: str = "GraphStreamCapacityError",
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
    ):
        # Load shedding, not a fault: a warning, not an error, in the logs.
        super().__init__(message, name, status_code, log_level="WARNING")
