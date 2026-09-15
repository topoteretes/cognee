"""Telemetry for errors the API returns (SDK-635).

Cognee reported what users do when it works and almost nothing when it broke:
of the telemetry events in the tree, only the pipeline runner emitted a failure
event (``Pipeline Run Errored``), and nothing covered the HTTP layer. Errors
surfaced only in logs, which for self-hosted installs are exactly the logs we
never see.

This module emits one event per error response, from the three places an error
can leave the API: the ``CogneeApiError`` handler, the request-validation
handler, and — via middleware — exceptions no handler claims, which Starlette
turns into a bare 500.

What the event deliberately does NOT carry is ``exc.message``. Messages are
assembled at the raise site and routinely interpolate user content
(``f"Dataset '{entry}' not found."``, ``f"Agent {agent_id} not found"``), so
shipping them would turn error telemetry into a content channel. The event
carries the exception type, the error's declared name, the status code and the
TEMPLATED route — never the resolved path, whose ids identify data.
"""

from cognee import __version__ as cognee_version
from cognee.shared.logging_utils import get_logger
from cognee.shared.utils import send_telemetry

logger = get_logger()

API_EXCEPTION_EVENT = "API Exception Raised"

# Routing has already matched by the time a handler runs, so the route object
# carries the templated path. Without it the raw URL would embed dataset and
# document ids, so an unmatched request reports this instead.
UNMATCHED_ROUTE = "unmatched"


def _endpoint(request) -> str:
    """``"POST /api/v1/datasets/{dataset_id}/graph"`` — templated, never resolved."""
    if request is None:
        return f"UNKNOWN {UNMATCHED_ROUTE}"
    route = request.scope.get("route") if hasattr(request, "scope") else None
    path = getattr(route, "path", None) or UNMATCHED_ROUTE
    method = getattr(request, "method", None) or "UNKNOWN"
    return f"{method} {path}"


def send_api_exception_telemetry(
    request,
    exc: BaseException,
    status_code: int,
    *,
    error_name: str | None = None,
    improperly_defined: bool = False,
) -> None:
    """Record one error response. Never raises, never alters the response.

    ``send_telemetry`` is already fire-and-forget (it schedules a task on the
    running loop and drops the event when there is none), so this adds no
    latency to the error path. The broad except is the belt to that braces: a
    telemetry defect must not turn a handled 404 into an unhandled crash.
    """
    try:
        properties = {
            "endpoint": _endpoint(request),
            "exception_type": type(exc).__name__,
            "status_code": int(status_code),
            "cognee_version": cognee_version,
        }
        # CogneeApiError.name is a literal set at the raise site (never
        # interpolated), so it is a safe, more specific dimension than the class.
        if error_name:
            properties["error_name"] = str(error_name)
        if improperly_defined:
            # The exception class itself is malformed — our bug, not a user error.
            properties["improperly_defined_exception"] = True

        send_telemetry(API_EXCEPTION_EVENT, None, additional_properties=properties)
    except Exception:
        logger.debug("API exception telemetry skipped", exc_info=True)
