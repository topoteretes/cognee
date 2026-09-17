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

UNMATCHED_ROUTE = "unmatched"


def _endpoint(request) -> str:
    """``"POST /api/v1/datasets/{dataset_id}/graph"`` — templated, never resolved.

    Built from the resolved path with every path parameter's value swapped back
    for its ``{name}``. It is not read off ``request.scope["route"]``: with
    routers included under a prefix (every cognee router), that object is the
    router-relative route, so it reports ``/{dataset_id}/graph`` without the
    ``/api/v1/datasets`` prefix and an empty path for routes declared as
    ``@router.post("")``. Ids only ever enter a path as path parameters, so
    substituting them keeps user data out of the event.
    """
    if request is None:
        return f"UNKNOWN {UNMATCHED_ROUTE}"
    method = getattr(request, "method", None) or "UNKNOWN"
    path = getattr(getattr(request, "url", None), "path", None)
    if not path:
        return f"{method} {UNMATCHED_ROUTE}"
    params = dict(getattr(request, "path_params", None) or {})
    segments = path.split("/")
    for name, value in params.items():
        value = str(value)
        if value in segments:
            segments[segments.index(value)] = f"{{{name}}}"
        else:
            # A ``{name:path}`` parameter spans several segments; fall back to
            # replacing the value inside the joined path.
            segments = "/".join(segments).replace(value, f"{{{name}}}", 1).split("/")
    return f"{method} {'/'.join(segments)}"


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
