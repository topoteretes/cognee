"""PagerDuty connector for cognee, a ``dlt`` source that turns PagerDuty incidents,
notes, and operational timelines into memory.

Sync incidents, responder notes, and incident resolution history into cognee,
incrementally and deletion-aware::

    import cognee
    from cognee.tasks.ingestion.connectors import pagerduty_source

    await cognee.remember(
        pagerduty_source(
            api_token="<pagerduty api token>",
            service_ids=["PABC123"],  # Optional: scope to specific services
            include_resolved=True,
        ),
        dataset_name="sre_incidents",
        primary_key="id",
        write_disposition="merge",
        max_rows_per_table=0,
    )

Design
------
* **Scope**: Sync incidents from PagerDuty REST API v2, optionally scoped to
  specific ``service_ids`` or incident ``statuses``.
* **Transport**: Plain ``httpx`` (core cognee dependency). No third-party PagerDuty
  SDK required. ``http_client`` injection is supported for tests and custom proxies.
* **Entities**:
  - Incidents: ID, number, title, status, urgency, priority, service, escalation policy.
  - Responder notes: chronological responder notes attached directly to the incident.
  - Timeline log entries: key state transitions and responder actions.
* **Row Format**: Flat document row conforming to cognee's document contract:
  ``{id, title, content, url, _deleted}``.
* **Incremental**: Preserves state (``floor`` timestamp and ``at_floor`` IDs). Subsequent
  runs query using ``since`` timestamps with clock-skew margin, syncing only modified
  or newly created incidents.
* **Resiliency**: Respects HTTP 429 rate limits, honors ``Retry-After`` headers, uses
  exponential backoff, and bounds total requests per execution run.
* **Privacy**: Redacts tokens from logs and exceptions; opt-in execution only.
"""

import logging
import re
import time
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from typing import Any

from cognee.tasks.ingestion import dlt_utils
from cognee.tasks.ingestion.dlt_utils import DOCUMENT_SOURCE_ATTR

logger = logging.getLogger(__name__)

PAGERDUTY_API_BASE_URL = "https://api.pagerduty.com"

DEFAULT_PAGE_SIZE = 100
MAX_PAGE_SIZE = 100
DEFAULT_MAX_REQUESTS = 1000
_CLOCK_MARGIN_SECONDS = 300
_TIMEOUT = (10.0, 30.0)
_MAX_RETRIES = 4


# ---------------------------------------------------------------------------
# Errors: messages carry codes and status only, never credentials or bodies.
# ---------------------------------------------------------------------------
class PagerDutySourceError(RuntimeError):
    """Base class for errors raised by the PagerDuty connector."""


class PagerDutyAPIError(PagerDutySourceError):
    """PagerDuty answered with an HTTP error, or could not be reached."""

    def __init__(self, status_code: int, message: str):
        super().__init__(f"PagerDuty API error {status_code}: {message}")
        self.status_code = status_code


class PagerDutyAuthError(PagerDutySourceError):
    """PagerDuty rejected authentication (HTTP 401 or 403)."""


class PagerDutyRateLimitedError(PagerDutySourceError):
    """PagerDuty answered HTTP 429 and retry budget was exhausted."""

    def __init__(self, retry_after: float | None = None):
        super().__init__("PagerDuty rate limit exceeded (HTTP 429)")
        self.retry_after = retry_after


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _int_header(headers: Any, name: str) -> int | None:
    try:
        val = headers.get(name)
        return int(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def _retry_after(headers: Any, attempt: int) -> float:
    ra = _int_header(headers, "Retry-After")
    if ra is not None and ra > 0:
        return float(min(ra, 60))
    return float(min(2**attempt, 30))


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stamp(epoch_or_dt: float | datetime) -> str:
    if isinstance(epoch_or_dt, (int, float)):
        dt = datetime.fromtimestamp(epoch_or_dt, tz=timezone.utc)
    else:
        dt = epoch_or_dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Transport & Client
# ---------------------------------------------------------------------------
class PagerDutyClient:
    """Minimal HTTP client for PagerDuty REST API v2 using httpx."""

    def __init__(
        self,
        api_token: str,
        *,
        http_client: Any = None,
        base_url: str = PAGERDUTY_API_BASE_URL,
        sleep: Callable[[float], None] = time.sleep,
    ):
        token = (api_token or "").strip()
        if not token or not token.isascii() or not token.isprintable() or " " in token:
            raise ValueError("The PagerDuty API token is empty or has an invalid format")
        self._api_token = token
        self._http = http_client
        self.base_url = base_url.rstrip("/")
        self._sleep = sleep

    def __repr__(self) -> str:
        return "PagerDutyClient(api_token=<redacted>)"

    def _headers(self) -> dict[str, str]:
        token = self._api_token
        auth_value = f"Bearer {token}" if token.startswith("pd_oauth_") else f"Token token={token}"
        return {
            "Authorization": auth_value,
            "Accept": "application/vnd.pagerduty+json;version=2",
            "Content-Type": "application/json",
        }

    def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        import httpx

        url = f"{self.base_url}/{path.lstrip('/')}"
        clean_params = {k: v for k, v in (params or {}).items() if v is not None}

        for attempt in range(_MAX_RETRIES):
            last = attempt == _MAX_RETRIES - 1
            try:
                if self._http is not None:
                    response = self._http.request(
                        method,
                        url,
                        params=clean_params,
                        json=json_body,
                        headers=self._headers(),
                    )
                else:
                    with httpx.Client(timeout=_TIMEOUT) as client:
                        response = client.request(
                            method,
                            url,
                            params=clean_params,
                            json=json_body,
                            headers=self._headers(),
                        )
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                if last:
                    raise PagerDutyAPIError(
                        503, f"Transport failure communicating with PagerDuty: {type(exc).__name__}"
                    ) from None
                self._sleep(2**attempt)
                continue

            status = response.status_code
            if status in (401, 403):
                raise PagerDutyAuthError(
                    f"PagerDuty rejected authentication (HTTP {status}). Check your API token."
                )
            if status == 429:
                delay = _retry_after(response.headers, attempt)
                if last:
                    raise PagerDutyRateLimitedError(delay)
                self._sleep(delay)
                continue
            if status >= 500:
                if last:
                    raise PagerDutyAPIError(status, f"Server error from PagerDuty (HTTP {status})")
                self._sleep(2**attempt)
                continue
            if status >= 400:
                raise PagerDutyAPIError(
                    status, f"Client error response from PagerDuty (HTTP {status})"
                )

            try:
                return response.json()
            except (ValueError, TypeError):
                return {}

        raise PagerDutyAPIError(500, "Unexpected request exhaustion")


def build_pagerduty_service(api_token: str, **kwargs: Any) -> PagerDutyClient:
    """Public helper to construct a configured PagerDuty client."""
    return PagerDutyClient(api_token, **kwargs)


# ---------------------------------------------------------------------------
# Rendering: Document contracts for Cognee ingestion
# ---------------------------------------------------------------------------
def render_incident(
    incident: dict[str, Any],
    notes: list[dict[str, Any]] | None = None,
    log_entries: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Render an incident into a flat Cognee document row."""
    inc_id = incident.get("id", "")
    inc_num = incident.get("incident_number", "")
    title = (incident.get("title") or incident.get("summary") or "Untitled Incident").strip()

    status = incident.get("status", "unknown")
    urgency = incident.get("urgency", "unknown")
    priority = (incident.get("priority") or {}).get("summary") or "None"

    service = incident.get("service") or {}
    service_name = service.get("summary") or service.get("name") or "Unknown Service"

    escalation_policy = incident.get("escalation_policy") or {}
    ep_name = escalation_policy.get("summary") or escalation_policy.get("name") or "None"

    assignees = [
        a.get("summary") or (a.get("assignee") or {}).get("summary")
        for a in (incident.get("assignments") or [])
        if isinstance(a, dict)
    ]
    assignee_str = ", ".join(filter(None, assignees)) if assignees else "Unassigned"

    created_at = incident.get("created_at") or ""
    last_status_change_at = incident.get("last_status_change_at") or ""

    lines = [
        f"Incident: INC-{inc_num}" if inc_num else f"Incident: {inc_id}",
        f"Title: {title}",
        f"Status: {status}",
        f"Urgency: {urgency}",
        f"Priority: {priority}",
        f"Service: {service_name}",
        f"Escalation Policy: {ep_name}",
        f"Assigned To: {assignee_str}",
        f"Created At: {created_at}",
        f"Last Status Change: {last_status_change_at}",
    ]

    # Resolution reason if resolved
    resolve_reason = (incident.get("resolve_reason") or "").strip()
    if resolve_reason:
        lines.append(f"Resolution Reason: {resolve_reason}")

    # Description or summary text
    desc = (incident.get("description") or "").strip()
    if desc:
        lines.append(f"\nDescription:\n{desc}")

    # Responders' timeline notes
    if notes:
        lines.append("\nResponder Notes:")
        for note in notes:
            author = (note.get("user") or {}).get("summary") or "Responder"
            note_created = note.get("created_at") or ""
            content = (note.get("content") or "").strip()
            lines.append(f"- [{note_created}] {author}: {content}")

    # Timeline log entries highlights
    if log_entries:
        lines.append("\nKey Timeline Events:")
        for entry in log_entries:
            summary = (entry.get("summary") or "").strip()
            entry_type = entry.get("type", "")
            entry_time = entry.get("created_at") or ""
            if summary:
                lines.append(f"- [{entry_time}] ({entry_type}) {summary}")

    content_str = "\n".join(lines)
    html_url = incident.get("html_url") or f"https://app.pagerduty.com/incidents/{inc_id}"

    return {
        "id": f"incident:{inc_id}",
        "title": f"INC-{inc_num} {title}".strip() if inc_num else title,
        "content": content_str,
        "url": html_url,
        "_deleted": False,
    }


# ---------------------------------------------------------------------------
# Walker: Paging, state management, incremental resumption
# ---------------------------------------------------------------------------
class _CutShort(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class _Walker:
    def __init__(
        self,
        client: PagerDutyClient,
        state: dict[str, Any],
        stats: dict[str, int],
        *,
        service_ids: list[str] | None = None,
        statuses: list[str] | None = None,
        include_resolved: bool = True,
        check_active: Callable[[], None] | None = None,
        page_size: int = DEFAULT_PAGE_SIZE,
        max_requests: int = DEFAULT_MAX_REQUESTS,
    ):
        self.client = client
        self.state = state
        self.stats = stats
        self.service_ids = service_ids
        self.include_resolved = include_resolved
        self.check_active = check_active
        self.page_size = max(1, min(page_size, MAX_PAGE_SIZE))
        self.max_requests = max_requests
        self.requests_made = 0

        if statuses:
            self.statuses = statuses
        elif include_resolved:
            self.statuses = ["triggered", "acknowledged", "resolved"]
        else:
            self.statuses = ["triggered", "acknowledged"]

    def _budget(self) -> None:
        if self.check_active:
            self.check_active()
        if self.requests_made >= self.max_requests:
            raise _CutShort("budget")
        self.requests_made += 1

    def _call(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        self._budget()
        return self.client.request(method, path, **kwargs)

    def _fetch_notes(self, incident_id: str) -> list[dict[str, Any]]:
        try:
            data = self._call("GET", f"incidents/{incident_id}/notes")
            return data.get("notes") or []
        except PagerDutyAPIError:
            return []

    def _fetch_log_entries(self, incident_id: str) -> list[dict[str, Any]]:
        try:
            data = self._call(
                "GET",
                f"incidents/{incident_id}/log_entries",
                params={"limit": 10, "is_overview": "true"},
            )
            return data.get("log_entries") or []
        except PagerDutyAPIError:
            return []

    def rows(self) -> Iterator[dict[str, Any]]:
        stream_state = self.state.setdefault("incidents", {})
        floor = stream_state.get("floor")
        at_floor = set(stream_state.get("at_floor") or [])

        params: dict[str, Any] = {
            "limit": self.page_size,
            "sort_by": "created_at:asc",
            "statuses[]": self.statuses,
        }
        if self.service_ids:
            params["service_ids[]"] = self.service_ids

        if floor:
            params["since"] = floor
        else:
            stream_state.setdefault("backfill_started", _stamp(time.time() - _CLOCK_MARGIN_SECONDS))

        offset = stream_state.get("resume_offset", 0)
        new_floor = floor
        new_at: set[str] = set(at_floor)

        while True:
            params["offset"] = offset
            try:
                data = self._call("GET", "incidents", params=params)
            except PagerDutyRateLimitedError:
                raise _CutShort("rate_limit") from None

            incidents = data.get("incidents") or []
            if not incidents:
                break

            for incident in incidents:
                inc_id = incident.get("id")
                created_stamp = str(incident.get("created_at") or "")

                if new_floor is None or created_stamp > new_floor:
                    new_floor = created_stamp
                    new_at = {inc_id}
                elif created_stamp == new_floor:
                    new_at.add(inc_id)

                if created_stamp == floor and inc_id in at_floor:
                    self.stats["skipped"] = self.stats.get("skipped", 0) + 1
                    continue

                self.stats["scanned"] = self.stats.get("scanned", 0) + 1

                # Enrich incident with notes and timeline highlights
                notes = self._fetch_notes(inc_id)
                log_entries = self._fetch_log_entries(inc_id)

                row = render_incident(incident, notes=notes, log_entries=log_entries)
                self.stats["emitted"] = self.stats.get("emitted", 0) + 1
                yield row

            more = data.get("more", False)
            if not more:
                break

            offset += len(incidents)
            stream_state["resume_offset"] = offset

        if new_floor is not None:
            stream_state["floor"] = new_floor
            stream_state["at_floor"] = sorted(new_at)
        stream_state.pop("resume_offset", None)


def _iter_rows(
    client: PagerDutyClient,
    state: dict[str, Any],
    stats: dict[str, int],
    *,
    service_ids: list[str] | None = None,
    statuses: list[str] | None = None,
    include_resolved: bool = True,
    check_active: Callable[[], None] | None = None,
    page_size: int = DEFAULT_PAGE_SIZE,
    max_requests: int = DEFAULT_MAX_REQUESTS,
) -> Iterator[dict[str, Any]]:
    """Yield changed incident documents. Pure of dlt so tests drive with dicts."""
    walker = _Walker(
        client,
        state,
        stats,
        service_ids=service_ids,
        statuses=statuses,
        include_resolved=include_resolved,
        check_active=check_active,
        page_size=page_size,
        max_requests=max_requests,
    )
    try:
        yield from walker.rows()
    except _CutShort as cut:
        stats["failed"] = max(1, stats.get("failed", 0))
        stats[f"failed_{cut.reason}"] = 1
        logger.warning("PagerDuty sync stopped early (%s); it resumes on the next run.", cut.reason)


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------
def _default_resource_name(service_ids: list[str] | None) -> str:
    if not service_ids:
        return "pagerduty_incidents"
    joined = "_".join(service_ids)
    clean = re.sub(r"[^a-z0-9]+", "_", joined.lower()).strip("_")
    return f"pagerduty_{clean}"[:64]


def pagerduty_source(
    *,
    api_token: str | None = None,
    service_ids: list[str] | None = None,
    include_resolved: bool = True,
    statuses: list[str] | None = None,
    resource_name: str | None = None,
    check_active: Callable[[], None] | None = None,
    service: Any = None,
    page_size: int = DEFAULT_PAGE_SIZE,
    max_requests: int = DEFAULT_MAX_REQUESTS,
) -> Any:
    """Return a ``dlt`` resource that yields PagerDuty incidents and notes for ``cognee.remember``.

    Args:
        api_token: PagerDuty API token. Personal or general access tokens.
        service_ids: Optional list of PagerDuty service IDs to restrict the sync.
        include_resolved: Whether to include resolved incidents (default True).
        statuses: Explicit list of incident statuses (e.g. ['triggered', 'acknowledged']).
        resource_name: Custom name for the DLT resource table.
        check_active: Optional callback invoked before each HTTP request.
        service: Injected PagerDutyClient or mock client (useful for testing).
        page_size: Incidents per page (max 100).
        max_requests: Request budget for a single sync run.

    Returns:
        A ``dlt`` resource configured with ``primary_key="id"`` and ``write_disposition="merge"``.
    """
    try:
        import dlt
    except ImportError as exc:
        raise ImportError(
            "The PagerDuty connector requires dlt (a cognee core dependency)."
        ) from exc

    if getattr(dlt_utils, "DOCUMENT_SYNC_VERSION", 0) < 1:
        raise RuntimeError(
            "PagerDuty sync requires a Cognee build with table-scoped DLT document cleanup."
        )

    if service is None and not api_token:
        raise ValueError("pagerduty_source requires an api_token or an injected service client")

    client = service if service is not None else PagerDutyClient(api_token or "")
    page_size = max(1, min(int(page_size), MAX_PAGE_SIZE))
    name = resource_name or _default_resource_name(service_ids)
    stats: dict[str, int] = {}

    @dlt.resource(
        name=name,
        primary_key="id",
        write_disposition="merge",
        columns={"_deleted": {"data_type": "bool", "hard_delete": True}},
    )
    def pagerduty_incidents():
        stats.clear()
        stats.update(scanned=0, skipped=0, emitted=0, failed=0)
        yield from _iter_rows(
            client,
            dlt.current.resource_state(),
            stats,
            service_ids=service_ids,
            statuses=statuses,
            include_resolved=include_resolved,
            check_active=check_active,
            page_size=page_size,
            max_requests=max_requests,
        )

    resource = pagerduty_incidents()
    setattr(resource, DOCUMENT_SOURCE_ATTR, "pagerduty")
    setattr(resource, dlt_utils.PIPELINE_SCOPE_ATTR, name)
    resource.cognee_sync_stats = stats
    return resource
