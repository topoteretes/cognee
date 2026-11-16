"""Minimal Todoist API client for the Todoist connector.

The Sync API (``POST /sync/v9/sync``) is one cursor: a ``sync_token`` (``"*"`` on the
first call) makes every call return exactly what changed since the token was issued —
items (tasks), projects and notes (comments), deletions included — plus the next token.
The REST API v2 fills the two gaps the Sync feed leaves: the comments of one task and
the current body of a task a changed comment belongs to.

``request`` returns the JSON object of a ``200`` answer, retried by
:func:`cognee_connectors_core.dlt_conventions.with_retries` on a transport error, a 429
or a 5xx. Every failure is a :class:`TodoistAPIError` carrying the HTTP status and an
endpoint with ids replaced, never a message or a body: Todoist's messages can quote task
content, and these errors end up in logs.
"""

import json
import logging
import re
import time
from collections.abc import Callable
from typing import Any

from cognee_connectors_core import dlt_conventions as dlt_utils

logger = logging.getLogger(__name__)

SYNC_URL = "https://api.todoist.com/sync/v9/sync"
REST_BASE_URL = "https://api.todoist.com/rest/v2"

# Retry budget for rate-limited / transient Todoist API responses.
_MAX_RETRIES = 5
# (connect, read) seconds; our own choice, Todoist documents no expected latency.
_TIMEOUT = (10, 30)


class TodoistError(RuntimeError):
    """Base class of the errors this client and source raise."""


class TodoistAPIError(TodoistError):
    """An error answer from Todoist (``status``), or no answer at all (``status`` 0).

    The message names the status and the endpoint with ids replaced. Todoist's own
    explanation is on ``detail`` with the token cut out; it is for the caller to read
    and is never logged.
    """

    def __init__(self, status: int, endpoint: str, detail: str = ""):
        answer = f"HTTP {status}" if status else "no answer"
        super().__init__(f"Todoist {endpoint}: {answer}")
        self.status = status
        self.endpoint = endpoint
        self.detail = detail
        # The response headers, for the retry policy (Retry-After). Set for a 429 only.
        self.headers: Any = None


class TodoistAuthError(TodoistAPIError):
    """Todoist rejected the API token (HTTP 401 or 403)."""

    def __init__(self, status: int, endpoint: str, detail: str = ""):
        super().__init__(status, endpoint, detail)


class TodoistRateLimitedError(TodoistAPIError):
    """Todoist answered HTTP 429."""

    def __init__(self, endpoint: str, detail: str = ""):
        super().__init__(429, endpoint, detail)


class TodoistResponseError(TodoistError):
    """Todoist answered ``200`` with a body that is not of the expected shape."""


def _endpoint(path: str) -> str:
    """The path with an object id replaced, so messages never carry one."""
    return re.sub(r"^/(tasks|projects|comments)/[^/]+$", r"/\1/{id}", path)


class TodoistClient:
    """One Todoist API session.

    ``request`` returns the JSON object of a ``200`` answer. Tests and hosts can pass any
    object with a compatible ``request`` method instead, one that takes ``check_active``
    and calls it before each attempt it makes.
    """

    def __init__(self, access_token: str, *, sleep: Callable[[float], None] = time.sleep):
        # This check is what keeps the token out of an exception: httpx names the value
        # of a header it rejects, and every token it would reject is refused here without
        # echoing it.
        token = access_token.strip() if isinstance(access_token, str) else ""
        if (
            not token
            or not token.isascii()
            or not token.isprintable()
            or any(char.isspace() for char in token)
        ):
            raise ValueError("The Todoist API token is empty or has an invalid format")
        self._access_token = token
        self._sleep = sleep

    def __repr__(self) -> str:
        return "TodoistClient(access_token=<redacted>)"

    def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        check_active: Callable[[], None] | None = None,
    ) -> dict:
        """The JSON object of a ``200`` answer, retried on a 429, a 5xx or a transport error.

        ``check_active`` is called before each retry wait (``with_retries``), so a host
        that has revoked the run does not wait out a backoff. What it raises propagates
        as it is: it is no network error, so it is neither retried nor wrapped.
        """
        import httpx  # lazy: importing the connectors package loads no client library

        url = path if path.startswith("https://") else f"{REST_BASE_URL}{path}"
        headers = {"Authorization": f"Bearer {self._access_token}"}

        def attempt() -> dict:
            try:
                response = httpx.request(
                    method,
                    url,
                    params=params,
                    data=data,
                    headers=headers,
                    timeout=_TIMEOUT,
                    follow_redirects=False,  # a 3xx is an error answer, never a new target
                )
            except httpx.TransportError:
                # Only the fact: the message of some httpx errors holds a header, so the
                # error is raised outside this block, where it is no one's __context__.
                raise TodoistAPIError(0, _endpoint(path)) from None
            return self._parse(response, path)

        return dlt_utils.with_retries(
            attempt,
            retry_on=(TodoistAPIError,),
            delay=self._retry_delay,
            attempts=_MAX_RETRIES,
            check_active=check_active,
            sleep=self._sleep,
        )

    def _parse(self, response: Any, path: str) -> dict:
        status, endpoint = response.status_code, _endpoint(path)
        try:
            body = response.json()
        except ValueError:
            body = None
        if status in (401, 403):
            raise TodoistAuthError(status, endpoint, self._detail(body))
        if status == 429:
            error = TodoistRateLimitedError(endpoint, self._detail(body))
            error.headers = response.headers
            raise error
        if status != 200:
            raise TodoistAPIError(status, endpoint, self._detail(body))
        if not isinstance(body, dict):
            raise TodoistResponseError(f"Todoist {endpoint}: HTTP 200 without a JSON object")
        return body

    def _retry_delay(self, error: BaseException, number: int) -> float | None:
        """A 429 waits Todoist's ``Retry-After`` (bounded), a 5xx or no answer backs off."""
        status = getattr(error, "status", None)
        if status == 429:
            header = getattr(error, "headers", None)
            try:
                delay = float(header.get("retry-after")) if header is not None else None
            except (TypeError, ValueError):
                delay = None
            if delay is None or not 0 < delay <= 30:
                delay = float(2**number)
            logger.warning(
                "Todoist: 429 on %s, retrying in %.1fs (%d/%d).",
                getattr(error, "endpoint", "?"),
                delay,
                number + 1,
                _MAX_RETRIES,
            )
            return delay
        if status == 0 or (status is not None and status >= 500):
            delay = float(2**number)
            logger.warning(
                "Todoist: %s on %s, retrying in %.1fs.",
                "no answer" if status == 0 else f"HTTP {status}",
                getattr(error, "endpoint", "?"),
                delay,
            )
            return delay
        return None

    def _detail(self, body: Any) -> str:
        """Todoist's ``error`` string with the token cut out, else empty. Never the raw body.

        The token is cut out before the text is shortened, so a cut can never leave a
        prefix of it.
        """
        message = body.get("error") if isinstance(body, dict) else None
        if not isinstance(message, str):
            return ""
        return message.replace(self._access_token, "[token]")[:200]

    # -- Todoist endpoints used by the source --------------------------------
    def sync(
        self,
        sync_token: str,
        resource_types: list[str],
        *,
        check_active: Callable[[], None] | None = None,
    ) -> dict:
        """One Sync API call: everything changed since ``sync_token``, plus the next token."""
        return self.request(
            "POST",
            SYNC_URL,
            data={
                "sync_token": sync_token,
                "resource_types": json.dumps(resource_types),
            },
            check_active=check_active,
        )

    def get_task(self, task_id: str, *, check_active: Callable[[], None] | None = None) -> dict:
        """One task's current body from the REST API."""
        return self.request("GET", f"/tasks/{task_id}", check_active=check_active)

    def list_comments(
        self, task_id: str, *, check_active: Callable[[], None] | None = None
    ) -> list:
        """Every comment of one task, across the REST API's cursor pagination."""
        comments: list[dict] = []
        cursor: str | None = None
        while True:
            params: dict[str, Any] = {"task_id": task_id}
            if cursor:
                params["cursor"] = cursor
            body = self.request("GET", "/comments", params=params, check_active=check_active)
            page = body.get("results")
            if not isinstance(page, list) or not all(isinstance(item, dict) for item in page):
                raise TodoistResponseError("Todoist /comments: results is not a list of objects")
            comments.extend(page)
            cursor = body.get("next_cursor")
            if not cursor:
                return comments


def build_todoist_service(access_token: str) -> TodoistClient:
    """Wrap a host-owned API token in a client, the way Asana and Apollo do.

    A ``service`` handed to :func:`cognee_connector_todoist.source.todoist_source` instead
    must offer ``sync``, ``get_task`` and ``list_comments`` (or a compatible ``request``)
    and call ``check_active`` before each attempt it makes.
    """
    return TodoistClient(access_token)
