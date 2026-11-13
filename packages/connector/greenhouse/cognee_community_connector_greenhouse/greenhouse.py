"""DLT source for Greenhouse (Harvest v3) — jobs, job posts, and opt-in scorecards.

Fetches Greenhouse recruiting data and renders every record as a text document,
then yields the documents as dlt resources for cognee's ingestion pipeline.

Design follows the ``notion`` connector (the closest reference for this family):
the source declares ``cognee_document_source = "greenhouse"``, so each row is
ingested as a *normal document* and flows through cognify entity extraction
rather than the deterministic dlt-row schema path.

How sync + forget-on-delete work
--------------------------------
Harvest v3 exposes **cursor**-based pagination on list endpoints (a ``Link``
header with ``rel="next"``) and the issue's ``updated_after`` filter exists only
on a handful of resources. There is no change or delete feed. Each run is
therefore a **full snapshot**: ``write_disposition="replace"`` rewrites staging
with exactly the records currently visible. A job, post, or scorecard that is
deleted — or a candidate anonymized under GDPR right-to-be-forgotten, which
removes their scorecards — simply drops out of the listing, so it is absent
from the snapshot and cognee's existing ``orphan_cleanup`` forgets it from the
graph and vector stores on the next sync. Unchanged records keep a stable
content-hash ``data_id``, so only genuinely new/changed content is re-ingested.

Harvest v1/v2 stopped accepting requests on 2026-08-31, so this connector speaks
only Harvest **v3**, authenticating with OAuth 2.0 client credentials
(``https://auth.greenhouse.io/token``) — never the legacy API-key basic auth.

Privacy
-------
Interview feedback (scorecards) is candidate personal data. The
``greenhouse_scorecards`` resource is **off by default**: it is only produced
when ``include_scorecards=True``, and even then the rendered documents contain
no private notes, no candidates' contact details, and no compensation — only the
interview, the interviewer's name, the overall recommendation, and attribute
ratings. Anonymous IDs (candidate/application) are kept for provenance; add
``scorecard_application_ids`` to restrict which applications' scorecards leave
Greenhouse at all. This connector never performs writes.
"""

from __future__ import annotations

import base64
import os
import re
import time
from typing import Any
from urllib.parse import urlencode

import httpx
from cognee.shared.logging_utils import get_logger

logger = get_logger("greenhouse_connector")

GREENHOUSE_TABLE_JOBS = "greenhouse_jobs"
GREENHOUSE_TABLE_JOB_POSTS = "greenhouse_job_posts"
GREENHOUSE_TABLE_SCORECARDS = "greenhouse_scorecards"
GREENHOUSE_SOURCE_NAME = "greenhouse"

_TOKEN_URL = "https://auth.greenhouse.io/token"
_BASE_URL = "https://harvest.greenhouse.io"

# Transient status codes worth retrying (rate limit / server).
_MAX_RETRIES = 5

_EXTRA_HINT = (
    'The Greenhouse connector requires the "greenhouse" extra: '
    'pip install "cognee[greenhouse]" (provides dlt and httpx).'
)

_TAG_RE = re.compile(r"<[^>]+>")


class GreenhouseAPIError(httpx.HTTPStatusError):
    """Raised when Harvest v3 returns an error status.

    Carries the parsed response body so callers can surface details (for
    example a missing permission scope) instead of a bare status code.
    """


class GreenhouseClient:
    """Minimal Harvest v3 API client with OAuth 2.0 client-credentials auth.

    Handles token acquisition, expiry/refresh-on-401, transient retries, and
    cursor-based pagination helpers. ``session`` is injectable for tests.
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        sub: str | None = None,
        session: httpx.Client | None = None,
        token_url: str = _TOKEN_URL,
        base_url: str = _BASE_URL,
    ):
        self._client_id = client_id
        self._client_secret = client_secret
        self._sub = sub
        self._token: dict[str, Any] = {}
        self._token_url = token_url
        self._base_url = base_url
        self._session = session or httpx.Client()

    # -- token lifecycle ----------------------------------------------------

    def _fetch_token(self) -> dict[str, Any]:
        """Exchange client credentials for a short-lived access token."""
        basic = base64.b64encode(f"{self._client_id}:{self._client_secret}".encode()).decode(
            "ascii"
        )
        data = {"grant_type": "client_credentials"}
        if self._sub:
            data["sub"] = self._sub
        resp = self._session.post(
            self._token_url,
            headers={
                "Authorization": f"Basic {basic}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data=data,
            timeout=60,
        )
        resp.raise_for_status()
        token = resp.json()
        token["_expires_at"] = time.time() + int(token.get("expires_in", 3600))
        return token

    def _access_token(self) -> str:
        if self._token.get("access_token") and self._token.get("_expires_at", 0) > time.time():
            return self._token["access_token"]
        self._token = self._fetch_token()
        return self._token["access_token"]

    # -- requests -----------------------------------------------------------

    def get(self, url: str, params: dict[str, Any] | None = None) -> tuple[Any, httpx.Headers]:
        """GET ``url`` (path or absolute) with bearer auth.

        Returns ``(json, headers)``. Retries rate-limit/transient errors with
        backoff, and transparently re-fetches the access token once on a 401
        before giving up. Non-2xx responses raise :class:`GreenhouseAPIError`.
        """
        if not url.startswith("http"):
            url = self._base_url + url
        if params:
            seperator = "&" if "?" in url else "?"
            url = f"{url}{seperator}{urlencode(params)}"

        headers = {"Authorization": f"Bearer {self._access_token()}"}
        for attempt in range(_MAX_RETRIES):
            try:
                resp = self._session.get(url, headers=headers, timeout=60)
            except httpx.TransportError:
                if attempt == _MAX_RETRIES - 1:
                    raise
                time.sleep(float(2**attempt))
                continue

            if resp.status_code == 401:
                # Stale/expired token — fetch a fresh one and retry once.
                self._token = self._fetch_token()
                headers = {"Authorization": f"Bearer {self._token['access_token']}"}
                resp = self._session.get(url, headers=headers, timeout=60)

            self._raise_if_error(resp)
            return resp.json(), resp.headers

        raise GreenhouseAPIError("Exhausted retries", request=httpx.Request("GET", url))

    @staticmethod
    def _raise_if_error(resp: httpx.Response) -> None:
        if resp.status_code < 400:
            return
        try:
            body = resp.json()
        except ValueError:
            body = resp.text
        raise GreenhouseAPIError(
            f"Greenhouse Harvest v3 error {resp.status_code}: {body}",
            request=resp.request,
            response=resp,
        )


def _parse_next_link(header: Any) -> str | None:
    """Extract the ``rel="next"`` URL from a (possibly multi-value) Link header."""
    if not header:
        return None
    if isinstance(header, (list, tuple)):
        header = ",".join(str(h) for h in header)
    for part in f"{header}".split(","):
        match = re.match(r"\s*<([^>]+)>\s*(.*)$", part)
        if not match:
            continue
        params = match.group(2)
        if 'rel="next"' in params or "rel=next" in params:
            return match.group(1)
    return None


def _iter_payload(client: GreenhouseClient, path: str, params: dict[str, Any] | None = None):
    """Yield every item across Harvest v3's Link-header cursor pagination."""
    url: str | None = path if params is None else f"{path}?{urlencode(params)}"
    while url:
        payload, headers = client.get(url)
        if isinstance(payload, dict):
            # Defensive: accept wrappers like {"data": [...]} / {"results": [...]}
            items = payload.get("data") or payload.get("results") or []
        else:
            items = payload or []
        yield from items
        url = _parse_next_link(headers.get("Link") or headers.get("link"))


# ---------------------------------------------------------------------------
# Rendering (allowlist-based; never dumps a raw upstream record)
# ---------------------------------------------------------------------------


def _plain_text(html: str | None) -> str:
    """Strip HTML tags/entities to plain text for ingestion as a document."""
    if not html:
        return ""
    text = _TAG_RE.sub(" ", html)
    replace = {
        "&amp;": "&",
        "&lt;": "<",
        "&gt;": ">",
        "&quot;": '"',
        "&#39;": "'",
        "&#x27;": "'",
        "&nbsp;": " ",
    }
    for entity, char in replace.items():
        text = text.replace(entity, char)
    text = re.sub(r"\s+([.,!?;:%]+)", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def _text(value: Any) -> str:
    """Coerce a scalar (or text-bearing dict) to a string."""
    if isinstance(value, dict):
        return _text(value.get("name", value.get("title", "")))
    return "" if value is None else str(value)


def _job_to_row(job: dict) -> dict[str, Any]:
    """Render a Harvest v3 job object to a document row (allowlist fields only)."""
    fields: list[str] = []
    for label, key in (
        ("Status", "status"),
        ("Department", "department"),
        ("Office", "office"),
        ("Requisition", "requisition_id"),
        ("Opened", "opened_at"),
        ("Closed", "closed_at"),
        ("Updated", "updated_at"),
    ):
        value = _text(job.get(key)).strip()
        if value:
            fields.append(f"{label}: {value}")
    return {
        "id": job.get("id"),
        "url": None,
        "title": _text(job.get("title") or job.get("name")) or f"Job {job.get('id')}",
        "content": "\n".join(fields),
    }


def _job_post_to_row(post: dict) -> dict[str, Any]:
    """Render a Harvest v3 job post to a document row (description text included).

    The issue's ``jobs`` acceptance criteria need real prose; in Harvest v3 the
    description lives on the job *post*, not the job object, so posts are their
    own resource. ``content`` is HTML in v3 and is reduced to plain text here.
    """
    fields: list[str] = []
    for label, key in (("Job ID", "job_id"), ("Live", "live"), ("Updated", "updated_at")):
        value = _text(post.get(key)).strip()
        if value:
            fields.append(f"{label}: {value}")
    description = _plain_text(post.get("content"))
    if description:
        fields.append(f"Description: {description}")
    return {
        "id": post.get("id"),
        "url": post.get("public_url"),
        "title": _text(post.get("title")) or f"Job post {post.get('id')}",
        "content": "\n".join(fields),
    }


def _interviewer_name(scorecard: dict) -> str:
    submitted_by = scorecard.get("submitted_by") or {}
    parts = [submitted_by.get("first_name"), submitted_by.get("last_name")]
    return " ".join(str(part) for part in parts if part) or _text(scorecard.get("interviewer"))


def _scorecard_to_row(scorecard: dict) -> dict[str, Any]:
    """Render a scorecard to a document row, excluding private data.

    No private/public notes, no candidate contact or compensation data. Only the
    structure (interview, interviewer, recommendation, ratings) is rendered,
    because that is the knowledge a memory graph should keep about a review.
    """
    fields: list[str] = []
    for label, key in (
        ("Application", "application_id"),
        ("Candidate", "candidate_id"),
        ("Interview", "interview"),
        ("Recommendation", "overall_recommendation"),
        ("Submitted by", None),
        ("Submitted at", "submitted_at"),
        ("Updated", "updated_at"),
    ):
        value = _interviewer_name(scorecard) if key is None else _text(scorecard.get(key))
        if isinstance(value, str) and value:
            fields.append(f"{label}: {value.strip()}")

    attributes = scorecard.get("ratings") or scorecard.get("question_ratings") or []
    if attributes:
        lines = []
        for attr in attributes:
            if isinstance(attr, dict):
                name = _text(attr.get("name") or attr.get("attribute") or attr.get("question"))
                value = _text(attr.get("score") or attr.get("rating_value"))
                if name:
                    lines.append(f"- {name}{f': {value}' if value else ''}".strip())
        if lines:
            fields.append("Ratings:")
            fields.extend(lines)

    return {
        "id": scorecard.get("id"),
        "url": None,
        "title": f"Scorecard {scorecard.get('id')}",
        "content": "\n".join(fields),
    }


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def greenhouse_source(
    client_id: str | None = None,
    client_secret: str | None = None,
    sub: str | None = None,
    client: GreenhouseClient | None = None,
    job_statuses: list[str] | None = None,
    include_scorecards: bool = False,
    scorecard_application_ids: list[int] | None = None,
):
    """Create a dlt source that yields Greenhouse documents for ``remember``.

    Args:
        client_id: Harvest v3 OAuth client id. Falls back to
            ``GREENHOUSE_CLIENT_ID``.
        client_secret: Harvest v3 OAuth client secret. Falls back to
            ``GREENHOUSE_CLIENT_SECRET``.
        sub: Optional Greenhouse ``user_id`` authorizing the request (see the
            OAuth client-credentials docs). Falls back to ``GREENHOUSE_SUB``.
        client: Pre-built :class:`GreenhouseClient` (mainly a test-injection
            point); when omitted one is built from the credentials above.
        job_statuses: Restrict to jobs with these statuses (e.g. ``["open"]``)
            passed through as the ``status`` filter; ``None`` = all statuses.
        include_scorecards: When False (default) no scorecard data is produced.
            Scorecards are candidate feedback and must be explicitly opted in.
        scorecard_application_ids: When set, only scorecards belonging to these
            application ids are produced (explicit scoping for sensitive data).

    Returns:
        A dlt source suitable for ``cognee.add(...)`` / ``cognee.remember(...)``.
        Resources: ``greenhouse_jobs``, ``greenhouse_job_posts``, and (only when
        ``include_scorecards``) ``greenhouse_scorecards``.
    """
    try:
        import dlt
    except ImportError as exc:
        raise ImportError(_EXTRA_HINT) from exc

    if client is None:
        resolved_id = client_id or os.environ.get("GREENHOUSE_CLIENT_ID")
        resolved_secret = client_secret or os.environ.get("GREENHOUSE_CLIENT_SECRET")
        resolved_sub = sub or os.environ.get("GREENHOUSE_SUB")
        if not resolved_id or not resolved_secret:
            raise ValueError(
                "Greenhouse client credentials required: pass client_id=/client_secret= or "
                "set GREENHOUSE_CLIENT_ID / GREENHOUSE_CLIENT_SECRET."
            )
        client = GreenhouseClient(resolved_id, resolved_secret, sub=resolved_sub)

    @dlt.resource(name=GREENHOUSE_TABLE_JOBS, primary_key="id", write_disposition="replace")
    def greenhouse_jobs():
        count = 0
        params = {"status": ",".join(job_statuses)} if job_statuses else None
        for job in _iter_payload(client, "/v3/jobs", params):
            count += 1
            yield _job_to_row(job)
        logger.info("Greenhouse: synced %d job(s).", count)

    @dlt.resource(name=GREENHOUSE_TABLE_JOB_POSTS, primary_key="id", write_disposition="replace")
    def greenhouse_job_posts():
        count = 0
        for post in _iter_payload(client, "/v3/job_posts"):
            count += 1
            yield _job_post_to_row(post)
        logger.info("Greenhouse: synced %d job post(s).", count)

    @dlt.resource(name=GREENHOUSE_TABLE_SCORECARDS, primary_key="id", write_disposition="replace")
    def greenhouse_scorecards():
        count = 0
        params: dict[str, Any] = {}
        if scorecard_application_ids:
            params["application_ids"] = ",".join(str(i) for i in scorecard_application_ids)
        for scorecard in _iter_payload(client, "/v3/scorecards", params):
            count += 1
            yield _scorecard_to_row(scorecard)
        logger.info("Greenhouse: synced %d scorecard(s).", count)

    @dlt.source(name=GREENHOUSE_SOURCE_NAME)
    def _greenhouse():
        resources: list[Any] = [greenhouse_jobs, greenhouse_job_posts]
        if include_scorecards:
            resources.append(greenhouse_scorecards)
        return resources

    source = _greenhouse()
    # Opt into the document ingestion path (row → text document → cognify).
    # resolve_dlt_sources reads this marker; it never imports this connector.
    from cognee.tasks.ingestion.dlt_utils import DOCUMENT_SOURCE_ATTR

    setattr(source, DOCUMENT_SOURCE_ATTR, GREENHOUSE_SOURCE_NAME)
    return source
