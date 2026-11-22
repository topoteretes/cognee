# Adapted from topoteretes/cognee-community (Apache-2.0), PR #286, commit 0d9c0a7 (itssaharsh).
# Copied from cognee.tasks.ingestion.connectors.vercel; keep source fixes in both copies.
"""Vercel connector for cognee: a ``dlt`` source that turns one Vercel project into memory.

``vercel_source(resource_name=..., project_id=..., team_id=..., access_token=...)`` yields the
project, its deployments and, with ``include_build_logs=True``, the build output of its failed
deployments, as rows ``project:<id>``, ``deployment:<uid>`` and ``build_log:<uid>`` in one
table named ``resource_name``. Hand it to ``cognee.remember`` with ``write_disposition="merge"``.
Sync model, deletion, limits and privacy: the package README.md.
"""

import hashlib
import json
import logging
import re
import time
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote

from cognee_connectors_core import dlt_conventions as dlt_utils
from cognee_connectors_core.dlt_conventions import (
    DOCUMENT_SOURCE_ATTR,
    NODE_SET_COLUMN,
    STRUCTURE_COLUMN,
    with_retries,
)

logger = logging.getLogger(__name__)

BASE_URL = "https://api.vercel.com"
# The deployments list answers at most 100 a page whatever is asked (the community author
# checked it live; Vercel does not document a maximum).
PAGE_SIZE = 100
# Version of the resource state; a later release reads it to migrate an older layout.
STATE_VERSION = 1
# (connect, read) seconds; our own choice, Vercel documents no expected latency.
_TIMEOUT = (10, 30)
# Attempts per request on 5xx or network errors, backing off 1 s then 2 s (Asana's policy).
_RETRIES = 3
_DAY_MS = 24 * 60 * 60 * 1000

PROJECT_ROUTE = "/v9/projects/{project}"
DEPLOYMENTS_ROUTE = "/v7/deployments"
DEPLOYMENT_ROUTE = "/v13/deployments/{deployment}"
EVENTS_ROUTE = "/v3/deployments/{deployment}/events"

# An id that goes into a URL path or a query. Vercel's ids are letters, digits and ``_``.
_ID = re.compile(r"[A-Za-z0-9_-]+")
# Prefixes of the commit keys Vercel puts in a deployment's ``meta``, one per Git provider.
_GIT_PROVIDERS = ("github", "gitlab", "bitbucket")
# Terminal escape sequences: CSI (colour, cursor), OSC closed on the same line, charset
# selections such as ESC ( B, and two-character ones such as ESC M. An unclosed OSC keeps its
# text; only the ESC goes, with the other control characters.
_ESCAPES = re.compile(
    r"\x1b\[[0-?]*[ -/]*[@-~]"
    r"|\x1b\][^\x07\x1b\n]*(?:\x07|\x1b\\)"
    r"|\x1b[ -/]+[0-~]"
    # ESC then any of 0x30-0x7E but [ and ], which open CSI and OSC above.
    r"|\x1b[0-9:;<=>?@A-Z\\^_`a-z{|}~]"
)
# Control characters other than tab and newline. A NUL makes Postgres refuse the whole load.
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
# Millisecond times a deployment or project can carry: from the epoch to the last moment
# ``datetime`` can render.
# A state that names a project but no team key is not one this source wrote.
_UNBOUND = object()
_MAX_MS = int(datetime(9999, 12, 31, 23, 59, tzinfo=timezone.utc).timestamp() * 1000)


# ---------------------------------------------------------------------------
# Errors: messages carry statuses and documented endpoints only.
# ---------------------------------------------------------------------------
class VercelSourceError(RuntimeError):
    """Base class of the errors this source raises."""


class VercelAPIError(VercelSourceError):
    """An error answer from Vercel (``status``), or no answer at all (``status`` 0).

    ``endpoint`` is the route as Vercel documents it, so no id, team id or query reaches the
    message.
    """

    def __init__(self, status: int, endpoint: str):
        answer = f"HTTP {status}" if status else "no answer"
        super().__init__(f"Vercel {endpoint}: {answer}")
        self.status = status
        self.endpoint = endpoint


class VercelAuthError(VercelAPIError):
    """401 or 403: the token is wrong, expired or revoked, or cannot see this team."""


class VercelRateLimitedError(VercelAPIError):
    """HTTP 429. The source never raises it: the run stops with ``failed_rate_limit``."""


class VercelNotFoundError(VercelAPIError):
    """HTTP 404 (not found) or 410 (gone)."""


class VercelResponseError(VercelSourceError):
    """A ``200`` whose body is not of the documented shape. Never read as an empty answer."""


class VercelStateError(VercelSourceError):
    """The resource state was written for another project or team, or by another version.

    One ``resource_name`` belongs to one project; reusing it for another would check the first
    project's deployments against the second forever.
    """


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------
class _BearerAuth:
    """``Authorization: Bearer`` as a requests auth object.

    A header passed in ``headers`` is replaced by a ``~/.netrc`` entry for the host, because
    requests consults netrc when no ``auth`` is given; an auth object turns that off and leaves
    the rest of the environment (proxies, CA bundle) as it is.
    """

    def __init__(self, token: str):
        self._token = token

    def __call__(self, request: Any) -> Any:
        request.headers["Authorization"] = f"Bearer {self._token}"
        return request

    def __repr__(self) -> str:
        return "_BearerAuth(<redacted>)"


class VercelClient:
    """Minimal synchronous REST client for Vercel.

    ``request`` returns the JSON of a ``200`` answer. A ``service`` handed to
    :func:`vercel_source` instead must offer the same ``request`` and call ``check_active``
    before each attempt it makes.
    """

    def __init__(self, access_token: str, *, sleep: Callable[[float], None] = time.sleep):
        # requests names the value of a header it rejects, so every token it would reject is
        # refused here, without echoing it.
        token = access_token.strip() if isinstance(access_token, str) else ""
        if (
            not token
            or not token.isascii()
            or not token.isprintable()
            or any(char.isspace() for char in token)
        ):
            raise ValueError("The Vercel access token is empty or has an invalid format")
        self._access_token = token
        self._sleep = sleep

    def __repr__(self) -> str:
        return "VercelClient(access_token=<redacted>)"

    def request(
        self,
        route: str,
        *,
        ids: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        check_active: Callable[[], None] | None = None,
    ) -> Any:
        """GET ``route`` (``/v13/deployments/{deployment}``) with ``ids`` filled in.

        A 5xx or no answer is retried after 1 s, then 2 s. ``check_active`` runs before every
        attempt and every wait, and what it raises propagates as it is.
        """
        import requests  # lazy: importing the connectors package loads no client library

        path = route.format(**{key: quote(value, safe="") for key, value in (ids or {}).items()})
        auth = _BearerAuth(self._access_token)

        def attempt() -> Any:
            try:
                response = requests.request(
                    "GET",
                    BASE_URL + path,
                    params=params,
                    headers={"Accept": "application/json"},
                    auth=auth,
                    timeout=_TIMEOUT,
                    allow_redirects=False,  # a 3xx is an error answer, never a new target
                )
            except requests.RequestException:
                # Only the fact: a requests error can hold the URL with its query. The error is
                # raised outside this block, so the provider's error is no one's __context__.
                response = None
            if response is None:
                raise VercelAPIError(0, route)
            return self._parse(response, route)

        if check_active is not None:
            check_active()
        return with_retries(
            attempt,
            retry_on=(VercelAPIError,),
            delay=_retry_delay,
            attempts=_RETRIES,
            check_active=check_active,
            sleep=self._sleep,
        )

    @staticmethod
    def _parse(response: Any, route: str) -> Any:
        status = response.status_code
        if status in (401, 403):
            raise VercelAuthError(status, route)
        if status == 429:
            raise VercelRateLimitedError(status, route)
        if status in (404, 410):
            raise VercelNotFoundError(status, route)
        if status != 200:
            raise VercelAPIError(status, route)
        try:
            # JSON is UTF-8 (RFC 8259); requests would guess another charset from the bytes.
            body = json.loads(response.content.decode("utf-8"))
        except (ValueError, RecursionError):  # UnicodeDecodeError is a ValueError
            body = None
        if body is None:
            raise VercelResponseError(f"Vercel {route}: HTTP 200 without a JSON body")
        return body


def _retry_delay(error: BaseException, number: int) -> float | None:
    """No answer or a 5xx is retried after 1 s, then 2 s; every other answer is final."""
    assert isinstance(error, VercelAPIError)  # the only class with_retries is given to retry
    return 2**number if error.status == 0 or error.status >= 500 else None


def build_vercel_service(access_token: str) -> VercelClient:
    """The client the integration layer injects (``_PROVIDERS`` service builder).

    ``access_token`` is a Vercel Integration's OAuth access token on Cloud, or a personal access
    token; both are sent as ``Authorization: Bearer``.
    """
    return VercelClient(access_token)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _clean(text: str) -> str:
    """Drop terminal escapes, control characters (tab and newline stay) and lone surrogates."""
    text = _CONTROL.sub("", _ESCAPES.sub("", text))
    return text.encode("utf-8", "replace").decode("utf-8")


def _line_value(value: Any) -> str:
    """A provider value as one clean line: a commit message cannot forge a line of its own."""
    return " ".join(_clean(str(value)).split())


def _lines(*pairs: tuple[str, Any]) -> list[str]:
    return [
        f"{label}: {_line_value(value)}"
        for label, value in pairs
        if value is not None and _line_value(value)
    ]


def _text(record: dict, key: str) -> str | None:
    """``record[key]`` when it is text, ``None`` when absent or null; another type is an error."""
    value = record.get(key)
    if value is None or isinstance(value, str):
        return value
    raise VercelResponseError(f"Vercel: the field {key} is not text")


def _object(record: dict, key: str) -> dict:
    """``record[key]`` when it is an object, ``{}`` when absent or null; else an error."""
    value = record.get(key)
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    raise VercelResponseError(f"Vercel: the field {key} is not an object")


def _timestamp(milliseconds: int) -> str:
    """A Vercel millisecond timestamp as UTC text, the same way every time."""
    moment = datetime.fromtimestamp(milliseconds / 1000, tz=timezone.utc)
    return moment.strftime("%Y-%m-%d %H:%M UTC")


def _commit(meta: dict, suffix: str) -> str | None:
    """One commit field of ``meta`` (``githubCommitSha`` ...), whichever Git provider set it."""
    for provider in _GIT_PROVIDERS:
        value = _text(meta, f"{provider}{suffix}")
        if value:
            return value
    return None


def _commit_author(meta: dict) -> str | None:
    """``Name (login)`` of the commit author, or whichever of the two is set.

    The name is how sibling connectors (Linear, Slack, Notion, Asana) write a person, so the
    same person merges across them; the login is what a Git source joins on.
    """
    name = _commit(meta, "CommitAuthorName")
    login = _commit(meta, "CommitAuthorLogin")
    if name and login and _line_value(name) != _line_value(login):
        return f"{name} ({login})"
    return login or name


def _repository(link: dict) -> str | None:
    """``<type>: <owner>/<repo>`` of a project's Git link, from its documented shapes."""
    keys = ("org", "repo", "projectNamespace", "projectName", "owner", "slug")
    parts = [value for value in (_text(link, key) for key in keys) if value]
    if not parts:
        return None
    return f"{_text(link, 'type') or 'git'}: {'/'.join(parts)}"


def _environment(deployment: dict) -> str:
    # Vercel reports a preview deployment with a null target.
    target = _text(deployment, "target")
    return target or _text(_object(deployment, "customEnvironment"), "slug") or "preview"


def project_row(project: dict) -> dict[str, Any]:
    """The project's document. Only named fields; ``updatedAt`` and every setting stay out.

    It sits at the top of the structure: the team is not a level, because a structure container
    is scoped by table and each project has its own, so a team node would group nothing.
    """
    name = _line_value(project["name"])
    link = _object(project, "link")
    created = project.get("createdAt")
    if created is not None and not _is_ms(created):
        raise VercelResponseError("Vercel: the field createdAt is not a time")
    lines = _lines(
        ("Project", name),
        ("Framework", _text(project, "framework")),
        ("Git repository", _repository(link)),
        ("Production branch", _text(link, "productionBranch")),
        ("Root directory", _text(project, "rootDirectory")),
        ("Node.js version", _text(project, "nodeVersion")),
        ("Created", _timestamp(created) if created is not None else None),
    )
    return {
        "id": f"project:{project['id']}",
        "title": f"Project {name}",
        "content": "\n".join(lines),
        "url": None,
        "_deleted": False,
        NODE_SET_COLUMN: ["project"],
        STRUCTURE_COLUMN: {"ancestors": []},
    }


def deployment_row(deployment: dict, project_id: str) -> dict[str, Any]:
    """A deployment's document: the facts a question names first, the uid in the title.

    ``name`` is the project's name when the deployment was built, so a project rename
    re-renders no deployment; it also names the parent in the structure, which search copies
    into a cited chunk as ``document_parent``.
    """
    meta = _object(deployment, "meta")
    creator = _object(deployment, "creator")
    uid = deployment["uid"]
    name = _line_value(deployment["name"])
    environment = _line_value(_environment(deployment))
    created = _timestamp(deployment["created"])
    state = deployment["readyState"]
    substate = _text(deployment, "readySubstate")
    host = _text(deployment, "url")
    inspector = _text(deployment, "inspectorUrl")
    lines = _lines(
        ("Project", name),
        ("Environment", environment),
        ("State", f"{state} ({substate})" if substate else state),
        ("Branch", _commit(meta, "CommitRef")),
        ("Commit", _commit(meta, "CommitSha")),
        ("Commit message", _commit(meta, "CommitMessage")),
        ("Commit author", _commit_author(meta)),
        ("Created by", _text(creator, "username")),
        ("Error code", _text(deployment, "errorCode")),
        ("Error message", _text(deployment, "errorMessage")),
        ("Created", created),
        ("Deployment URL", f"https://{host}" if host else None),
    )
    return {
        "id": f"deployment:{uid}",
        "title": f"{name}: {environment} deployment {created} ({uid})",
        "content": "\n".join(lines),
        "url": _clean(inspector) if inspector else None,
        "_deleted": False,
        NODE_SET_COLUMN: ["deployment"],
        STRUCTURE_COLUMN: {
            "ancestors": [
                {"kind": "project", "id": f"project:{project_id}", "name": name, "document": True}
            ]
        },
    }


def event_text(event: dict) -> str:
    """The log line of a build event, from either documented shape, as clean text.

    Carriage returns become line breaks (progress lines end with ``\\r``).
    """
    text = _text(event, "text")
    if text is None:
        text = _text(_object(event, "payload"), "text")
    if text is None:
        return ""  # a documented event without a line (a delimiter)
    text = _ESCAPES.sub("", text).replace("\r\n", "\n").replace("\r", "\n")
    return _clean(text).strip("\n")


def build_log_row(deployment: dict, events: list[dict]) -> dict[str, Any] | None:
    """The build output of one failed deployment, every line in order, or ``None`` when empty."""
    text = "\n".join(line for line in (event_text(event) for event in events) if line)
    if not text:
        return None
    uid = deployment["uid"]
    name = _line_value(deployment["name"])
    header = _lines(("Project", name), ("Deployment", uid), ("State", deployment["readyState"]))
    return {
        "id": f"build_log:{uid}",
        "title": f"Build output of failed deployment {uid} of {name}",
        "content": "\n".join([*header, "", text]),
        "url": _clean(deployment["inspectorUrl"]) if deployment.get("inspectorUrl") else None,
        "_deleted": False,
        NODE_SET_COLUMN: ["build_log"],
        STRUCTURE_COLUMN: {
            "ancestors": [
                {
                    "kind": "deployment",
                    "id": f"deployment:{uid}",
                    "name": f"{name} deployment {uid}",
                    "document": True,
                }
            ]
        },
    }


def tombstone(row_id: str) -> dict[str, Any]:
    return {"id": row_id, "_deleted": True}


def _fingerprint(row: dict[str, Any]) -> str:
    """Stable across processes, over every column the row is stored with."""
    return hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()[:16]


def _is_ms(value: Any) -> bool:
    """A finite, whole number of milliseconds ``datetime`` can render (no NaN, no infinity)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    if isinstance(value, float) and not value.is_integer():  # NaN and infinity are not either
        return False
    return 0 <= value <= _MAX_MS


# ---------------------------------------------------------------------------
# Walker
# ---------------------------------------------------------------------------
class _CutShort(Exception):
    """Internal: stop the run here, keeping what it emitted.

    ``resumes`` is False when the next run cannot get further on its own (a project of another
    account), so the log does not promise it.
    """

    def __init__(self, reason: str, *, resumes: bool = True):
        super().__init__(reason)
        self.reason = reason
        self.resumes = resumes


def new_stats() -> dict[str, int]:
    """The counts of a run, every key present from its start so a host finds them on any exit.

    ``failed`` means this project's run failed or was cut, and is always the sum of the
    ``failed_*`` keys. A cut (``rate_limit``, ``budget``, ``auth``, ``unavailable``) and a
    raised run end the run, so each happens at most once a run. A raised run is
    ``failed_response`` (a ``200`` that is not JSON or of no documented shape), ``failed_error``
    (another error answer: a 3xx, another 4xx, a 404 on the listing) or ``failed_state`` (state
    written for another project, team or version).

    Not failures of the run: ``check_errors`` (deletion checks that could not be read) and
    ``build_log_errors`` (build outputs that could not be read) count items the pass kept as
    they were and asks again next pass; a host that blocks retiring other projects' tables on
    ``failed`` never sees them. ``unresolved`` is 1 when the project answered 404 or 410 and
    nothing was read or forgotten.
    """
    return dict.fromkeys(
        (
            "scanned",
            "skipped",
            "deleted",
            "failed",
            "unresolved",
            "failed_rate_limit",
            "failed_budget",
            "failed_auth",
            "failed_unavailable",
            "failed_response",
            "failed_error",
            "failed_state",
            "check_errors",
            "build_log_errors",
        ),
        0,
    )


def _deployment(item: Any, route: str, project_id: str) -> dict:
    """A listed deployment of the documented shape and of this project, else an error."""
    if (
        not isinstance(item, dict)
        or not isinstance(item.get("uid"), str)
        or not _ID.fullmatch(item["uid"])
        or not isinstance(item.get("readyState"), str)
        or not _is_ms(item.get("created"))
        or not isinstance(item.get("name"), str)
    ):
        raise VercelResponseError(f"Vercel {route}: a deployment without its uid, state or time")
    if item.get("projectId") != project_id:
        raise VercelResponseError(f"Vercel {route}: a deployment of another project")
    return item


def _single_read_is_live(answer: Any, uid: str, project_id: str) -> bool:
    """Whether a ``200`` of ``/v13/deployments/{idOrUrl}`` is this live deployment.

    This endpoint answers in its own shape, not the list's: ``id`` (not ``uid``) and
    ``createdAt``, and its minimal variant has neither ``projectId`` nor ``createdAt``. Only
    ``id`` and ``readyState`` are required in every variant, and its ``readyState`` has no
    ``DELETED``: a deployment that is gone answers 404 or 410. ``False`` means an answer of no
    documented shape, or one about another deployment or project.
    """
    if (
        not isinstance(answer, dict)
        or answer.get("id") != uid
        or not isinstance(answer.get("readyState"), str)
    ):
        return False
    return "projectId" not in answer or answer["projectId"] == project_id


class _Walker:
    """One run over one project, writing its progress into ``state`` (a plain dict).

    A pass over the listing can span runs: ``state["pass"]`` keeps the page position reached
    (``until``), the deployments the pass has seen, whether the listing is complete, and the
    deletion checks left. Each run continues the pass where the last one stopped, so a request
    budget smaller than the listing still gets through it over several runs. Deletion checks
    run only once the pass has listed everything.
    """

    def __init__(
        self,
        client: Any,
        state: dict,
        stats: dict[str, int],
        *,
        project_id: str,
        team_id: str | None,
        account_id: str,
        since_ms: int | None,
        include_build_logs: bool,
        max_requests: int | None,
        check_active: Callable[[], None] | None,
    ):
        self.client = client
        self.state = state
        self.stats = stats
        self.project_id = project_id
        self.team_id = team_id
        self.account_id = account_id
        self.since_ms = since_ms
        self.include_build_logs = include_build_logs
        self.max_requests = max_requests
        self.check_active = check_active
        self.used = 0
        # Rows handed out so far this run (``_iter_rows`` counts them as they leave).
        self.emitted = 0

    def _call(
        self, route: str, *, ids: dict[str, str] | None = None, params=None, cut: bool = True
    ) -> Any:
        """One request. ``cut=False`` leaves a 5xx or no answer to the caller."""
        if self.max_requests is not None and self.used >= self.max_requests:
            raise _CutShort("budget")
        self.used += 1
        query = dict(params or {})
        if self.team_id:
            query["teamId"] = self.team_id
        try:
            # check_active is not caught here: a revoked connection is not an item failure.
            return self.client.request(route, ids=ids, params=query, check_active=self.check_active)
        except VercelRateLimitedError:
            raise _CutShort("rate_limit") from None
        except VercelAuthError:
            if not self.emitted:
                # Nothing to keep yet: a bad or expired token is an error the caller sees, not
                # an empty success.
                raise
            raise _CutShort("auth") from None
        except VercelAPIError as error:
            if cut and (error.status == 0 or error.status >= 500):
                raise _CutShort("unavailable") from None
            raise

    def _emit(self, row: dict[str, Any], known: dict[str, str]) -> dict[str, Any] | None:
        """``row`` when it differs from what was last emitted under its id, else ``None``."""
        self.stats["scanned"] += 1
        fingerprint = _fingerprint(row)
        if known.get(row["id"]) == fingerprint:
            self.stats["skipped"] += 1
            return None
        known[row["id"]] = fingerprint
        return row

    def _forget(self, uid: str) -> Iterator[dict]:
        """Tombstones for a deployment Vercel says is gone, and for its build output."""
        for row_id in (f"deployment:{uid}", f"build_log:{uid}"):
            if self.known.pop(row_id, None) is not None:
                self.stats["deleted"] += 1
                yield tombstone(row_id)
        self.created.pop(uid, None)
        self.logs.pop(uid, None)

    def rows(self) -> Iterator[dict[str, Any]]:
        version = self.state.get("v", STATE_VERSION)
        if version != STATE_VERSION:
            raise VercelStateError(f"Vercel: state version {version!r} is not one this reads")
        bound = (self.state.get("project_id"), self.state.get("team_id", _UNBOUND))
        if "project_id" in self.state and bound != (self.project_id, self.team_id):
            raise VercelStateError(
                "Vercel: this resource_name's state belongs to another project or team; use one "
                "resource_name per project"
            )
        try:
            project = self._call(PROJECT_ROUTE, ids={"project": self.project_id})
        except VercelNotFoundError:
            # Deleted, or out of this token's reach: the API does not tell them apart, so
            # nothing is forgotten: lost access must not erase memory. The documents stay until
            # the project is deselected.
            self.stats["unresolved"] = 1
            logger.warning("Vercel: the project answered 404 or 410; nothing was read.")
            return
        if not isinstance(project, dict) or not isinstance(project.get("name"), str):
            raise VercelResponseError(f"Vercel {PROJECT_ROUTE}: a project without its name")
        if project.get("id") != self.project_id:
            raise VercelResponseError(
                f"Vercel {PROJECT_ROUTE}: answered for another id; pass the project id "
                "(prj_...), not its name"
            )
        if project.get("accountId") != self.account_id:
            logger.warning(
                "Vercel: the project belongs to another account than the expected one; nothing "
                "was read. Check team_id and account_id."
            )
            raise _CutShort("auth", resumes=False)

        self.state["v"] = STATE_VERSION
        self.state["project_id"] = self.project_id
        self.state["team_id"] = self.team_id
        self.known: dict[str, str] = self.state.setdefault("rows", {})
        self.created: dict[str, int] = self.state.setdefault("created", {})
        # uid -> {"name", "url", "status"}: the failed deployments whose build output is to be
        # read ("queued"), was refused this pass ("retry") or is empty or gone ("none").
        self.logs: dict[str, dict] = self.state.setdefault("logs", {})

        row = self._emit(project_row(project), self.known)
        if row is not None:
            yield row
        if not self.include_build_logs:
            # Build output was switched off: it leaves memory like a deselected scope.
            for row_id in sorted(key for key in self.known if key.startswith("build_log:")):
                del self.known[row_id]
                self.stats["deleted"] += 1
                yield tombstone(row_id)
            self.logs.clear()

        fresh = "pass" not in self.state
        if fresh:
            # Build output refused in the last pass is asked again, once per pass.
            for entry in self.logs.values():
                if entry["status"] == "retry":
                    entry["status"] = "queued"
            self.state["pass"] = {"until": None, "top": None, "seen": [], "listed": False}
        current = self.state["pass"]
        if not fresh:
            # A pass started in an earlier run: read the newest deployments first, so a new one
            # or a state change at the top never waits behind the rest of the pass.
            yield from self._head(current)
        if not current["listed"]:
            yield from self._listing(current)
        yield from self._build_logs()
        if current["listed"]:
            yield from self._check_missing(current)
            self._finish_pass()

    def _page(self, params: dict, previous: int | None) -> tuple[list[dict], int | None]:
        """One listing page and the ``until`` of the next one (``None`` when it is the last)."""
        body = self._call(DEPLOYMENTS_ROUTE, params=params)
        items = body.get("deployments") if isinstance(body, dict) else None
        pagination = body.get("pagination") if isinstance(body, dict) else None
        if not isinstance(items, list) or not isinstance(pagination, dict):
            raise VercelResponseError(
                f"Vercel {DEPLOYMENTS_ROUTE}: an answer without deployments or pagination"
            )
        if "next" not in pagination:
            raise VercelResponseError(f"Vercel {DEPLOYMENTS_ROUTE}: pagination without next")
        deployments = [_deployment(item, DEPLOYMENTS_ROUTE, self.project_id) for item in items]
        following = pagination["next"]
        if following is None:
            return deployments, None
        if not _is_ms(following):
            raise VercelResponseError(f"Vercel {DEPLOYMENTS_ROUTE}: a next that is no time")
        # ``next`` is the time of the page's last item and ``until`` is exclusive, so a second
        # deployment of that same millisecond would fall between two pages: one millisecond
        # later reads it, and the repeat is dropped by uid.
        cursor = int(following) + 1
        if previous is not None and cursor >= previous:
            raise VercelResponseError(f"Vercel {DEPLOYMENTS_ROUTE}: the cursor did not move")
        return deployments, cursor

    def _params(self, until: int | None) -> dict[str, Any]:
        params: dict[str, Any] = {"projectId": self.project_id, "limit": PAGE_SIZE}
        if self.since_ms is not None:
            params["since"] = self.since_ms
        if until is not None:
            params["until"] = until
        return params

    def _listing(self, current: dict) -> Iterator[dict[str, Any]]:
        """Continue the pass's listing from the page it reached; the position is saved per page."""
        seen = set(current["seen"])
        until = current["until"]
        while True:
            deployments, following = self._page(self._params(until), until)
            if current["top"] is None and deployments:
                current["top"] = max(int(d["created"]) for d in deployments)
            for deployment in deployments:
                if deployment["uid"] not in seen:
                    seen.add(deployment["uid"])
                    yield from self._settle(deployment)
            current["seen"] = list(seen)
            if following is None:
                current["listed"] = True
                current["pending"] = self._missing(seen)
                return
            until = following
            current["until"] = until

    def _head(self, current: dict) -> Iterator[dict[str, Any]]:
        """The newest deployments, down to the newest one the pass started from.

        Under a request budget a head page is read only while at least one more request is left
        after it, so the rest of the pass always moves on: with ``max_requests=2`` a continued
        run reads no head, and what the head would have shown waits for the next pass. A head
        read to its end moves ``top`` to the newest deployment it saw, so the next run reads
        only what is newer (and its first page), not the same block again.
        """
        top = current["top"]
        newest = top
        until = None
        while self.max_requests is None or self.max_requests - self.used >= 2:
            deployments, following = self._page(self._params(until), until)
            for deployment in deployments:
                yield from self._settle(deployment)
                if newest is None or deployment["created"] > newest:
                    newest = int(deployment["created"])
            current["seen"] = list(set(current["seen"]) | {d["uid"] for d in deployments})
            reached = top is None or not deployments or deployments[-1]["created"] <= top
            if following is None or reached:
                current["top"] = newest
                return
            until = following

    def _settle(self, deployment: dict) -> Iterator[dict[str, Any]]:
        uid = deployment["uid"]
        if deployment["readyState"] == "DELETED":
            yield from self._forget(uid)
            return
        self.created[uid] = int(deployment["created"])
        row = self._emit(deployment_row(deployment, self.project_id), self.known)
        if row is not None:
            yield row
        if (
            self.include_build_logs
            and deployment["readyState"] == "ERROR"
            and f"build_log:{uid}" not in self.known
            and uid not in self.logs
        ):
            # Read after the listing pages, so a cut events read never holds a page back. The
            # state keeps the values the row uses, cleaned: a raw provider string (a lone
            # surrogate) cannot be stored by dlt, and the run would fail on every sync.
            inspector = _text(deployment, "inspectorUrl")
            self.logs[uid] = {
                "name": _line_value(deployment["name"]),
                "url": _clean(inspector) if inspector else None,
                "status": "queued",
            }

    def _build_logs(self) -> Iterator[dict[str, Any]]:
        """Read the queued build output, each deployment settling its own entry.

        A deployment's output is read at most once per pass. One that answers 5xx, nothing,
        another 4xx, or a 200 that is not JSON or of the wrong shape is counted in
        ``build_log_errors`` (not a failure of the run) and asked again in the next pass, so it
        never holds back the rest. A 401, 403 or 429 still stops the run. A finished build with
        no output (404, 410 or no lines) is recorded and never asked again.
        """
        for uid in sorted(u for u, entry in self.logs.items() if entry["status"] == "queued"):
            entry = self.logs[uid]
            if uid not in self.created or self._left_window(self.created[uid]):
                del self.logs[uid]  # asked for only while its deployment is in the window
                continue
            try:
                events = self._call(
                    EVENTS_ROUTE,
                    ids={"deployment": uid},
                    params={"limit": -1, "follow": 0},
                    cut=False,
                )
            except VercelNotFoundError:
                events = []
            except VercelAuthError:
                raise  # nothing emitted yet: the caller sees the rejected token
            except (VercelAPIError, VercelResponseError):  # an error, or a 200 that is not JSON
                events = None
            deployment = {
                "uid": uid,
                "name": entry["name"],
                "readyState": "ERROR",
                "inspectorUrl": entry["url"],
            }
            try:
                if not isinstance(events, list) or not all(isinstance(e, dict) for e in events):
                    raise VercelResponseError(f"Vercel {EVENTS_ROUTE}: an answer of no event list")
                row = build_log_row(deployment, events)
            except VercelResponseError:
                entry["status"] = "retry"
                self.stats["build_log_errors"] += 1
                continue
            if row is None:
                entry["status"] = "none"
                continue
            del self.logs[uid]
            row = self._emit(row, self.known)
            if row is not None:
                yield row

    def _left_window(self, created_ms: int) -> bool:
        return self.since_ms is not None and created_ms < self.since_ms

    def _missing(self, seen: set[str]) -> list[str]:
        return sorted(
            uid
            for uid, when in self.created.items()
            if uid not in seen and (self.since_ms is None or when >= self.since_ms)
        )

    def _check_missing(self, current: dict) -> Iterator[dict[str, Any]]:
        """Read each known deployment of the window the completed pass did not list.

        Missing from a listing is not gone: only Vercel saying so (404 or 410) forgets it. Each
        read settles its own deployment, so a run cut between two reads keeps the tombstones it
        already emitted and the next run asks only the rest. A read that fails (5xx, no answer,
        another 4xx, a 200 that is not JSON or of no documented shape) keeps the deployment,
        counts ``check_errors`` (not a failure of the run) and moves on: it is asked again in the
        next pass, and never holds the pass open. A 401, 403 or 429 still stops the run.
        """
        pending = current["pending"]
        while pending:
            uid = pending[0]
            if uid in self.created:
                try:
                    answer = self._call(DEPLOYMENT_ROUTE, ids={"deployment": uid}, cut=False)
                except VercelNotFoundError:
                    yield from self._forget(uid)
                except VercelAuthError:
                    raise  # nothing emitted yet: the caller sees the rejected token
                except (VercelAPIError, VercelResponseError):  # an error, or a 200 that is not JSON
                    self.stats["check_errors"] += 1
                else:
                    if not _single_read_is_live(answer, uid, self.project_id):
                        self.stats["check_errors"] += 1
            pending.pop(0)

    def _finish_pass(self) -> None:
        if self.since_ms is not None:
            # Out of the window: never checked again, never forgotten for it; staging keeps it.
            # A build log's fingerprint stays, so switching build output off still forgets it.
            for uid in [uid for uid, when in self.created.items() if when < self.since_ms]:
                del self.created[uid]
                self.known.pop(f"deployment:{uid}", None)
        # Build output is asked for only while its deployment is in the window: an entry left
        # waiting (to be retried, or recorded as having no output) goes with it.
        for uid in [uid for uid in self.logs if uid not in self.created]:
            del self.logs[uid]
        del self.state["pass"]


def _iter_rows(
    client: Any,
    state: dict,
    stats: dict[str, int],
    *,
    project_id: str,
    team_id: str | None,
    account_id: str,
    since_ms: int | None = None,
    include_build_logs: bool = False,
    max_requests: int | None = None,
    check_active: Callable[[], None] | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield the changed rows of one project. Pure of dlt, so tests drive it with a dict."""
    walker = _Walker(
        client,
        state,
        stats,
        project_id=project_id,
        team_id=team_id,
        account_id=account_id,
        since_ms=since_ms,
        include_build_logs=include_build_logs,
        max_requests=max_requests,
        check_active=check_active,
    )
    try:
        for row in walker.rows():
            walker.emitted += 1
            yield row
    except _CutShort as cut:
        stats["failed"] += 1
        stats[f"failed_{cut.reason}"] += 1
        if cut.resumes:
            logger.warning(
                "Vercel sync stopped early (%s); it resumes on the next run.", cut.reason
            )
    except VercelSourceError as error:
        # The run raises and dlt drops its state; a host still reads these counts.
        stats["failed"] += 1
        stats[_raised_reason(error)] += 1
        raise


def _raised_reason(error: VercelSourceError) -> str:
    if isinstance(error, VercelAuthError):
        return "failed_auth"
    if isinstance(error, VercelResponseError):
        return "failed_response"
    if isinstance(error, VercelStateError):
        return "failed_state"
    return "failed_error"


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------
def _check_resource_name(name: Any) -> None:
    """A non-blank string that is not a name dlt keeps for itself."""
    from dlt.common.normalizers.naming.snake_case import NamingConvention

    if not isinstance(name, str) or not name.strip():
        raise ValueError("vercel_source needs a resource_name, one per Vercel connection")
    if NamingConvention().normalize_table_identifier(name).startswith("_dlt_"):
        raise ValueError("resource_name must not start with _dlt_, which dlt keeps for itself")


def _check_id(value: Any, what: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError(f"vercel_source needs {what} as an id string (letters, digits, _ or -)")
    return value


def vercel_source(
    *,
    resource_name: str,
    project_id: str,
    team_id: str | None = None,
    account_id: str | None = None,
    access_token: str | None = None,
    service: Any = None,
    check_active: Callable[[], None] | None = None,
    max_requests: int | None = None,
    lookback_days: int | None = None,
    include_build_logs: bool = False,
):
    """Return a ``dlt`` resource that yields one Vercel project's documents for ``remember``.

    Args:
        resource_name: Stable name of this connection and project. It names the staging table
            and scopes the sync state, so two projects or two users must never share one. There
            is no default.
        project_id: The project to sync, by id (``prj_...``), never by name. Required.
        team_id: The team that owns the project, sent as ``teamId`` on every call. ``None`` for
            a personal (Hobby) account.
        account_id: The account the project must belong to (its ``accountId``). Defaults to
            ``team_id``; a personal account passes its user id. A project of any other account
            stops the run with ``failed_auth`` before anything is yielded.
        access_token: A Vercel access token, sent as ``Authorization: Bearer``. Used to build
            the client when ``service`` is omitted. Never read from the environment. A token
            Vercel refuses (401, 403) raises ``VercelAuthError`` while the run has emitted no
            row; after a row the run stops with ``failed_auth``, keeps its rows and resumes.
        service: Pre-built client (see :func:`build_vercel_service`), the injection point for
            hosts and tests. It must accept ``request(route, *, ids, params, check_active)`` and
            call ``check_active`` before each attempt it makes.
        check_active: Host authorization checkpoint, called before each request attempt and
            each retry wait (by the client) and around each row. What it raises stops the run
            as it is.
        max_requests: Request budget of one run: ``None`` (the default) means none, an int of
            at least 2 (the project and one page) bounds the run, and ``0`` defers it (no
            request, no state touched, no row; stats ``failed_budget``, as a cut). A run that
            spends its budget stops with ``failed_budget`` and keeps what it emitted; the next
            run continues the pass where it stopped. A retried call counts once. ``1`` is
            refused: it reads the project and never moves the pass.
        lookback_days: Read only the deployments created in this many days. It bounds reading,
            never memory: a deployment that leaves the window keeps its document and is no
            longer checked for deletion. ``None`` (the default) reads every deployment Vercel
            lists.
        include_build_logs: Also read the build output of failed deployments: one request per
            failed deployment, once; output that is empty or gone is not asked again, one that
            fails is asked again in the next pass. Off by default: a build can print a
            secret. Switching it off again forgets the build output already synced.

    Returns:
        A ``dlt`` resource configured with ``primary_key="id"``, ``write_disposition="merge"``
        and a ``_deleted`` hard-delete column.
    """
    import dlt

    dlt_utils.require_document_sync_version(3, provider="Vercel", feature="reads per-row structure")
    _check_resource_name(resource_name)
    project = _check_id(project_id, "project_id")
    team = None if team_id is None else _check_id(team_id, "team_id")
    if account_id is None and team is None:
        raise ValueError(
            "vercel_source needs team_id, or account_id (the user id) for a personal account"
        )
    account = _check_id(team if account_id is None else account_id, "account_id")
    if max_requests is not None and (
        not isinstance(max_requests, int)
        or isinstance(max_requests, bool)
        or max_requests < 0
        or max_requests == 1
    ):
        # One request reads the project and nothing else, so a run could never progress; 0
        # defers the run (a host with no budget left for this project still settles it).
        raise ValueError(
            "max_requests must be None, 0 (defer this run) or an int of at least 2 "
            "(the project and a page)"
        )
    if lookback_days is not None and (
        not isinstance(lookback_days, int) or isinstance(lookback_days, bool) or lookback_days < 1
    ):
        raise ValueError("lookback_days must be a positive int, or None")
    if not isinstance(include_build_logs, bool):
        raise ValueError("include_build_logs must be True or False")

    # Built here so a missing or malformed token fails at construction (the client checks it)
    # and the closure holds no raw token.
    client = service if service is not None else VercelClient(access_token or "")
    # Every key from the start, so a host that reads the counts before the first row finds them.
    stats: dict[str, int] = new_stats()

    @dlt.resource(
        name=resource_name,
        primary_key="id",
        write_disposition="merge",
        columns={
            "_deleted": {"data_type": "bool", "hard_delete": True},
            # dlt creates no column that is None in every row; without the hint a later load
            # that has a url would add the column and change every older row's identity.
            "url": {"data_type": "text", "nullable": True},
        },
    )
    def vercel_documents():
        stats.clear()
        stats.update(new_stats())
        if max_requests == 0:
            # Deferred: no request, no state read or written (a pass in progress stays as it
            # is), no row. A cut by the budget, so a host keeps the project and resumes it.
            stats.update(failed=1, failed_budget=1)
            logger.warning("Vercel sync deferred (no request budget this run); it resumes later.")
            return
        since = None
        if lookback_days is not None:
            # The local clock only places the window. It bounds reading and the deletion check
            # alike, so a skewed clock moves what is read and never forgets anything.
            since = int(time.time() * 1000) - lookback_days * _DAY_MS
        rows = _iter_rows(
            client,
            dlt.current.resource_state(),
            stats,
            project_id=project,
            team_id=team,
            account_id=account,
            since_ms=since,
            include_build_logs=include_build_logs,
            max_requests=max_requests,
            check_active=check_active,
        )
        yield from dlt_utils.guarded_rows(rows, check_active)

    resource = vercel_documents()
    # The _PROVIDERS key, the document source tag and the Cloud key: host retire finds the
    # tables by it.
    setattr(resource, DOCUMENT_SOURCE_ATTR, "vercel")
    setattr(resource, dlt_utils.PIPELINE_SCOPE_ATTR, resource_name)
    resource.cognee_sync_stats = stats
    return resource
