# Adapted from topoteretes/cognee-community (Apache-2.0), PR #286, commit 0d9c0a7 (itssaharsh).
# Modified to ship in the Cognee SDK with SDK-local imports and test paths.
"""In-memory stand-in for the Vercel REST API, shaped like its documented answers.

Shapes follow Vercel's OpenAPI document (read 2026-11-11): ``GET /v9/projects/{idOrName}``
answers the project object (env values, deploy hooks and bypass secrets included),
``GET /v7/deployments`` answers ``{deployments, pagination: {count, next, prev}}`` newest first,
paged with ``until`` (exclusive) at ``next``, the time of the page's last item; ``GET
/v13/deployments/{idOrUrl}`` answers one deployment in that endpoint's own shape (``id`` and
``createdAt``, not the list's ``uid`` and ``created``; its minimal variant, served for the uids in
``minimal_reads``, carries no ``projectId`` and no ``createdAt``; its ``readyState`` has no
``DELETED``) or 404/410; ``GET /v3/deployments/{idOrUrl}/events`` answers a list of events whose
line is ``text`` or ``payload.text``, and must be asked with ``follow=0`` (otherwise it
streams). Every call must carry the fake's ``teamId`` (none for a personal account), or it
answers 403, as Vercel does for a token used without its team. An endpoint the connector
does not use is not answered: a call to it raises ``AssertionError``.

``FakeVercel.request`` is the ``service=`` surface; ``serve_over_http`` answers at the level of
``requests.request``, so the real client (header, retries, error mapping) runs.
"""

import json
from types import SimpleNamespace
from urllib.parse import quote, unquote

import requests

from cognee_connector_vercel.source import BASE_URL, VercelAPIError, VercelClient

# 2026-11-10 12:00:00 UTC
NOW_MS = 1_794_312_000_000
MINUTE_MS = 60_000
DAY_MS = 24 * 60 * MINUTE_MS

# Values that must never leave the process. Each sits in a field the real API returns next to
# the fields the connector does read.
CANARY_ENV = "CANARY-ENV-VALUE-0001"
CANARY_HOOK = "CANARYHOOKSECRET0002"
CANARY_BYPASS = "CANARYBYPASSSECRET0003"
CANARY_META = "CANARY-META-VALUE-0004"
CANARY_EMAIL = "canary-0005@example.com"
CANARY_IP = "203.0.113.77"
CANARY_COMMAND = "CANARY-BUILD-COMMAND-0006"
CANARY_INVITE = "CANARY-INVITE-CODE-0007"
CANARIES = (
    CANARY_ENV,
    CANARY_HOOK,
    CANARY_BYPASS,
    CANARY_META,
    CANARY_EMAIL,
    CANARY_IP,
    CANARY_COMMAND,
    CANARY_INVITE,
)


class Resp:
    """The part of ``requests.Response`` the client reads: the status and the body's bytes.

    ``payload`` is a JSON value, raw ``bytes``, or an exception, which stands for a body that
    is not JSON (an HTML error page).
    """

    def __init__(self, status: int, payload=None):
        self.status_code = status
        self._payload = payload

    @property
    def content(self) -> bytes:
        if isinstance(self._payload, Exception):
            return b"<html>Bad gateway</html>"
        if isinstance(self._payload, bytes):
            return self._payload
        return json.dumps(self._payload).encode("utf-8")


def sent_authorization(kwargs: dict) -> str | None:
    """The Authorization header a ``requests.request`` call would send, auth object applied."""
    request = SimpleNamespace(headers=dict(kwargs.get("headers") or {}))
    if kwargs.get("auth") is not None:
        kwargs["auth"](request)
    return request.headers.get("Authorization")


def sha(n: int) -> str:
    """A full 40-character commit sha."""
    return f"{n:040x}"


class FakeVercel:
    def __init__(self, *, team_id: str | None = "team_acme", page_size: int | None = None):
        self.team_id = team_id
        self.account_id = team_id or "user_ada"
        self.page_size = page_size  # None: the ``limit`` the caller asks for
        self.projects: dict[str, dict] = {}
        self.deployments: dict[str, dict] = {}
        self.events: dict[str, list] = {}
        self.calls: list[tuple[str, dict]] = []  # (path, params) of every request
        # [path prefix, status or None for no answer, times left or None for always]
        self.failures: list[list] = []
        self.answers: dict[str, object] = {}  # path -> body answered with 200 instead
        self.quota: int | None = None  # requests served before every one answers 429
        self.unlisted: set[str] = set()  # live deployments the listing leaves out
        self.gone: dict[str, int] = {}  # uid -> 404 or 410 on its own read, never listed
        self.stuck_cursor = False  # every page names the same ``next``
        self.minimal_reads: set[str] = set()  # uids whose own read answers the minimal variant

    # -- seeding ------------------------------------------------------------
    def add_project(self, project_id: str = "prj_web", name: str = "web", **fields) -> dict:
        project = {
            "id": project_id,
            "name": name,
            "accountId": self.account_id,
            "framework": "nextjs",
            "nodeVersion": "22.x",
            "rootDirectory": "apps/web",
            "createdAt": NOW_MS - 90 * DAY_MS,
            "updatedAt": NOW_MS,
            "buildCommand": CANARY_COMMAND,
            "installCommand": CANARY_COMMAND,
            "env": [{"id": "env_1", "key": "API_KEY", "type": "plain", "value": CANARY_ENV}],
            "link": {
                "type": "github",
                "org": "acme",
                "repo": name,
                "productionBranch": "main",
                "gitCredentialId": "cred_1",
                "deployHooks": [
                    {
                        "id": "hook_1",
                        "name": "nightly",
                        "ref": "main",
                        "url": f"https://api.vercel.com/v1/integrations/deploy/x/{CANARY_HOOK}",
                    }
                ],
            },
            "protectionBypass": {CANARY_BYPASS: {"scope": "automation-bypass"}},
            "trustedIps": {"addresses": [{"value": CANARY_IP}]},
            "latestDeployments": [{"id": "dpl_latest", "meta": {"token": CANARY_META}}],
            "targets": {"production": {"id": "dpl_latest", "createdAt": NOW_MS}},
            "creator": {"email": CANARY_EMAIL, "username": "ada"},
            "inviteCode": CANARY_INVITE,
            **fields,
        }
        self.projects[project_id] = project
        return project

    def add_deployment(
        self,
        uid: str,
        project_id: str = "prj_web",
        *,
        state: str = "READY",
        minutes_ago: int = 10,
        target: str | None = "production",
        commit: int = 1,
        **fields,
    ) -> dict:
        created = NOW_MS - minutes_ago * MINUTE_MS
        name = self.projects[project_id]["name"] if project_id in self.projects else "web"
        deployment = {
            "uid": uid,
            "name": name,
            "projectId": project_id,
            "url": f"{name}-{uid.lower()}.vercel.app",
            "inspectorUrl": f"https://vercel.com/acme/{name}/{uid}",
            "state": state,
            "readyState": state,
            "type": "LAMBDAS",
            "target": target,
            "created": created,
            "createdAt": created,
            "buildingAt": created + 1000,
            "ready": created + 60_000,
            "creator": {"uid": "user_1", "username": "ada", "email": CANARY_EMAIL},
            "meta": {
                "githubCommitSha": sha(commit),
                "githubCommitRef": "main",
                "githubCommitMessage": f"Commit {commit}",
                "githubCommitAuthorLogin": "ada",
                "githubCommitAuthorName": "Ada Lovelace",
                "githubCommitAuthorEmail": CANARY_EMAIL,
                "token": CANARY_META,
            },
            **fields,
        }
        self.deployments[uid] = deployment
        return deployment

    def set_log(self, uid: str, *lines: str) -> None:
        self.events[uid] = [{"type": "stdout", "created": NOW_MS, "text": line} for line in lines]

    # -- changes the way Vercel makes them ------------------------------------
    def update(self, uid: str, **fields) -> None:
        self.deployments[uid].update(fields)

    def set_state(self, uid: str, state: str, **fields) -> None:
        self.deployments[uid].update(state=state, readyState=state, **fields)

    def delete(self, uid: str) -> None:
        """Deleted for good: no longer listed, its own read answers 404."""
        del self.deployments[uid]
        self.gone[uid] = 404

    def expire(self, uid: str) -> None:
        """Retention removed it after its recovery period: not listed, its read answers 410."""
        del self.deployments[uid]
        self.gone[uid] = 410

    def rename_project(self, project_id: str, name: str) -> None:
        self.projects[project_id]["name"] = name

    def fail(self, prefix: str, status: int | None, times: int | None = None) -> None:
        self.failures.append([prefix, status, times])

    def heal(self) -> None:
        self.failures.clear()
        self.quota = None

    # -- the API --------------------------------------------------------------
    def answer(self, path: str, params: dict) -> tuple[int | None, object]:
        """``(status, body)`` for one GET; status ``None`` means no answer at all."""
        self.calls.append((path, dict(params)))
        for failure in self.failures:
            prefix, status, times = failure
            if path.startswith(prefix) and (times is None or times > 0):
                if times is not None:
                    failure[2] -= 1
                return status, {"error": {"code": "forbidden", "message": "No way"}}
        if self.quota is not None:
            if self.quota <= 0:
                return 429, {"error": {"code": "rate_limited"}}
            self.quota -= 1
        if params.get("teamId") != self.team_id:
            return 403, {"error": {"code": "forbidden", "message": "Not authorized"}}
        if path in self.answers:
            return 200, self.answers[path]
        parts = path.strip("/").split("/")
        if parts[:2] == ["v9", "projects"] and len(parts) == 3:
            project = self.projects.get(unquote(parts[2]))
            return (200, project) if project else (404, {"error": {"code": "not_found"}})
        if parts == ["v7", "deployments"]:
            return 200, self._listing(params)
        if parts[:2] == ["v13", "deployments"] and len(parts) == 3:
            uid = unquote(parts[2])
            if uid in self.gone:
                return self.gone[uid], {"error": {"code": "not_found"}}
            deployment = self.deployments.get(uid)
            if deployment is None:
                return 404, {"error": {"code": "not_found"}}
            return 200, self._single(deployment)
        if parts[:2] == ["v3", "deployments"] and len(parts) == 4 and parts[3] == "events":
            uid = unquote(parts[2])
            if uid not in self.deployments:
                return 404, {"error": {"code": "not_found"}}
            assert params.get("limit") == -1, "every line of the build is asked for"
            assert params.get("follow") == 0, "a finite read, never the live stream"
            return 200, list(self.events.get(uid, []))
        raise AssertionError(f"the connector called an endpoint the fake does not serve: {path}")

    def _single(self, deployment: dict) -> dict:
        """``/v13/deployments/{idOrUrl}``: ``id`` and ``createdAt``, not ``uid`` or ``created``."""
        if deployment["uid"] in self.minimal_reads:
            return {
                "id": deployment["uid"],
                "readyState": deployment["readyState"],
                "aliasAssigned": True,
                "duration": 1,
                "target": deployment.get("target"),
                "errorCode": deployment.get("errorCode"),
                "errorMessage": deployment.get("errorMessage"),
            }
        answer = {k: v for k, v in deployment.items() if k not in ("uid", "created")}
        return {"id": deployment["uid"], **answer, "createdAt": deployment["created"]}

    def _listing(self, params: dict) -> dict:
        project_id = params["projectId"]
        limit = int(self.page_size or params["limit"])
        items = [
            d
            for d in self.deployments.values()
            if d["projectId"] == project_id and d["uid"] not in self.unlisted
        ]
        if "since" in params:
            items = [d for d in items if d["created"] >= params["since"]]
        if "until" in params:
            items = [d for d in items if d["created"] < params["until"]]
        items.sort(key=lambda d: (d["created"], d["uid"]), reverse=True)
        page = items[:limit]
        following = page[-1]["created"] if len(items) > limit else None
        if self.stuck_cursor and page:
            following = page[0]["created"] + 10
        return {
            "deployments": [dict(d) for d in page],
            "pagination": {"count": len(page), "next": following, "prev": None},
        }

    def request(self, route, *, ids=None, params=None, check_active=None):
        """The ``service=`` surface: the real client's answer mapping, without its transport."""
        if check_active is not None:
            check_active()
        path = route.format(**{key: quote(value, safe="") for key, value in (ids or {}).items()})
        status, body = self.answer(path, params or {})
        if status is None:
            raise VercelAPIError(0, route)
        return VercelClient._parse(Resp(status, body), route)

    def paths(self, prefix: str = "") -> list[str]:
        return [path for path, _ in self.calls if path.startswith(prefix)]


def serve_over_http(monkeypatch, fake: FakeVercel, token: str) -> list[dict]:
    """Answer ``requests.request`` from ``fake``; return the keyword arguments of every call."""
    sent: list[dict] = []

    def fake_request(method, url, **kwargs):
        sent.append({"method": method, "url": url, **kwargs})
        assert method == "GET"
        assert url.startswith(BASE_URL)
        if sent_authorization(kwargs) != f"Bearer {token}":
            return Resp(401, {"error": {"code": "forbidden", "message": "Not authorized"}})
        status, body = fake.answer(url[len(BASE_URL) :], kwargs.get("params") or {})
        if status is None:
            raise requests.ConnectionError(f"connection to {url} refused")
        return Resp(status, body)

    monkeypatch.setattr("requests.request", fake_request)
    return sent
