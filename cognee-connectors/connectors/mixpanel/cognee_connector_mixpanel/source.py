"""Mixpanel connector for cognee: a ``dlt`` source that turns a project's analytics definitions into memory.

One source is one Mixpanel project. It yields the project's own knowledge — annotations,
cohort definitions, saved funnel reports and the Lexicon event schema — as flat document
rows, never raw event streams (definitions are knowledge; raw events are noise), and an
opt-in ``events`` kind that exports raw events as daily per-event counts over a date
window. It is meant to be handed directly to :func:`cognee.remember`::

    import cognee
    from cognee_connector_mixpanel import mixpanel_source

    await cognee.remember(
        mixpanel_source(
            resource_name="mixpanel_product",
            service_account_username="<service account username>",
            service_account_secret="<service account secret>",
        ),
        dataset_name="mixpanel",
        primary_key="id",
        write_disposition="merge",   # REQUIRED, the add pipeline defaults to "replace"
        max_rows_per_table=0,
    )

Design
------
* **Rows** are flat ``{id, title, content, url, _deleted, cognee_node_set}``. Ids are
  prefixed by kind (``cohort:<id>``, ``events:<date>:<name>``). Every column this source
  emits feeds the stored document identity, so the rendering and the node set are frozen
  and deterministic: nothing volatile (member counts, computed results) is rendered and
  lists are sorted, except funnel steps, whose order is the definition.
* **Node set** ``mixpanel:<project id>:<kind>`` scopes a ``recall`` to Mixpanel, to one
  kind, or to one project.
* **Incremental** by fingerprint. Mixpanel's definition endpoints carry no change feed and
  no updated-since filter, so every run lists each included kind in full (one request per
  kind) and emits a row only when the hash of its rendered text changed. The fingerprints
  live in dlt resource state. ``events`` is the exception: it is incremental by a
  date-window cursor kept in state, because that is how the export API works.
* **Forget-on-delete.** A definition that stopped appearing in a complete one-request
  listing of its kind is gone: it is emitted as a ``_deleted`` tombstone. There are no
  filters that could hide a listed record, so no per-record re-check is needed. A kind left
  out of ``include`` is forgotten. ``events`` rows are never tombstoned: days do not get
  deleted.
* **A failed run changes nothing.** Each listing is a single request, so there is no
  partial listing to resume and no cut-short machinery: an error raises, dlt drops the
  state of the run, and the next run lists again. Deletions are computed only after every
  included kind listed successfully, so a failed run never forgets.
* **One project per resource_name.** The first run resolves and stores the Mixpanel
  project id; a run under another project with the same ``resource_name`` is refused,
  because it would forget the first project's rows.
* **Credentials** are a service account (username + secret, recommended) or a legacy
  project API secret. They are sent as HTTP Basic auth and live in the client object
  only; errors carry statuses and endpoints, never the credentials. A redirect is an
  error, not followed: ``requests`` drops an ``Authorization`` header on a hop to
  another host, but the check keeps the run honest about what it asked.

Privacy
-------
This connector reads analytics metadata about your product. Cohort definitions describe
groups of users but the connector never reads cohort membership or user profiles.
"""

import base64
import hashlib
import json
import logging
import time
from collections.abc import Callable, Iterable, Iterator
from datetime import date, datetime, timedelta, timezone
from typing import Any

from cognee_connectors_core import dlt_conventions as dlt_utils
from cognee_connectors_core.dlt_conventions import DOCUMENT_SOURCE_ATTR, NODE_SET_COLUMN

logger = logging.getLogger(__name__)

APP_URL = "https://mixpanel.com/project"
# Per-region API bases: the Query API, the App API (Lexicon schemas) and the raw
# data export host. EU-resident projects answer only under eu.mixpanel.com.
REGION_BASES = {
    "us": {
        "query": "https://mixpanel.com/api/2.0",
        "app": "https://mixpanel.com/api/app",
        "export": "https://data.mixpanel.com/api/2.0",
    },
    "eu": {
        "query": "https://eu.mixpanel.com/api/2.0",
        "app": "https://eu.mixpanel.com/api/app",
        "export": "https://data-eu.mixpanel.com/api/2.0",
    },
}
# The definition kinds this source syncs by default; raw events are opt-in only.
KINDS = ("annotations", "cohorts", "funnels", "schemas")
# The kind name of the opt-in raw-event export.
EVENTS_KIND = "events"
# (connect, read) seconds; our own choice, Mixpanel documents no expected latency.
_TIMEOUT = (10, 60)
# Attempts per request on 5xx or network errors, backing off 1s then 2s.
_RETRIES = 3

# The single-request listings of each definition kind: the Query API path, the API base
# it lives on, and whether the project id goes in the query string.
_LISTINGS = {
    "annotations": ("/annotations", "query", True),
    "cohorts": ("/cohorts/list", "query", True),
    "funnels": ("/funnels/list", "query", True),
    "schemas": ("/projects/{project_id}/schemas", "app", False),
}
# The singular id prefix of each kind in row ids and fingerprints.
_KIND_PREFIX = {
    "annotations": "annotation",
    "cohorts": "cohort",
    "funnels": "funnel",
    "schemas": "schema",
    EVENTS_KIND: "events",
}


class MixpanelSourceError(RuntimeError):
    """Base class of the errors this source raises."""


class MixpanelAPIError(MixpanelSourceError):
    """Mixpanel answered with an error, or could not be reached."""


class MixpanelAuthError(MixpanelSourceError):
    """Mixpanel rejected the credentials (HTTP 401)."""


class MixpanelAccessError(MixpanelAPIError):
    """The credentials or their role cannot call an endpoint (HTTP 403)."""


class MixpanelRateLimitedError(MixpanelAPIError):
    """Mixpanel rate limit exceeded (HTTP 429)."""


class _Transient(Exception):
    """One failed attempt worth another: a network error (``response`` None) or a 5xx answer."""

    def __init__(self, response: Any, reason: str):
        super().__init__(reason)
        self.response = response
        self.reason = reason


def _check_credential(name: str, value: str | None, *, allow_empty: bool = False) -> str:
    """Refuse a credential ``requests`` would reject; this keeps it out of exceptions.

    Mixpanel service account usernames and secrets are printable ASCII without
    whitespace; anything else is refused here, before it can reach a header value
    that ``requests`` names in the error it raises. The legacy project API secret
    authenticates as the Basic auth username with an empty password, so an empty
    password is allowed only for that shape.
    """
    text = value or ""
    empty = not text
    if (empty and not allow_empty) or (text and (not text.isascii() or not text.isprintable())):
        raise ValueError(f"The Mixpanel {name} is empty or has an invalid format")
    if any(c.isspace() for c in text):
        raise ValueError(f"The Mixpanel {name} is empty or has an invalid format")
    return text


class MixpanelClient:
    """Minimal synchronous REST client for Mixpanel.

    Authenticates with HTTP Basic auth: a service account (username + secret) or a
    legacy project API secret (as the username, empty password). Tests and hosts can
    pass any object with the same ``request`` method instead: ``request`` takes a
    ``check_active`` keyword and must call it before each request attempt, retries
    and waits included.
    """

    def __init__(
        self,
        username: str,
        secret: str,
        *,
        region: str = "us",
        sleep: Callable[[float], None] = time.sleep,
    ):
        username = _check_credential("service account username", username)
        secret = _check_credential("service account secret", secret, allow_empty=True)
        if region not in REGION_BASES:
            raise ValueError(f"region must be one of {sorted(REGION_BASES)}; got {region!r}")
        credentials = base64.b64encode(f"{username}:{secret}".encode()).decode()
        self._authorization = f"Basic {credentials}"
        self.bases = REGION_BASES[region]
        self._sleep = sleep

    def __repr__(self) -> str:
        return "MixpanelClient(credentials=<redacted>)"

    def request(
        self,
        method: str,
        path: str,
        *,
        base: str = "query",
        params: dict[str, Any] | None = None,
        check_active: Callable[[], None] | None = None,
    ) -> Any:
        """One Mixpanel API call returning the decoded JSON body (always an object here)."""
        import requests

        url = self.bases[base] + path
        headers = {"Authorization": self._authorization, "Accept": "application/json"}

        def attempt() -> Any:
            try:
                response = requests.request(
                    method,
                    url,
                    params=params,
                    headers=headers,
                    timeout=_TIMEOUT,
                    allow_redirects=False,
                )
            except requests.RequestException as exc:
                raise _Transient(None, type(exc).__name__) from None
            if response.status_code >= 500:
                raise _Transient(response, f"HTTP {response.status_code}")
            return self._parse(response, path)

        try:
            return dlt_utils.with_retries(
                attempt,
                retry_on=(_Transient,),
                delay=lambda error, number: 2**number,
                attempts=_RETRIES,
                check_active=check_active,
                sleep=self._sleep,
            )
        except _Transient as error:
            # `from None` keeps the cause out of this error, not out of __context__,
            # which dlt's trace file still reads (dlt 1.30.0, pipeline/trace.py): the
            # credential check in __init__ is what keeps the secret out of a trace.
            if error.response is None:
                raise MixpanelAPIError(f"Mixpanel request failed: {error.reason}") from None
            return self._parse(error.response, path)

    def export_events(
        self,
        from_date: date,
        to_date: date,
        *,
        check_active: Callable[[], None] | None = None,
    ) -> list[dict[str, Any]]:
        """The raw events of the window as a list of ``{event, properties}`` dicts.

        The export API answers with newline-delimited JSON, one event per line, not a
        JSON document; an empty window answers with an empty body.
        """
        import requests

        params = {"from_date": from_date.isoformat(), "to_date": to_date.isoformat()}
        url = self.bases["export"] + "/export"
        headers = {"Authorization": self._authorization, "Accept": "application/x-ndjson"}

        def attempt() -> Any:
            try:
                response = requests.request(
                    "GET",
                    url,
                    params=params,
                    headers=headers,
                    timeout=_TIMEOUT,
                    allow_redirects=False,
                )
            except requests.RequestException as exc:
                raise _Transient(None, type(exc).__name__) from None
            if response.status_code >= 500:
                raise _Transient(response, f"HTTP {response.status_code}")
            return response

        try:
            response = dlt_utils.with_retries(
                attempt,
                retry_on=(_Transient,),
                delay=lambda error, number: 2**number,
                attempts=_RETRIES,
                check_active=check_active,
                sleep=self._sleep,
            )
        except _Transient as error:
            if error.response is None:
                raise MixpanelAPIError(f"Mixpanel export failed: {error.reason}") from None
            response = error.response
        self._raise_for_status(response, "/export")
        events = []
        try:
            for line in response.text.splitlines():
                if not line.strip():
                    continue
                events.append(json.loads(line))
        except ValueError:
            raise MixpanelAPIError("Mixpanel sent an unreadable export answer") from None
        if not all(isinstance(event, dict) for event in events):
            raise MixpanelAPIError("Mixpanel sent an unreadable export answer")
        return events

    def _raise_for_status(self, response: Any, path: str) -> None:
        status = response.status_code
        if 300 <= status < 400:
            raise MixpanelAPIError(f"Mixpanel answered {path} with a redirect (HTTP {status})")
        if status == 401:
            raise MixpanelAuthError("Mixpanel rejected the credentials (HTTP 401)")
        if status == 403:
            raise MixpanelAccessError(f"The Mixpanel credentials cannot call {path} (HTTP 403)")
        if status == 429:
            raise MixpanelRateLimitedError("Mixpanel rate limit exceeded (HTTP 429)")
        if status != 200:
            raise MixpanelAPIError(f"Mixpanel request failed: HTTP {status}")

    def _parse(self, response: Any, path: str) -> Any:
        self._raise_for_status(response, path)
        try:
            payload = response.json()
        except ValueError:
            payload = None
        # Mixpanel answers some Query API listings (annotations, cohorts, funnels)
        # with a bare JSON array, not an object.
        if not isinstance(payload, (dict, list)):
            raise MixpanelAPIError(f"Mixpanel sent an unreadable answer for {path}")
        return payload


def build_mixpanel_service(
    *,
    service_account_username: str | None = None,
    service_account_secret: str | None = None,
    api_secret: str | None = None,
    region: str = "us",
) -> MixpanelClient:
    """Wrap a host-stored credential in a client, the way Apollo does.

    Exactly one credential shape is required: a service account (username + secret)
    or a legacy project API secret.
    """
    if api_secret is not None:
        if service_account_username is not None or service_account_secret is not None:
            raise ValueError("pass either a service account or an api_secret, not both")
        return MixpanelClient(api_secret, "", region=region)
    if not service_account_username or not service_account_secret:
        raise ValueError(
            "build_mixpanel_service needs a service account (service_account_username "
            "and service_account_secret) or an api_secret"
        )
    return MixpanelClient(service_account_username, service_account_secret, region=region)


def mixpanel_project_id(
    service: Any,
    *,
    project_id: str | None = None,
    check_active: Callable[[], None] | None = None,
) -> str:
    """The Mixpanel project a sync will bind to, for a host that names connections.

    Pass an explicit ``project_id`` to skip discovery. Without one, this lists the
    projects the credential can access and requires exactly one, so a sync never
    binds to a guess.
    """
    if project_id:
        return project_id
    data = service.request("GET", "/projects", check_active=check_active)
    projects = (data.get("results") or {}).get("projects") or {}
    if len(projects) != 1:
        raise MixpanelSourceError(
            "The Mixpanel credential can access "
            f"{len(projects)} projects; pass project_id to pick one"
        )
    return next(iter(projects))


# ---------------------------------------------------------------------------
# Rendering: deterministic, definitions only, nothing volatile.
# ---------------------------------------------------------------------------
def _text(value: Any) -> str:
    if isinstance(value, list):
        # A multi-value comes back in no fixed order, and the text decides re-extraction.
        return ", ".join(sorted(text for text in map(_text, value) if text))
    return "" if value is None else str(value).strip()


def _lines(fields: list[tuple[str, str]]) -> list[str]:
    return [f"{label}: {value}" for label, value in fields if value]


def _record_id(kind: str, record: dict[str, Any]) -> str:
    """The stable id of a definition record; ``name`` for schemas, ``id`` elsewhere."""
    key = "name" if kind == "schemas" else "id"
    value = record.get(key)
    if value is None and kind == "funnels":
        value = record.get("funnel_id")
    if not isinstance(value, (str, int)) or isinstance(value, bool):
        raise MixpanelAPIError(f"Mixpanel listed a {kind} record without a readable id")
    return str(value)


def render_annotation(annotation: dict[str, Any], project_id: str) -> dict[str, Any]:
    """An annotation's row: a dated note the team left on its metrics."""
    description = _text(annotation.get("description"))
    day = _text(annotation.get("date"))
    title = description or f"Annotation {day}"
    intro = f"Mixpanel annotation from {day}." if day else "Mixpanel annotation."
    row = {
        "id": f"annotation:{_record_id('annotations', annotation)}",
        "title": title,
        "content": "\n".join([intro, *_lines([("Date", day), ("Description", description)])]),
        "url": f"{APP_URL}/{project_id}",
        "_deleted": False,
    }
    return row


def render_cohort(cohort: dict[str, Any], project_id: str) -> dict[str, Any]:
    """A cohort's row: its definition, never its membership or size."""
    name = _text(cohort.get("name")) or "Unnamed cohort"
    fields = [
        ("Description", _text(cohort.get("description"))),
        ("Created", _text(cohort.get("created"))),
    ]
    return {
        "id": f"cohort:{_record_id('cohorts', cohort)}",
        "title": name,
        "content": "\n".join([f"Mixpanel cohort {name}.", *_lines(fields)]),
        "url": f"{APP_URL}/{project_id}/cohorts",
        "_deleted": False,
    }


def _funnel_steps(funnel: dict[str, Any]) -> list[str]:
    """The event names of a funnel's steps, in order: the order is the definition."""
    steps = funnel.get("steps")
    if not isinstance(steps, list):
        return []
    names = []
    for step in steps:
        if isinstance(step, dict):
            names.append(_text(step.get("event") or step.get("name")))
        else:
            names.append(_text(step))
    return [name for name in names if name]


def render_funnel(funnel: dict[str, Any], project_id: str) -> dict[str, Any]:
    """A saved funnel report's row: its name and step sequence, not its computed numbers."""
    name = _text(funnel.get("name")) or "Unnamed funnel"
    steps = _funnel_steps(funnel)
    fields = [("Steps", "; ".join(steps))]
    return {
        "id": f"funnel:{_record_id('funnels', funnel)}",
        "title": name,
        "content": "\n".join([f"Mixpanel saved report (funnel) {name}.", *_lines(fields)]),
        "url": f"{APP_URL}/{project_id}/funnels",
        "_deleted": False,
    }


def render_schema(schema: dict[str, Any], project_id: str) -> dict[str, Any]:
    """An event schema's row (Lexicon): the event, its description and its properties."""
    name = _text(schema.get("name") or schema.get("schema_name"))
    if not name:
        raise MixpanelAPIError("Mixpanel listed an event schema without a name")
    properties = schema.get("properties")
    if isinstance(properties, dict):
        property_names = sorted(properties)
    elif isinstance(properties, list):
        property_names = sorted(
            _text(p.get("name")) if isinstance(p, dict) else _text(p) for p in properties
        )
        property_names = [p for p in property_names if p]
    else:
        property_names = []
    fields = [
        ("Description", _text(schema.get("description"))),
        ("Properties", ", ".join(property_names)),
    ]
    return {
        "id": f"schema:{name}",
        "title": name,
        "content": "\n".join([f"Mixpanel event schema {name}.", *_lines(fields)]),
        "url": f"{APP_URL}/{project_id}/lexicon",
        "_deleted": False,
    }


_RENDERERS = {
    "annotations": render_annotation,
    "cohorts": render_cohort,
    "funnels": render_funnel,
    "schemas": render_schema,
}


def _node_set(project_id: str, kind: str) -> list[str]:
    """The node set of a row: ``mixpanel:<project id>:<kind>``.

    It is hashed into the document identity like every other column, so changing it gives
    every record of the kind a new ``data_id`` and one more extraction on the tenant's LLM key.
    """
    return [f"mixpanel:{project_id}:{kind}"]


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
class _Walker:
    """One run over the selected kinds, writing progress into ``state``.

    Every definition kind lists in a single request, so a run is all-or-nothing:
    an error raises, dlt drops the run's state, and nothing partial survives.
    """

    def __init__(
        self,
        client: Any,
        state: dict,
        stats: dict[str, int],
        *,
        include: tuple[str, ...],
        project_id: str | None,
        events_from: date | None,
        today: Callable[[], date],
        check_active: Callable[[], None] | None,
    ):
        self.client = client
        self.state = state
        self.stats = stats
        for key in ("scanned", "skipped", "deleted", "failed", "requests"):
            stats.setdefault(key, 0)
        self.include = include
        self.project_id = project_id
        self.events_from = events_from
        self.today = today
        self.check_active = check_active
        self.emitted = 0
        self.fingerprints: dict[str, str] = state.setdefault("fingerprints", {})
        # Local to the run: tombstones are computed only after every included kind
        # listed successfully, so a plain list survives no resume by design.
        self.seen: list[str] = []

    # -- requests ----------------------------------------------------------
    def _call(self, method: str, path: str, **kwargs: Any) -> Any:
        if self.check_active is not None:
            self.check_active()
        self.stats["requests"] += 1
        try:
            return self.client.request(method, path, check_active=self.check_active, **kwargs)
        except MixpanelRateLimitedError:
            self.stats["failed"] += 1
            self.stats["failed_rate_limit"] = 1
            raise
        except MixpanelAuthError:
            self.stats["failed"] += 1
            self.stats["failed_auth"] = 1
            raise

    def _project_id(self) -> str:
        """Resolve and pin the Mixpanel project id, refusing to mix two projects' rows."""
        stored = self.state.get("project_id")
        if self.project_id is not None:
            project_id = self.project_id
        elif stored:
            project_id = stored
        else:
            project_id = mixpanel_project_id(self.client, check_active=self.check_active)
            self.state["project_id"] = project_id
        if stored is not None and stored != project_id:
            raise MixpanelSourceError(
                "This resource_name already holds another Mixpanel project's sync; "
                "give each Mixpanel project its own resource_name"
            )
        return project_id

    # -- listings ----------------------------------------------------------
    def _listing(self, kind: str, project_id: str) -> list[dict[str, Any]]:
        path, base, needs_project = _LISTINGS[kind]
        path = path.format(project_id=project_id)
        params = {"project_id": project_id} if needs_project else None
        data = self._call("GET", path, base=base, params=params)
        key = "schemas" if kind == "schemas" else None
        records = data.get(key) if key else data
        if not isinstance(records, list) or not all(isinstance(r, dict) for r in records):
            raise MixpanelAPIError(f"Mixpanel sent an unreadable answer for {path}")
        return records

    def _changed_rows(
        self, kind: str, records: Iterable[dict[str, Any]], project_id: str
    ) -> Iterator[dict]:
        """Render each record and yield the rows whose fingerprint changed.

        The fingerprint hashes the whole row, so it follows what is rendered: a renamed
        cohort changes its own row and nothing else, because nothing names it.
        """
        for record in records:
            row: dict[str, Any] = {
                **_RENDERERS[kind](record, project_id),
                NODE_SET_COLUMN: _node_set(project_id, kind),
            }
            fingerprint = _hash(row)
            self.seen.append(row["id"])
            self.stats["scanned"] += 1
            if self.fingerprints.get(row["id"]) == fingerprint:
                self.stats["skipped"] += 1
                continue
            self.fingerprints[row["id"]] = fingerprint
            self.emitted += 1
            yield row

    # -- forget-on-delete --------------------------------------------------
    def _forget(self) -> Iterator[dict]:
        """Tombstone every definition that did not appear in this run.

        That is the records of included kinds that disappeared from a complete
        one-request listing — there are no filters that could hide one — and every
        record of a kind left out of ``include``: deselecting a kind leaves memory,
        like Apollo. Event rows carry no tombstones: days are not deleted, and their
        fingerprints only ever change to a corrected count.
        """
        definition_prefixes = {_KIND_PREFIX[kind] for kind in KINDS}
        seen = set(self.seen)
        missing = sorted(
            row_id
            for row_id in self.fingerprints
            if row_id.split(":", 1)[0] in definition_prefixes and row_id not in seen
        )
        for row_id in missing:
            self.fingerprints.pop(row_id, None)
            self.stats["deleted"] += 1
            yield {"id": row_id, "_deleted": True}

    # -- raw events (opt-in) ------------------------------------------------
    def _event_rows(self, project_id: str) -> Iterator[dict]:
        """Daily per-event counts over the window the cursor has not covered yet.

        The window ends the day before the run: today's counts are partial, and a day
        a reader can still change would re-extract every time it does. Late events in a
        closed day raise the count, the fingerprint changes and the day re-emits once.
        """
        cursor = self.state.get("events_cursor")
        start = date.fromisoformat(cursor) + timedelta(days=1) if cursor else self.events_from
        if start is None:
            raise MixpanelSourceError(
                'include "events" on a first run needs events_from (YYYY-MM-DD), the '
                "first day of the export window; later runs advance the cursor themselves"
            )
        end = self.today() - timedelta(days=1)
        if start > end:
            return
        self.stats["requests"] += 1
        events = self.client.export_events(start, end, check_active=self.check_active)
        counts: dict[tuple[str, str], int] = {}
        for event in events:
            if not isinstance(event, dict):
                raise MixpanelAPIError("Mixpanel sent an unreadable export answer")
            name = _text(event.get("event"))
            properties = event.get("properties")
            timestamp = properties.get("time") if isinstance(properties, dict) else None
            if not name or not isinstance(timestamp, (int, float)):
                raise MixpanelAPIError("Mixpanel sent an export event without a name or time")
            day = datetime.fromtimestamp(timestamp, tz=timezone.utc).date().isoformat()
            key = (day, name)
            counts[key] = counts.get(key, 0) + 1
        for day, name in sorted(counts):
            row: dict[str, Any] = {
                "id": f"events:{day}:{name}",
                "title": f"{name} on {day}",
                "content": (
                    f"On {day}, Mixpanel project {project_id} recorded "
                    f"{counts[(day, name)]} '{name}' events."
                ),
                "url": f"{APP_URL}/{project_id}/segmentation",
                "_deleted": False,
                NODE_SET_COLUMN: _node_set(project_id, "events"),
            }
            fingerprint = _hash(row)
            self.stats["scanned"] += 1
            if self.fingerprints.get(row["id"]) == fingerprint:
                self.stats["skipped"] += 1
                continue
            self.fingerprints[row["id"]] = fingerprint
            self.emitted += 1
            yield row
        self.state["events_cursor"] = end.isoformat()

    # -- the run -----------------------------------------------------------
    def rows(self) -> Iterator[dict]:
        project_id = self._project_id()
        for kind in self.include:
            if kind != EVENTS_KIND:
                yield from self._changed_rows(kind, self._listing(kind, project_id), project_id)
        yield from self._forget()
        if EVENTS_KIND in self.include:
            yield from self._event_rows(project_id)


def _validate_include(include: tuple[str, ...]) -> None:
    if not include or set(include) - (set(KINDS) | {EVENTS_KIND}):
        raise ValueError(f"include must be a non-empty subset of {KINDS + (EVENTS_KIND,)}")


def _iter_rows(
    client: Any,
    state: dict,
    stats: dict[str, int],
    *,
    include: tuple[str, ...],
    project_id: str | None = None,
    events_from: date | None = None,
    today: Callable[[], date],
    check_active: Callable[[], None] | None = None,
) -> Iterator[dict]:
    """Yield the changed rows of a project. Pure of dlt, so tests drive it with a dict."""
    _validate_include(include)
    walker = _Walker(
        client,
        state,
        stats,
        include=include,
        project_id=project_id,
        events_from=events_from,
        today=today,
        check_active=check_active,
    )
    try:
        yield from walker.rows()
    except MixpanelSourceError:
        # Every raise counts as a failure; a raise also means dlt drops the run's state,
        # so the next run lists again and nothing is forgotten on a bad run.
        stats["failed"] += 1
        raise


def _as_date(value: str | date | None, name: str) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise ValueError(f"{name} must be an ISO date (YYYY-MM-DD); got {value!r}") from None


def _today_provider(today: Any) -> Callable[[], date]:
    if today is None:
        return date.today
    if isinstance(today, date):
        return lambda: today
    if callable(today):
        return today
    raise ValueError("today must be a date, a callable returning one, or None")


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------
def mixpanel_source(
    *,
    resource_name: str,
    service_account_username: str | None = None,
    service_account_secret: str | None = None,
    api_secret: str | None = None,
    region: str = "us",
    project_id: str | None = None,
    include: tuple[str, ...] | None = None,
    events_from: str | date | None = None,
    today: Any = None,
    service: Any = None,
    check_active: Callable[[], None] | None = None,
):
    """Return a ``dlt`` resource that yields a Mixpanel project's documents for ``remember``.

    Args:
        resource_name: Stable name of this connection, one per Mixpanel project. It
            scopes the sync state and the cleanup of deleted records, so keep it fixed
            across runs. There is no default: two projects sharing a name would forget
            each other's records, and a run refuses state written for another project.
        service_account_username, service_account_secret: A Mixpanel service account
            (Organization Settings > Service Accounts), the recommended credential.
        api_secret: A legacy project API secret, as an alternative to a service account.
        region: ``"us"`` or ``"eu"``, the project's data residency.
        project_id: The Mixpanel project to sync. Without it, the project is discovered
            through ``GET /projects`` and the credential must see exactly one project.
        include: Which kinds to sync. The definition kinds ``"annotations"``,
            ``"cohorts"``, ``"funnels"`` (saved funnel reports) and ``"schemas"``
            (the Lexicon event schema) default to all; ``"events"`` (daily per-event
            counts over a date window) is opt-in, because raw events are noise.
        events_from: First day of the first ``events`` window (YYYY-MM-DD). Required
            only on the first run that includes ``events``; later runs advance the
            cursor in the sync state themselves.
        today: The run's date: a ``date``, a callable returning one, or None for
            ``date.today``. Injectable for tests and backfills.
        service: Pre-built client (see :func:`build_mixpanel_service`). Mainly an
            injection point for tests and hosts. Its ``request`` takes a
            ``check_active`` keyword and must call it before each request attempt,
            retries and waits included.
        check_active: Optional host authorization checkpoint, called before each
            request.

    Returns:
        A ``dlt`` resource configured with ``primary_key="id"``,
        ``write_disposition="merge"`` and a ``_deleted`` hard-delete column.
    """
    import dlt

    dlt_utils.require_document_sync_version(
        2, provider="Mixpanel", feature="reads per-row node sets"
    )
    if not isinstance(resource_name, str) or not resource_name.strip():
        raise ValueError("mixpanel_source needs a resource_name, one per Mixpanel project")
    if api_secret is not None and (
        service_account_username is not None or service_account_secret is not None
    ):
        raise ValueError("pass either a service account or an api_secret, not both")
    if (
        service is None
        and not api_secret
        and not (service_account_username and service_account_secret)
    ):
        raise ValueError(
            "mixpanel_source needs a credential: a service account (service_account_username "
            "and service_account_secret) or an api_secret, or a pre-built service"
        )
    kinds = KINDS if include is None else tuple(include)
    _validate_include(kinds)
    window_start = _as_date(events_from, "events_from")
    day = _today_provider(today)

    # Built here so malformed credentials fail at construction and the closure holds none.
    client = (
        service
        if service is not None
        else build_mixpanel_service(
            service_account_username=service_account_username,
            service_account_secret=service_account_secret,
            api_secret=api_secret,
            region=region,
        )
    )
    stats: dict[str, int] = {}

    @dlt.resource(
        name=resource_name,
        primary_key="id",
        write_disposition="merge",
        # cognee_node_set needs no json hint here: ingest_dlt_source applies
        # DOCUMENT_COLUMN_HINTS to every document source, merged with this one.
        columns={"_deleted": {"data_type": "bool", "hard_delete": True}},
    )
    def mixpanel_project():
        stats.clear()
        rows = _iter_rows(
            client,
            dlt.current.resource_state(),
            stats,
            include=kinds,
            project_id=project_id,
            events_from=window_start,
            today=day,
            check_active=check_active,
        )
        yield from dlt_utils.guarded_rows(rows, check_active)

    resource = mixpanel_project()
    setattr(resource, DOCUMENT_SOURCE_ATTR, "mixpanel")
    setattr(resource, dlt_utils.PIPELINE_SCOPE_ATTR, resource_name)
    resource.cognee_sync_stats = stats
    return resource
