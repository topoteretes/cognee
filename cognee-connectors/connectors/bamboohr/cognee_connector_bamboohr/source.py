"""BambooHR connector for cognee: a ``dlt`` source that turns an employee roster into memory.

One source is one BambooHR company (``subdomain``) and one explicit field allowlist.
It yields one document per employee as flat rows and is meant to be handed directly
to :func:`cognee.remember`::

    import cognee
    from cognee_connector_bamboohr import bamboohr_source

    await cognee.remember(
        bamboohr_source(
            resource_name="bamboohr_acme",
            subdomain="acme",
            api_key="<bamboohr api key>",
            fields=["firstName", "lastName", "jobTitle", "department", "location", "status"],
        ),
        dataset_name="bamboohr",
        primary_key="id",
        write_disposition="merge",   # REQUIRED, the add pipeline defaults to "replace"
        max_rows_per_table=0,
    )

Design
------
* **The field allowlist is required, never defaulted.** Employee data is sensitive,
  so ``fields`` names exactly the BambooHR fields the run reads — there is no "sync
  everything" mode. Every request (the report and each deletion check) asks for the
  allowlisted fields and nothing else; fields outside the allowlist are never fetched.
  The allowlist is checked against ``GET /meta/fields`` on every run: a name BambooHR
  does not list stops the run before it emits or forgets anything.
* **Rows** are flat ``{id, title, content, url, _deleted, cognee_node_set}``. Ids are
  prefixed by kind (``employee:<id>``). Every column feeds the stored document
  identity, so the rendering is frozen and deterministic: the allowlisted values in
  the sorted allowlist order, nothing volatile added on top. A tombstone carries no
  node set.
* **Node set** ``bamboohr:<subdomain>`` scopes a ``recall`` to one company. It is
  hashed into the document identity, so changing it re-extracts every employee.
* **Incremental** by fingerprint. BambooHR has no change feed, so every run asks for
  one custom report of the whole company with exactly the allowlisted fields and a
  row is emitted only when the hash of its rendered text changed. The fingerprints
  live in dlt resource state.
* **Forget-on-delete.** An id that stopped appearing in a complete report is checked
  once by reading the employee: a 404 is emitted as a ``_deleted`` tombstone. Any
  other answer — a 200 (the report can miss an employee the API still serves), a
  403, a 5xx or no answer — keeps the record.
* **A record whose check keeps failing** backs off: after five failed checks in a
  row it is left alone for ten cycles, so an employee BambooHR keeps refusing does
  not cost a request and a ``failed`` count on every run. ``held_back`` counts the
  records sitting out.
* **A run is bounded and resumable.** On a rate limit (HTTP 429), a spent request
  budget or a rejected key after rows were emitted, the source stops cleanly, keeps
  the state it reached and reports the reason in ``cognee_sync_stats``
  (``failed_rate_limit``, ``failed_budget``, ``failed_auth``). The next run finishes
  the pending deletion checks first. It never sleeps on a 429: ``dlt`` rolls back
  the state of a failed run and the walk holds the process-wide staging lock.
* **BambooHR asks for at most one request per second.** The client paces every
  request (a retry included) at least ``_MIN_REQUEST_INTERVAL`` seconds after the
  previous one, so a deletion-check tail is slow by design; ``max_requests`` bounds
  a run and the rest resumes next run.
* **One company per resource_name.** The first run stores the subdomain, and a run
  under another subdomain with the same ``resource_name`` is refused, because it
  would forget the first company's rows.
* **Key** is a plain BambooHR API key, sent as the user name of an HTTP Basic pair
  (password ``x``). It lives in the client object only; errors carry statuses and
  endpoints, never the key or an employee id. A redirect is an error, not followed:
  ``requests`` drops ``Authorization`` on a hop to another host, and the safe side
  of that behavior is to never test it.

Privacy
-------
This connector reads personal data about your own organisation's employees. It is
**opt-in**: nothing is fetched until you construct a source and call ``remember``,
and it fetches only the fields you name. What it reads goes to your configured LLM
and embedding providers, and anyone with read access to the target dataset can see
it, so keep the dataset private to people allowed to see the HR data.
"""

import base64
import hashlib
import json
import logging
import re
import time
from collections.abc import Callable, Iterable, Iterator
from typing import Any

from cognee_connectors_core import dlt_conventions as dlt_utils
from cognee_connectors_core.dlt_conventions import DOCUMENT_SOURCE_ATTR, NODE_SET_COLUMN

logger = logging.getLogger(__name__)

BASE_URL = "https://api.bamboohr.com/api/gateway.php"
# The employee profile page in BambooHR's web app, used as the row's url.
APP_URL = "https://{subdomain}.bamboohr.com/employees/view/{employee_id}"
# Budget of the requests of one run, our own choice, not an API limit.
DEFAULT_MAX_REQUESTS = 500
# BambooHR asks callers to limit themselves to one request per second; the client
# paces requests at least this many seconds apart, a retry included.
_MIN_REQUEST_INTERVAL = 1.1
# (connect, read) seconds; our own choice, BambooHR documents no expected latency.
_TIMEOUT = (10, 30)
# Attempts per request on 5xx or network errors, backing off 1s then 2s.
_RETRIES = 3
# Failed deletion checks in a row after which a record backs off. Chosen, not measured.
_MAX_CHECK_FAILURES = 5
# Cycles a record in back-off is left alone before its check is tried again.
_BACKOFF_CYCLES = 10
# Version of the resource state; a later release reads it to migrate an older layout.
STATE_VERSION = 1
# An employee id goes into the URL of its own read, so no ``.``, ``/``, ``?``, ``#``
# or ``%`` may get through. Ids are digits; only the character set is checked.
_EMPLOYEE_ID = re.compile(r"[0-9A-Za-z]+")
# A BambooHR subdomain names the company in the API path and the app URL.
_SUBDOMAIN = re.compile(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?")
# A field alias goes into a query string on the deletion check, so nothing but
# letters, digits, ``_`` and ``-`` may get through.
_FIELD = re.compile(r"[A-Za-z0-9_-]+")


class BambooHRSourceError(RuntimeError):
    """Base class of the errors this source raises."""


class BambooHRAPIError(BambooHRSourceError):
    """An error answer from BambooHR (``status``), or no answer at all (``status`` 0).

    The message names the status and the endpoint; never the key, an employee id or
    BambooHR's own explanation.
    """

    def __init__(self, status: int, endpoint: str):
        answer = f"HTTP {status}" if status else "no answer"
        super().__init__(f"BambooHR {endpoint}: {answer}")
        self.status = status
        self.endpoint = endpoint


class BambooHRAuthError(BambooHRAPIError):
    """BambooHR rejected the API key (HTTP 401)."""

    def __init__(self, endpoint: str):
        super().__init__(401, endpoint)


class BambooHRAccessError(BambooHRAPIError):
    """The API key cannot call an endpoint (HTTP 403)."""

    def __init__(self, endpoint: str):
        super().__init__(403, endpoint)


class BambooHRRateLimitedError(BambooHRAPIError):
    """BambooHR answered HTTP 429."""

    def __init__(self, endpoint: str):
        super().__init__(429, endpoint)


class BambooHRNotFoundError(BambooHRAPIError):
    """The employee was deleted or never existed (HTTP 404)."""

    def __init__(self, endpoint: str):
        super().__init__(404, endpoint)


class BambooHRResponseError(BambooHRSourceError):
    """BambooHR answered ``200`` with a body that is not of the expected shape.

    Never read as an empty report or a deleted employee: on the report or the field
    list it raises and aborts the run, on the read of one employee it keeps the
    record, because a body that cannot be read cannot prove the record is gone.
    """


# The count key of an error that ends a run by raising, matched with isinstance. A
# record read that is refused never gets here: _should_forget counts it and goes on.
_RAISED_REASONS = {BambooHRAuthError: "failed_auth", BambooHRAccessError: "failed_access"}


def _endpoint(path: str) -> str:
    """The path with an employee id replaced, so error messages never carry one."""
    return re.sub(r"^/employees/[^/]+", "/employees/{id}", path)


def _employee_path(employee_id: str) -> str:
    """The path that reads one employee, for an id that came out of a BambooHR answer.

    Raises for an id that is not alphanumeric. The error names the endpoint, never
    the id. It guards the request only: the id stays as it is in the row id and the
    ``url`` column, where it is never requested.
    """
    if not _EMPLOYEE_ID.fullmatch(employee_id):
        raise BambooHRAPIError(
            0, "/employees/{id} (an id with characters other than letters and digits)"
        )
    return f"/employees/{employee_id}/"


class _Transient(Exception):
    """One failed attempt worth another: a network error (``response`` None) or a 5xx."""

    def __init__(self, response: Any, reason: str):
        super().__init__(reason)
        self.response = response
        self.reason = reason


class BambooHRClient:
    """Minimal synchronous REST client for BambooHR.

    BambooHR authenticates with an API key as the user name of an HTTP Basic pair
    (password ``x``) and asks for at most one request per second, so every request
    (a retry included) is paced at least ``_MIN_REQUEST_INTERVAL`` seconds after the
    previous one. Tests and hosts can pass any object with the same ``request``
    method instead: ``request(method, path, *, params, body, check_active)`` that
    takes ``check_active`` and calls it before each attempt it makes.
    """

    def __init__(
        self,
        api_key: str,
        subdomain: str,
        *,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ):
        # This check is what keeps the key out of an exception: requests names the value of a
        # header it rejects, and every key it would reject is refused here without echoing it.
        key = api_key.strip() if isinstance(api_key, str) else ""
        if (
            not key
            or not key.isascii()
            or not key.isprintable()
            or any(char.isspace() for char in key)
        ):
            raise ValueError("The BambooHR API key is empty or has an invalid format")
        if not isinstance(subdomain, str) or not _SUBDOMAIN.fullmatch(subdomain):
            raise ValueError(f"The BambooHR subdomain {subdomain!r} is not a valid subdomain")
        self._api_key = key
        self._subdomain = subdomain
        self._sleep = sleep
        self._monotonic = monotonic
        self._last_request_at: float | None = None

    def __repr__(self) -> str:
        return "BambooHRClient(api_key=<redacted>)"

    def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
        check_active: Callable[[], None] | None = None,
    ) -> Any:
        """The JSON payload of a ``200`` answer, retrying a 5xx or a network error.

        ``check_active`` is called before each retry and its backoff sleep
        (``with_retries``), so a host that has revoked the run does not wait out the
        retries. What it raises propagates as it is: it is no network error, so it is
        neither retried nor wrapped.
        """
        import requests  # lazy: importing the connectors package loads no client library

        headers = {
            # The key is the user name of the pair, the password is the literal ``x``.
            "Authorization": "Basic " + base64.b64encode(f"{self._api_key}:x".encode()).decode(),
            "Accept": "application/json",
        }

        def attempt() -> Any:
            self._pace()
            try:
                response = requests.request(
                    method,
                    f"{BASE_URL}/{self._subdomain}/v1{path}",
                    params=params,
                    json=body,
                    headers=headers,
                    timeout=_TIMEOUT,
                    allow_redirects=False,  # a 3xx is an error answer, never a new target
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
            # The last attempt failed the same way: a 5xx answer is read like any other
            # answer, a network error is reported by type.
            if error.response is None:
                raise BambooHRAPIError(0, _endpoint(path)) from None
            return self._parse(error.response, path)

    def _pace(self) -> None:
        """Wait until at least ``_MIN_REQUEST_INTERVAL`` seconds after the previous request."""
        now = self._monotonic()
        if self._last_request_at is not None:
            wait = _MIN_REQUEST_INTERVAL - (now - self._last_request_at)
            if wait > 0:
                self._sleep(wait)
        self._last_request_at = self._monotonic()

    def _parse(self, response: Any, path: str) -> Any:
        status, endpoint = response.status_code, _endpoint(path)
        if 300 <= status < 400:
            raise BambooHRAPIError(status, endpoint)
        if status == 401:
            raise BambooHRAuthError(endpoint)
        if status == 403:
            raise BambooHRAccessError(endpoint)
        if status == 429:
            raise BambooHRRateLimitedError(endpoint)
        if status == 404:
            raise BambooHRNotFoundError(endpoint)
        if status != 200:
            raise BambooHRAPIError(status, endpoint)
        try:
            return response.json()
        except ValueError:
            raise BambooHRResponseError(f"BambooHR {endpoint}: HTTP 200 without JSON") from None


def build_bamboohr_service(api_key: str, subdomain: str) -> BambooHRClient:
    """Wrap a host-owned API key and subdomain in a client, the way Apollo does.

    A ``service`` handed to :func:`bamboohr_source` instead must offer
    ``request(method, path, *, params, body, check_active)`` and call ``check_active``
    before each attempt it makes.
    """
    return BambooHRClient(api_key, subdomain)


# ---------------------------------------------------------------------------
# Rendering: the allowlisted fields only, in a frozen deterministic order.
# ---------------------------------------------------------------------------
def _text(value: Any) -> str:
    if isinstance(value, list):
        # A multi-value field comes back in no fixed order, and the text decides
        # re-extraction.
        return ", ".join(sorted(text for text in map(_text, value) if text))
    return "" if value is None else str(value).strip()


def render_employee(
    employee_id: str,
    record: dict[str, Any],
    *,
    fields: list[str],
    labels: dict[str, str],
    subdomain: str,
) -> dict[str, Any]:
    """An employee's row: one sentence, then one ``Label: value`` line per set field.

    ``fields`` is the sorted allowlist; a field with no value renders no line, so two
    records that differ only in an empty allowlisted field keep one text. Nothing
    volatile is added (no sync timestamp), so an unchanged employee keeps its content
    hash and is not extracted again.
    """
    first, last = _text(record.get("firstName")), _text(record.get("lastName"))
    name = " ".join(part for part in (first, last) if part)
    lines = [
        f"{labels.get(field, field)}: {_text(record.get(field))}"
        for field in fields
        if _text(record.get(field))
    ]
    if name:
        title, intro = name, f"BambooHR employee {name}."
    else:
        title = f"Employee {employee_id}"
        intro = f"BambooHR employee {employee_id} has no name in the synced fields."
    return {
        "id": f"employee:{employee_id}",
        "title": title,
        "content": "\n".join([intro, *lines]),
        "url": APP_URL.format(subdomain=subdomain, employee_id=employee_id),
        "_deleted": False,
    }


def _node_set(subdomain: str) -> list[str]:
    """The node set of a row: ``bamboohr:<subdomain>``.

    It is hashed into the document identity like every other column, so changing it
    gives every employee a new ``data_id`` and one more extraction on the tenant's LLM
    key.
    """
    return [f"bamboohr:{subdomain}"]


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
class _CutShort(Exception):
    """Internal: stop the run here, keeping the state reached so far."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class _Walker:
    """One run over the company, writing progress into ``state``.

    BambooHR has no change feed, so every run asks for one custom report with exactly
    the allowlisted fields, emits the rows whose fingerprint changed, and re-checks the
    ids that stopped appearing one by one before it forgets them.
    """

    def __init__(
        self,
        client: Any,
        state: dict,
        stats: dict[str, int],
        *,
        subdomain: str,
        fields: list[str],
        max_requests: int,
        check_active: Callable[[], None] | None,
    ):
        self.client = client
        self.state = state
        self.stats = stats
        self.subdomain = subdomain
        self.fields = fields
        self.max_requests = max_requests
        self.check_active = check_active
        self.emitted = 0
        self.fingerprints: dict[str, str] = state.setdefault("fingerprints", {})
        self.seen: list[str] = state.setdefault("seen", [])
        # Per employee whose deletion check failed: {"failed": checks failed in a row,
        # "wait": cycles still to sit out}. Dropped as soon as the employee is listed
        # again, checked successfully or forgotten, so it only holds stuck records.
        self.check_failures: dict[str, dict[str, int]] = state.setdefault("check_failures", {})
        # Ids whose deletion check ran in this run. A run that resumes a cut one checks
        # its pending ids first, and the same ids come up again when the report closes:
        # checking them twice would count one run's failure twice.
        self.checked: set[str] = set()
        self.kept: set[str] = set()  # ids whose check failed this run
        stats["held_back"] = sum(self._in_back_off(row_id) for row_id in self.check_failures)

    # -- requests ----------------------------------------------------------
    def _call(
        self, method: str, path: str, *, params: dict[str, Any] | None = None, body: Any = None
    ) -> Any:
        # Before the budget, so a revoked run ends as revoked, not as a resumable cut.
        if self.check_active is not None:
            self.check_active()
        if self.stats["requests"] >= self.max_requests:
            raise _CutShort("budget")
        # A call the client retries on a 5xx or a network error still counts once: the
        # budget and this count are about what the run asks for, and the client owns
        # its retries.
        self.stats["requests"] += 1
        try:
            return self.client.request(
                method, path, params=params, body=body, check_active=self.check_active
            )
        except BambooHRRateLimitedError:
            raise _CutShort("rate_limit") from None
        except BambooHRAuthError:
            if self.emitted:
                raise _CutShort("auth") from None
            raise

    # -- the allowlist -----------------------------------------------------
    def _load_labels(self) -> dict[str, str]:
        """The display label of every allowlisted field, and a check of the allowlist.

        ``GET /meta/fields`` lists every field BambooHR serves, standard and custom,
        under its ``alias``; the report and the deletion checks ask for fields by that
        alias. A name BambooHR does not list — a typo, a renamed custom field — stops
        the run with a :class:`BambooHRSourceError` naming it, before it emits or
        forgets anything: a report that fails on an unknown field could look empty,
        and every synced employee would seem deleted.
        """
        payload = self._call("GET", "/meta/fields")
        if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
            raise BambooHRResponseError("BambooHR /meta/fields: not a list of fields")
        labels: dict[str, str] = {}
        for item in payload:
            alias = item.get("alias")
            if isinstance(alias, str) and alias:
                name = item.get("name")
                labels[alias] = name if isinstance(name, str) and name else alias
        unknown = set(self.fields) - labels.keys()
        if unknown:
            raise BambooHRSourceError(
                "fields names BambooHR does not list: "
                f"{sorted(unknown)}. GET /meta/fields lists the field aliases."
            )
        return labels

    # -- listing -----------------------------------------------------------
    def _report(self) -> list[dict[str, Any]]:
        """Every employee of the company, with exactly the allowlisted fields.

        The custom report is one request and one answer; there is no pagination. A 200
        of the wrong shape raises :class:`BambooHRResponseError` instead of reading as
        an empty company (which would forget everyone).
        """
        payload = self._call(
            "POST",
            "/employees/custom-report",
            params={"format": "JSON"},
            body={"fields": self.fields},
        )
        if not isinstance(payload, list):
            raise BambooHRResponseError("BambooHR /employees/custom-report: not a list of rows")
        rows = []
        for row in payload:
            if not isinstance(row, dict):
                raise BambooHRResponseError(
                    "BambooHR /employees/custom-report: a row is not an object"
                )
            employee_id = row.get("employeeId")
            if isinstance(employee_id, int) and not isinstance(employee_id, bool):
                employee_id = str(employee_id)
            if not isinstance(employee_id, str) or not employee_id:
                raise BambooHRResponseError(
                    "BambooHR /employees/custom-report: a row without a usable employeeId"
                )
            rows.append(row)
        return rows

    def _changed_rows(
        self, rows: Iterable[dict[str, Any]], labels: dict[str, str]
    ) -> Iterator[dict[str, Any]]:
        """Render each employee and yield the rows whose fingerprint changed.

        The fingerprint hashes the whole row, so it follows what is rendered: changing
        the allowlist re-renders every employee once, and a renamed department changes
        the text of its employees.
        """
        for row in rows:
            employee_id = str(row["employeeId"])
            document: dict[str, Any] = {
                **render_employee(
                    employee_id, row, fields=self.fields, labels=labels, subdomain=self.subdomain
                ),
                NODE_SET_COLUMN: _node_set(self.subdomain),
            }
            fingerprint = _hash(document)
            self.seen.append(document["id"])
            self.stats["scanned"] += 1
            if self.fingerprints.get(document["id"]) == fingerprint:
                self.stats["skipped"] += 1
                continue
            self.fingerprints[document["id"]] = fingerprint
            self.emitted += 1
            yield document

    # -- forget-on-delete --------------------------------------------------
    def _close_cycle(self) -> None:
        """Queue ids not seen in a complete report, then start a fresh cycle."""
        self.state["pending_deletes"] = self._due(set(self.fingerprints) - set(self.seen))
        self.seen.clear()

    def _due(self, missing: set[str]) -> list[str]:
        """The missing ids to check this run. A record in back-off sits the cycle out."""
        for row_id in set(self.check_failures) - missing:
            del self.check_failures[row_id]
        due = []
        for row_id in sorted(missing):
            if row_id in self.checked:
                continue
            if self._in_back_off(row_id):
                self.check_failures[row_id]["wait"] -= 1
            else:
                due.append(row_id)
        return due

    def _in_back_off(self, row_id: str) -> bool:
        """Whether the employee's deletion check is left alone until its wait is over.

        A back-off record is not a failure, so ``failed`` stays 0 for it and a sync
        that only holds records back reads as healthy; ``held_back`` says otherwise.
        """
        streak = self.check_failures.get(row_id)
        return bool(streak and streak["wait"] > 0)

    def _forget(self) -> Iterator[dict[str, Any]]:
        pending = self.state.setdefault("pending_deletes", [])
        while pending:
            row_id = pending[0]
            if self._should_forget(row_id):
                self.fingerprints.pop(row_id, None)
                self.stats["deleted"] += 1
                yield {"id": row_id, "_deleted": True}
            pending.pop(0)

    def _should_forget(self, row_id: str) -> bool:
        """Whether the employee is gone. A check that fails keeps the record and counts."""
        self.checked.add(row_id)
        try:
            gone = self._gone(row_id)
        except BambooHRNotFoundError:
            self.check_failures.pop(row_id, None)
            return True
        except BambooHRAccessError as exc:
            self._check_failed(row_id, "failed_access", exc)
            return False
        except (BambooHRAPIError, BambooHRResponseError) as exc:
            self._check_failed(row_id, "failed_deletion_check", exc)
            return False
        self.check_failures.pop(row_id, None)
        return gone

    def _check_failed(self, row_id: str, reason: str, exc: Exception) -> None:
        self.stats["failed"] += 1
        self.stats[reason] = self.stats.get(reason, 0) + 1
        self.kept.add(row_id)
        streak = self.check_failures.setdefault(row_id, {"failed": 0, "wait": 0})
        streak["failed"] += 1
        if streak["failed"] >= _MAX_CHECK_FAILURES:
            streak["wait"] = _BACKOFF_CYCLES
        # The message names the status and endpoint, never the employee id.
        logger.warning(
            "BambooHR deletion check failed for one employee, %d in a row; it is kept: %s",
            streak["failed"],
            exc,
        )

    def _gone(self, row_id: str) -> bool:
        employee_id = row_id.removeprefix("employee:")
        record = self._call(
            "GET",
            _employee_path(employee_id),
            params={"fields": ",".join(self.fields)},
        )
        # A 404 raised by _call means gone. Still readable means kept: there is no
        # narrower selection the employee could have left, and a report can miss an
        # employee the API still serves. A 200 of the wrong shape cannot prove
        # anything: it fails the check like an error and the record stays.
        if not isinstance(record, dict):
            raise BambooHRResponseError("BambooHR /employees/{id}: not an object")
        return False

    # -- the run -----------------------------------------------------------
    def rows(self) -> Iterator[dict[str, Any]]:
        stored = self.state.setdefault("subdomain", self.subdomain)
        if stored != self.subdomain:
            raise BambooHRSourceError(
                "This resource_name already holds another BambooHR company's sync; "
                "give each BambooHR company its own resource_name"
            )
        self.state["v"] = STATE_VERSION
        labels = self._load_labels()
        # Finish deletions a cut run left behind before starting a new cycle.
        yield from self._forget()
        yield from self._changed_rows(self._report(), labels)
        self._close_cycle()
        yield from self._forget()
        self.stats["kept"] = len(self.kept)
        self.stats["pending"] = len(self.state.get("pending_deletes") or [])


def new_stats() -> dict[str, int]:
    """The counts of a run, every key present from its start so a host finds them."""
    return dict.fromkeys(
        ("scanned", "skipped", "deleted", "failed", "kept", "pending", "requests", "held_back"), 0
    )


def _iter_rows(
    client: Any,
    state: dict,
    stats: dict[str, int],
    *,
    subdomain: str,
    fields: list[str],
    max_requests: int = DEFAULT_MAX_REQUESTS,
    check_active: Callable[[], None] | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield the changed rows of one company. Pure of dlt, so tests drive it with a dict."""
    walker = _Walker(
        client,
        state,
        stats,
        subdomain=subdomain,
        fields=fields,
        max_requests=max_requests,
        check_active=check_active,
    )
    try:
        yield from walker.rows()
    except _CutShort as cut:
        stats["failed"] += 1
        stats[f"failed_{cut.reason}"] = 1
        stats["kept"] = len(walker.kept)
        stats["pending"] = len(state.get("pending_deletes") or [])
        logger.warning("BambooHR sync stopped early (%s); it resumes on the next run.", cut.reason)
    except BambooHRSourceError as exc:
        # A refused field list, report or key ends the run with a raise instead of a
        # cut: a cut run would keep state that marks the cycle closed and go on
        # forgetting against a listing it never saw whole. dlt drops the state of a
        # run that raises, so the next run starts the cycle over and nothing is
        # forgotten. The host reads these counts after the raise: dlt drops the
        # state, not this object.
        stats["failed"] += 1
        reason = next((k for t, k in _RAISED_REASONS.items() if isinstance(exc, t)), None)
        if reason:
            stats[reason] = stats.get(reason, 0) + 1
        raise


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------
def _field_list(fields: list[str] | tuple[str, ...]) -> list[str]:
    """The explicit field allowlist: sorted, without repeats.

    Employee data is sensitive, so ``fields`` is required and there is no default that
    would dump every field: anything but a non-empty list or tuple of alias strings is
    refused. A bare string would be read as one field per letter, and an empty
    allowlist would sync nobody.
    """
    if not isinstance(fields, (list, tuple)) or not fields:
        raise ValueError(
            "bamboohr_source needs an explicit field allowlist: fields must be a "
            "non-empty list or tuple of BambooHR field aliases, like "
            '["firstName", "lastName", "jobTitle"]. GET /meta/fields lists the aliases.'
        )
    bad_shape = [
        field for field in fields if not isinstance(field, str) or not _FIELD.fullmatch(field)
    ]
    if bad_shape:
        raise ValueError(
            f"fields must be BambooHR field aliases of letters, digits, _ and - only; "
            f"got {bad_shape}. GET /meta/fields lists the aliases."
        )
    return sorted(set(fields))


def bamboohr_source(
    *,
    resource_name: str,
    subdomain: str,
    fields: list[str] | tuple[str, ...],
    api_key: str | None = None,
    service: Any = None,
    check_active: Callable[[], None] | None = None,
    max_requests: int = DEFAULT_MAX_REQUESTS,
):
    """Return a ``dlt`` resource that yields a BambooHR company's employee documents.

    Args:
        resource_name: Stable name of this connection, one per BambooHR company. It
            scopes the sync state and the cleanup of deleted employees, so keep it
            fixed across runs. There is no default: two companies sharing a name
            would forget each other's employees, and a run refuses state written
            for another company.
        subdomain: The company's BambooHR subdomain (the ``acme`` of
            ``acme.bamboohr.com``), lowercase letters, digits and dashes.
        fields: The explicit field allowlist, as ``GET /meta/fields`` lists the field
            aliases, e.g. ``["firstName", "lastName", "jobTitle", "department",
            "location", "status"]``. Only these fields are ever requested. Required:
            employee data is sensitive and this connector never dumps every field.
            Changing the list re-renders every employee once.
        api_key: BambooHR API key. Used to build the client when ``service`` is
            omitted. Never read from the environment.
        service: Pre-built client (see :func:`build_bamboohr_service`). Mainly an
            injection point for tests and hosts. Its ``request`` takes ``params``,
            ``body`` and ``check_active`` keywords and must call ``check_active``
            before each attempt it makes.
        check_active: Optional host authorization checkpoint, called before each
            request.
        max_requests: Budget of API calls in one run. A call the client retries
            counts once. BambooHR asks for at most one request per second, so a run
            with many deletion checks is slow; a bigger company finishes over several
            runs.

    Returns:
        A ``dlt`` resource configured with ``primary_key="id"``,
        ``write_disposition="merge"`` and a ``_deleted`` hard-delete column.
    """
    import dlt

    dlt_utils.require_document_sync_version(
        2, provider="BambooHR", feature="reads per-row node sets"
    )
    if not isinstance(resource_name, str) or not resource_name.strip():
        raise ValueError("bamboohr_source needs a resource_name, one per BambooHR company")
    if not isinstance(subdomain, str) or not _SUBDOMAIN.fullmatch(subdomain):
        raise ValueError(
            f"bamboohr_source needs the company's BambooHR subdomain, lowercase letters, "
            f"digits and dashes, like 'acme'; got {subdomain!r}"
        )
    if not isinstance(max_requests, int) or isinstance(max_requests, bool) or max_requests < 1:
        raise ValueError("max_requests must be a positive int")
    allowlist = _field_list(fields)
    if service is None and not api_key:
        raise ValueError("bamboohr_source needs a service or an api_key")

    # Built here so a malformed key fails at construction and the closure holds no raw key.
    client = service if service is not None else BambooHRClient(api_key or "", subdomain)
    stats: dict[str, int] = {}

    @dlt.resource(
        name=resource_name,
        primary_key="id",
        write_disposition="merge",
        columns={"_deleted": {"data_type": "bool", "hard_delete": True}},
    )
    def bamboohr_company():
        stats.clear()
        stats.update(new_stats())
        rows = _iter_rows(
            client,
            dlt.current.resource_state(),
            stats,
            subdomain=subdomain,
            fields=allowlist,
            max_requests=max_requests,
            check_active=check_active,
        )
        yield from dlt_utils.guarded_rows(rows, check_active)

    resource = bamboohr_company()
    setattr(resource, DOCUMENT_SOURCE_ATTR, "bamboohr")
    setattr(resource, dlt_utils.PIPELINE_SCOPE_ATTR, resource_name)
    resource.cognee_sync_stats = stats
    return resource
