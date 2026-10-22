# Adapted from topoteretes/cognee-community (Apache-2.0), PR #304, commit 83a980e.
# Modified to ship in the Cognee SDK with SDK-local imports and test paths.
"""Apollo.io connector for cognee: a ``dlt`` source that turns a team's CRM into memory.

One source is one Apollo team. It yields the team's own contacts, accounts and sequences
as flat document rows, never Apollo's enrichment database, and is meant to be handed
directly to :func:`cognee.remember`::

    import cognee
    from cognee.tasks.ingestion.connectors import apollo_source

    await cognee.remember(
        apollo_source(api_key="<apollo api key>", resource_name="apollo_sales"),
        dataset_name="apollo",
        primary_key="id",
        write_disposition="merge",   # REQUIRED, the add pipeline defaults to "replace"
        max_rows_per_table=0,
    )

Design
------
* **Rows** are flat ``{id, title, content, url, _deleted, cognee_node_set}``. Ids are
  prefixed by kind (``contact:<id>``, ``account:<id>``, ``sequence:<id>``). Every column this
  source emits feeds the stored document identity, so the rendering and the node set are
  frozen and deterministic: nothing volatile (last-activity dates, step positions, rates) is
  rendered and lists are sorted. A tombstone carries no node set.
* **Node set** ``apollo:<team id>:<kind>`` (``contacts``, ``accounts``, ``sequences``) scopes a
  ``recall`` to Apollo, to one kind, or to one team.
* **What is rendered** is names, titles, accounts, domains, stages, lists and, per contact,
  the names and statuses of its sequences. Each text opens with one sentence that names its
  kind, and a contact names its account by the account's own title, so the graph can join
  the two records. Emails, phone numbers and social profiles are never rendered. Custom
  fields are free text that teams use for exactly those, so none are rendered unless
  ``custom_field_ids`` names them.
* **Incremental** by fingerprint. Apollo has no change feed and no deleted flag, so every
  cycle lists each included kind in full, oldest first, and a row is emitted only when
  the hash of its rendered text changed. The fingerprints live in dlt resource state.
* **Forget-on-delete.** An id that stopped appearing in a complete cycle is checked once
  by reading the record: gone (404, or Apollo's legacy "deleted" text), or no longer in the
  selected stages or lists, is emitted as a ``_deleted`` tombstone. Any other answer, an
  error included, keeps the record. A kind left out of ``include`` is forgotten. Apollo
  search stops at 50,000 records, so a bigger listing is incomplete and forgets nothing.
  A stage or list filter that names an id Apollo does not list fails the run first: it would
  match nothing, and every synced record would look deleted.
* **A record whose check keeps failing** backs off: after five failed checks in a row it
  is left alone for ten cycles, so a record Apollo keeps refusing does not cost a request
  and a ``failed`` count on every run. ``held_back`` counts the records sitting out.
* **A run is bounded and resumable.** On a rate limit, low remaining quota, a spent
  request budget or a rejected key after rows were emitted, the source stops cleanly, keeps
  the state it reached and reports it in ``cognee_sync_stats`` (``failed_rate_limit``,
  ``failed_budget``, ``failed_auth``). Apollo counts per endpoint, so the quota that stops a
  run is the one of the endpoint it is about to call. It neither raises nor waits out a rate
  limit: ``dlt`` rolls the state of a failed run back, so a workspace too big for one window
  would never finish, and waiting would hold the process-wide staging lock.
* **One team per resource_name.** The first run stores the team id Apollo reports, and a
  run under another team's key is refused, because it would forget the first team's rows.
* **Key** is a plain Apollo API key. It lives in the client object only; errors carry
  statuses and endpoints, never the key or a record id. A redirect is an error, not
  followed: ``requests`` would send the key to whatever host it names.

Privacy
-------
This connector reads personal data about people outside your organisation. It is
**opt-in**: nothing is fetched until you construct a source and call ``remember``. What
it reads goes to your configured LLM and embedding providers, and anyone with read access
to the target dataset can see it.
"""

import hashlib
import json
import re
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

from cognee.shared.logging_utils import get_logger
from cognee.tasks.ingestion import dlt_utils
from cognee.tasks.ingestion.dlt_utils import DOCUMENT_SOURCE_ATTR, NODE_SET_COLUMN

logger = get_logger("apollo_connector")

BASE_URL = "https://api.apollo.io/api/v1"
APP_URL = "https://app.apollo.io/#"
# The first-party record types this source syncs; enrichment data stays out.
KINDS = ("contacts", "accounts", "sequences")
# Apollo's largest per_page on the search endpoints.
PAGE_SIZE = 100
# Apollo search stops at 500 pages of 100 records.
SEARCH_CAP = 50_000
# One capped listing (500 pages) per run; the rate-limit windows stop smaller plans first.
DEFAULT_MAX_REQUESTS = 500
# Requests left untouched in each rate-limit window of an endpoint, for the key's other users.
# Apollo counts per endpoint, so one endpoint running low never stops calls to another.
_REQUESTS_RESERVE = 10
# (connect, read) seconds; our own choice, Apollo documents no expected latency.
_TIMEOUT = (10, 30)
# Attempts per request on 5xx or network errors, backing off 1s then 2s.
_RETRIES = 3
# Failed deletion checks in a row after which a record backs off. Chosen, not measured:
# enough that a bad hour is not a streak, few enough that a stuck record soon stops costing
# a request per cycle.
_MAX_CHECK_FAILURES = 5
# Cycles a record in back-off is left alone before its check is tried again, so a stuck
# record costs one request per eleven cycles.
_BACKOFF_CYCLES = 10
_RATE_HEADERS = {
    "minute": "x-minute-requests-left",
    "hourly": "x-hourly-requests-left",
    "daily": "x-24-hour-requests-left",
}
# Legacy 422 texts seen live for a deleted or unknown contact or account, matched whole
# so a record name can't trip them. error_details.code replaces them on 16 Feb 2027, but
# Apollo documents no deleted-record code yet, so until then only these and a 404 count.
_GONE_MESSAGE = re.compile(
    r"(oops! )?this (contact|account) (has been deleted|does not exist in apollo)[.!]?"
)
# A record id goes into the URL of its own read, so no ``.``, ``/``, ``?``, ``#`` or ``%`` may
# get through: ``../organizations/enrich?domain=x.com`` reads another endpoint, and ``..`` alone
# still normalizes to /api/v1/ after quote(safe="") (requests 2.34.2). Ids seen live are 24
# hex characters; only the character set is checked, so a longer or non-hex id still works.
_RECORD_ID = re.compile(r"[0-9A-Za-z]+")


class ApolloSourceError(RuntimeError):
    """Base class of the errors this source raises."""


class ApolloAPIError(ApolloSourceError):
    """Apollo answered with an error, or could not be reached."""


class ApolloAuthError(ApolloSourceError):
    """Apollo rejected the API key (HTTP 401)."""


class ApolloAccessError(ApolloSourceError):
    """The API key or plan cannot call an endpoint (HTTP 403)."""


class ApolloRateLimitedError(ApolloSourceError):
    """Apollo answered HTTP 429."""


class ApolloNotFoundError(ApolloAPIError):
    """The record was deleted or never existed."""


# The count key of an error that ends a run by raising, matched with isinstance. An error with no
# entry (a 5xx, a bad filter, another team's state) raises with ``failed`` alone. A record read
# that is refused never gets here: _should_forget counts it and goes on.
_RAISED_REASONS = {ApolloAuthError: "failed_auth", ApolloAccessError: "failed_access"}


def _int_header(headers: Any, name: str) -> int | None:
    try:
        return int(headers.get(name))
    except (TypeError, ValueError):
        return None


def _is_gone(body: Any) -> bool:
    """Whether Apollo said the record is gone. Only matched, never echoed."""
    if not isinstance(body, dict):
        return False
    texts = (body.get("error"), body.get("message"))
    return any(
        isinstance(text, str) and _GONE_MESSAGE.fullmatch(text.strip().lower()) for text in texts
    )


def _endpoint(path: str) -> str:
    """The path with a record id replaced, so error messages never carry one."""
    head, _, tail = path.rpartition("/")
    if head in ("/contacts", "/accounts") and tail != "search":
        return f"{head}/{{id}}"
    return path


def _record_path(collection: str, record_id: str) -> str:
    """The path that reads one record of a ``collection``, the plural path segment (``contacts``).

    Raises for an id that is not alphanumeric. The id comes out of an Apollo answer, and the
    error names the endpoint, never the id. It guards the request only: the id stays as it is
    in the row id and the ``url`` column, where it is never requested.
    """
    if not _RECORD_ID.fullmatch(record_id):
        raise ApolloAPIError(
            f"Apollo sent a record id with characters other than letters and digits, "
            f"so /{collection}/{{id}} was not requested"
        )
    return f"/{collection}/{record_id}"


class ApolloClient:
    """Minimal synchronous REST client for Apollo.

    Apollo limits per endpoint, so after every answer that carries the rate-limit headers, an
    error answer included, ``rate_limit`` holds the requests left in the minute, hourly and
    daily windows under that endpoint's label (``_endpoint``):
    ``{"/contacts/{id}": {"minute": 40, "hourly": 390, "daily": 1900}}``. Tests and hosts can
    pass any object with the same ``request`` method and ``rate_limit`` attribute instead.
    """

    def __init__(self, api_key: str, *, sleep: Callable[[float], None] = time.sleep):
        key = (api_key or "").strip()
        # This check is what keeps the key out of an exception: requests names the value of a
        # header it rejects, and every key it would reject is refused here.
        if not key or not key.isascii() or not key.isprintable() or " " in key:
            raise ValueError("The Apollo API key is empty or has an invalid format")
        self._api_key = key
        self._sleep = sleep
        self.rate_limit: dict[str, dict[str, int | None]] = {}

    def __repr__(self) -> str:
        return "ApolloClient(api_key=<redacted>)"

    def request(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        import requests

        headers = {
            "x-api-key": self._api_key,
            "Content-Type": "application/json",
            "Cache-Control": "no-cache",
        }
        for attempt in range(_RETRIES):
            last = attempt == _RETRIES - 1
            try:
                response = requests.request(
                    method,
                    BASE_URL + path,
                    json=body,
                    params=params,
                    headers=headers,
                    timeout=_TIMEOUT,
                    # requests drops Authorization on a hop to another host but not x-api-key.
                    allow_redirects=False,
                )
            except requests.RequestException as exc:
                if last:
                    # `from None` keeps the cause out of this error, not out of __context__,
                    # which dlt's trace file still reads (dlt 1.30.0, pipeline/trace.py): the
                    # key is kept out of an exception by the check in __init__.
                    raise ApolloAPIError(f"Apollo request failed: {type(exc).__name__}") from None
                self._sleep(2**attempt)
                continue
            self._read_rate_limit(response, path)
            if response.status_code >= 500 and not last:
                self._sleep(2**attempt)
                continue
            return self._parse(response, path)

    def _read_rate_limit(self, response: Any, path: str) -> None:
        """Keep the windows an answer reports under the endpoint it came from.

        An answer without the headers (a 5xx from a proxy) leaves what is known about the
        endpoint: it says nothing about the windows.
        """
        windows = {
            window: _int_header(response.headers, name) for window, name in _RATE_HEADERS.items()
        }
        if any(left is not None for left in windows.values()):
            self.rate_limit[_endpoint(path)] = windows

    def _parse(self, response: Any, path: str) -> Any:
        status = response.status_code
        if 300 <= status < 400:
            raise ApolloAPIError(
                f"Apollo answered {_endpoint(path)} with a redirect (HTTP {status})"
            )
        if status == 401:
            raise ApolloAuthError("Apollo rejected the API key (HTTP 401)")
        if status == 403:
            raise ApolloAccessError(
                f"The Apollo API key or plan cannot call {_endpoint(path)} (HTTP 403)"
            )
        if status == 429:
            raise ApolloRateLimitedError("Apollo rate limit exceeded (HTTP 429)")
        try:
            payload = response.json()
        except ValueError:
            payload = None
        if status == 404 or (status == 422 and _is_gone(payload)):
            raise ApolloNotFoundError(f"Apollo record not found (HTTP {status})")
        if status != 200:
            raise ApolloAPIError(f"Apollo request failed: HTTP {status}")
        # Every endpoint used answers with an object, except /labels with a list.
        expected = list if path == "/labels" else dict
        if not isinstance(payload, expected):
            raise ApolloAPIError(f"Apollo sent an unreadable answer for {_endpoint(path)}")
        return payload


def build_apollo_service(api_key: str) -> ApolloClient:
    """Wrap a host-owned API key in a client, the way Linear does."""
    return ApolloClient(api_key)


def _team_id(stages: list[dict]) -> str:
    """The Apollo team id the stage rows carry. Raises when none does.

    The team id keys the state check that keeps one team's sync from forgetting another's
    rows, so a run that cannot read one must not go on unscoped.
    """
    team_id = next((s["team_id"] for s in stages if s.get("team_id")), None)
    if team_id is None:
        raise ApolloSourceError(
            "Apollo's stage list carries no team id, so this sync cannot tell which "
            "Apollo team it belongs to"
        )
    return team_id


def apollo_team_id(service: Any) -> str:
    """The id of the Apollo team behind ``service``'s key.

    For a host that keeps one connection per team: Apollo has no endpoint that names the
    team, so this reads it off the contact stage list, one request.
    """
    stages = service.request("GET", "/contact_stages").get("contact_stages") or []
    return _team_id(stages)


# ---------------------------------------------------------------------------
# Rendering: deterministic, first-party fields only, nothing volatile.
# ---------------------------------------------------------------------------
@dataclass
class Lookups:
    """Id-to-name maps that turn Apollo ids into readable text."""

    contact_stages: dict[str, str] = field(default_factory=dict)
    account_stages: dict[str, str] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)
    # Only the custom fields the source was asked to render, by the id /fields lists:
    # "contact.<id>" or "account.<id>".
    fields: dict[str, str] = field(default_factory=dict)
    sequences: dict[str, str] = field(default_factory=dict)
    # Account id to [title, domain], as the accounts listing read them. The walker keeps it in
    # the dlt state: a run that resumes after the accounts listing finished never reads them
    # again, and a contact must still render the title its account's own document carries.
    accounts: dict[str, list[str]] = field(default_factory=dict)


def _text(value: Any) -> str:
    if isinstance(value, list):
        # A multi-select comes back in no fixed order, and the text decides re-extraction.
        return ", ".join(sorted(text for text in map(_text, value) if text))
    return "" if value is None else str(value).strip()


def _lines(fields: list[tuple[str, str]]) -> list[str]:
    return [f"{label}: {value}" for label, value in fields if value]


def _names(ids: Any, lookup: dict[str, str]) -> str:
    return ", ".join(sorted({lookup[i] for i in ids or [] if lookup.get(i)}))


def _custom_fields(kind: str, values: Any, lookup: dict[str, str]) -> list[tuple[str, str]]:
    if not isinstance(values, dict):
        return []
    # /fields lists an id as "<kind>.<id>"; a record keys its values by the bare id.
    pairs = (
        (lookup.get(f"{kind}.{field_id}", ""), _text(value)) for field_id, value in values.items()
    )
    return sorted((label, text) for label, text in pairs if label and text)


def _account_title(account: dict[str, Any]) -> str:
    """The name an account is known by: its name, else its domain, else empty."""
    return _text(account.get("name")) or _text(account.get("domain"))


def render_contact(contact: dict[str, Any], lookups: Lookups) -> dict[str, Any]:
    """A contact's row. Its account is named by the account's own title when the listing read
    it, else by the contact's ``organization_name``.

    The text opens with one sentence naming the kind and the account ("Apollo contact Dana
    Whitfield works at Acme as Chief Financial Officer."): each record is its own document,
    and the graph only joins two of them when their texts use the same name.
    """
    name = _text(contact.get("name")) or _text(
        f"{contact.get('first_name') or ''} {contact.get('last_name') or ''}"
    )
    title = name or "Unnamed contact"
    account, domain = lookups.accounts.get(contact.get("account_id")) or ("", "")
    # A known account with neither name nor domain names nothing: that contact keeps its own
    # organization_name too, or every such account would merge into one "Unnamed account".
    account = account or _text(contact.get("organization_name"))
    job_title = _text(contact.get("title"))
    employment = " ".join(
        part
        for part in (f"at {account}" if account else "", f"as {job_title}" if job_title else "")
        if part
    )
    sequences = sorted(
        f"{lookups.sequences.get(status.get('emailer_campaign_id'), 'Unknown sequence')}"
        f" ({_text(status.get('status')) or 'unknown'})"
        for status in contact.get("contact_campaign_statuses") or []
    )
    # Contact and account stages are told apart: a stage called "Cold" on both would
    # otherwise merge into one entity.
    fields = [
        ("Account", f"{account} ({domain})" if domain and domain != account else account),
        ("Contact stage", lookups.contact_stages.get(contact.get("contact_stage_id"), "")),
        ("Lists", _names(contact.get("label_ids"), lookups.labels)),
        *_custom_fields("contact", contact.get("typed_custom_fields"), lookups.fields),
        ("Sequences", "; ".join(sequences)),
    ]
    intro = (
        f"Apollo contact {title} works {employment}." if employment else f"Apollo contact {title}."
    )
    return {
        "id": f"contact:{contact['id']}",
        "title": title,
        "content": "\n".join([intro, *_lines(fields)]),
        "url": f"{APP_URL}/contacts/{contact['id']}",
        "_deleted": False,
    }


def render_account(account: dict[str, Any], lookups: Lookups) -> dict[str, Any]:
    title = _account_title(account) or "Unnamed account"
    domain = _text(account.get("domain"))
    fields = [
        ("Account stage", lookups.account_stages.get(account.get("account_stage_id"), "")),
        ("Lists", _names(account.get("label_ids"), lookups.labels)),
        *_custom_fields("account", account.get("typed_custom_fields"), lookups.fields),
    ]
    # An account with no name is titled by its domain: say it once.
    has_domain = domain and domain != title
    intro = (
        f"Apollo account {title} has the domain {domain}."
        if has_domain
        else f"Apollo account {title}."
    )
    return {
        "id": f"account:{account['id']}",
        "title": title,
        "content": "\n".join([intro, *_lines(fields)]),
        "url": f"{APP_URL}/accounts/{account['id']}",
        "_deleted": False,
    }


def render_sequence(sequence: dict[str, Any]) -> dict[str, Any]:
    title = _text(sequence.get("name")) or "Untitled sequence"
    if sequence.get("archived"):
        status = "archived"
    else:
        status = "active" if sequence.get("active") else "inactive"
    intro = f"Apollo sequence {title} is {status}."
    fields = [("Steps", _text(sequence.get("num_steps")))]
    return {
        "id": f"sequence:{sequence['id']}",
        "title": title,
        "content": "\n".join([intro, *_lines(fields)]),
        "url": f"{APP_URL}/sequences/{sequence['id']}",
        "_deleted": False,
    }


def _node_set(team_id: str, kind: str) -> list[str]:
    """The node set of a row: ``apollo:<team id>:<kind>``, with ``kind`` one of ``KINDS``.

    It is hashed into the document identity like every other column, so changing it gives
    every record of the kind a new ``data_id`` and one more extraction on the tenant's LLM key.
    """
    return [f"apollo:{team_id}:{kind}"]


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _total_pages(data: dict) -> int:
    """The page count of a search answer; without it a listing can't be trusted as complete."""
    pagination = data.get("pagination")
    total = pagination.get("total_pages") if isinstance(pagination, dict) else None
    if not isinstance(total, int):
        raise ApolloAPIError("Apollo search answered without pagination")
    return total


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
class _CutShortError(Exception):
    """Internal: stop the run here, keeping the state reached so far."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


_SEARCHES = {
    "contacts": ("/contacts/search", "contact"),
    "accounts": ("/accounts/search", "account"),
}


class _Walker:
    """One run over the selected kinds, writing progress into ``state``.

    Apollo has no change feed and no deleted flag, so every cycle lists each kind
    in full (oldest first) and emits a row only when its fingerprint changed. Ids
    that stopped appearing are re-checked one by one before they are forgotten.
    """

    def __init__(
        self,
        client: Any,
        state: dict,
        stats: dict[str, int],
        *,
        include: tuple[str, ...],
        filters: dict[str, list[str]],
        custom_field_ids: frozenset[str],
        max_requests: int,
        check_active: Callable[[], None] | None,
    ):
        self.client = client
        self.state = state
        self.stats = stats
        for key in ("scanned", "skipped", "deleted", "failed", "requests"):
            stats.setdefault(key, 0)
        self.include = include
        self.filters = filters
        self.custom_field_ids = custom_field_ids
        self.max_requests = max_requests
        self.check_active = check_active
        self.emitted = 0
        self.fingerprints: dict[str, str] = state.setdefault("fingerprints", {})
        self.seen: list[str] = state.setdefault("seen", [])
        # Per record whose deletion check failed: {"failed": checks failed in a row,
        # "wait": cycles still to sit out}. Dropped as soon as the record is listed again,
        # checked successfully or forgotten, so it only holds records that are stuck now.
        self.check_failures: dict[str, dict[str, int]] = state.setdefault("check_failures", {})
        # Ids whose deletion check ran in this run. A run that resumes a cut one checks its
        # pending ids first, and the same ids come up again when the listing closes: checking
        # them twice would count one run's failure twice.
        self.checked: set[str] = set()
        # The records in back-off as the run starts, so a run that is cut or capped before it
        # closes the listing still reports the ones it leaves alone.
        stats["held_back"] = sum(self._in_back_off(row_id) for row_id in self.check_failures)
        # The titles contacts name their accounts by, ids to [title, domain]: state, because a run
        # resumed after the accounts listing finished skips it, and the contact text would flip
        # between the account's title and the contact's organization_name. A forgotten account
        # leaves it (_forget).
        self.account_names: dict[str, list[str]] = state.setdefault("account_names", {})
        self.lookups = Lookups(accounts=self.account_names)

    # -- requests ----------------------------------------------------------
    def _call(self, method: str, path: str, **kwargs: Any) -> Any:
        if self.check_active is not None:
            self.check_active()
        self._check_budget(path)
        # A call the client retries on a 5xx or a network error still counts once: the budget
        # and this count are about what the run asks for, and the client owns its retries.
        self.stats["requests"] += 1
        try:
            return self.client.request(method, path, **kwargs)
        except ApolloRateLimitedError:
            raise _CutShortError("rate_limit") from None
        except ApolloAuthError:
            if self.emitted:
                raise _CutShortError("auth") from None
            raise

    def _check_budget(self, path: str) -> None:
        if self.stats["requests"] >= self.max_requests:
            raise _CutShortError("budget")
        # Apollo counts per endpoint: only the windows of the one about to be called matter.
        remaining = (getattr(self.client, "rate_limit", None) or {}).get(_endpoint(path)) or {}
        for window in _RATE_HEADERS:
            left = remaining.get(window)
            if left is not None and left < _REQUESTS_RESERVE:
                raise _CutShortError("rate_limit")

    # -- lookups -----------------------------------------------------------
    def _load_lookups(self) -> list[dict]:
        """Read the id-to-name maps the included kinds render; return the sequence list.

        Only what a kind renders is read, so a key scoped to fewer endpoints still syncs the
        kinds it can reach: accounts need no sequence list, and sequences need no stage,
        list or field lookups. Contacts name their sequences, so they need the sequence list.
        The contact stage list is read on every run: it carries the team id the state check
        needs.
        """
        contacts = "contacts" in self.include
        accounts = "accounts" in self.include
        sequences = "sequences" in self.include
        contact_stages = self._call("GET", "/contact_stages").get("contact_stages") or []
        self._check_team(contact_stages)
        account_stages = []
        if accounts:
            account_stages = self._call("GET", "/account_stages").get("account_stages") or []
        # /labels answers with a bare list
        labels = self._call("GET", "/labels") if contacts or accounts else []
        # A sequence's rows need the list, and so do the sequence names in a contact's text.
        sequence_list = list(self._all_sequences()) if contacts or sequences else []
        self.lookups = Lookups(
            accounts=self.account_names,
            contact_stages={s["id"]: _text(s.get("name")) for s in contact_stages},
            account_stages={s["id"]: _text(s.get("name")) for s in account_stages},
            labels={label["id"]: _text(label.get("name")) for label in labels},
            fields=self._custom_field_labels() if contacts or accounts else {},
            sequences={s["id"]: _text(s.get("name")) for s in sequence_list},
        )
        return sequence_list

    def _custom_field_labels(self) -> dict[str, str]:
        """Labels of the custom fields to render, read only when some are asked for."""
        if not self.custom_field_ids:
            return {}
        listed = self._call("GET", "/fields", params={"source": "custom"}).get("fields") or []
        labels = {f["id"]: _text(f.get("label")) for f in listed}
        unknown = self.custom_field_ids - labels.keys()
        if unknown:
            raise ApolloSourceError(
                "custom_field_ids names fields Apollo does not list as custom fields: "
                f"{sorted(unknown)}. GET /fields?source=custom lists the ids."
            )
        return {field_id: labels[field_id] for field_id in self.custom_field_ids}

    def _check_filters(self) -> None:
        """Refuse a stage or list filter that names an id Apollo does not list.

        A typo, a blank or a list the key cannot see matches nothing, so the listing comes
        back empty and the walk would find every synced record missing and forget it. Only
        the filters of included kinds are checked, against the lists this run read.
        """
        listed = {
            "contact_stage_ids": ("contacts", self.lookups.contact_stages, "/contact_stages"),
            "contact_label_ids": ("contacts", self.lookups.labels, "/labels"),
            "account_stage_ids": ("accounts", self.lookups.account_stages, "/account_stages"),
            "account_label_ids": ("accounts", self.lookups.labels, "/labels"),
        }
        for name, (kind, ids, endpoint) in listed.items():
            if kind not in self.include:
                continue
            unknown = set(self.filters.get(name) or []) - ids.keys()
            if unknown:
                raise ApolloSourceError(
                    f"{name} names ids Apollo does not list: {sorted(unknown)}. "
                    f"GET {endpoint} lists the ids."
                )

    def _check_team(self, stages: list[dict]) -> None:
        """Refuse state written for another Apollo team, which would forget that team's rows."""
        team_id = _team_id(stages)
        stored = self.state.setdefault("team_id", team_id)
        if stored != team_id:
            raise ApolloSourceError(
                "This resource_name already holds another Apollo team's sync; "
                "give each Apollo account its own resource_name"
            )

    def _all_sequences(self) -> Iterator[dict]:
        page = 1
        while True:
            # The docs list page and per_page as query params, with no request body.
            data = self._call(
                "POST", "/emailer_campaigns/search", params={"page": page, "per_page": PAGE_SIZE}
            )
            records = data.get("emailer_campaigns") or []
            total_pages = _total_pages(data)
            yield from records
            if len(records) < PAGE_SIZE or page >= total_pages:
                return
            page += 1

    # -- listing -----------------------------------------------------------
    def _search_body(self, kind: str) -> dict[str, Any]:
        _, singular = _SEARCHES[kind]
        body: dict[str, Any] = {
            "sort_by_field": f"{singular}_created_at",
            "sort_ascending": True,
            "per_page": PAGE_SIZE,
        }
        for key in (f"{singular}_stage_ids", f"{singular}_label_ids"):
            if self.filters.get(key):
                body[key] = self.filters[key]
        return body

    def _walk(self, kind: str) -> Iterator[dict]:
        """Yield every record of a kind, resuming at the page a cut run stopped on."""
        path, _ = _SEARCHES[kind]
        body = self._search_body(kind)
        walks = self.state.setdefault("walks", {})
        query = _hash(body)
        # A page reached under other filters points into a different listing.
        if walks.get(kind, {}).get("query") != query:
            walks[kind] = {"page": 1, "query": query}
            # Ids seen under the old query must not shield records the new one dropped.
            prefix = f"{_SEARCHES[kind][1]}:"
            self.seen[:] = [row_id for row_id in self.seen if not row_id.startswith(prefix)]
        progress = walks[kind]
        if progress.get("done"):
            return
        while True:
            data = self._call("POST", path, body={**body, "page": progress["page"]})
            records = data.get(kind) or []
            total_pages = _total_pages(data)
            if (data["pagination"].get("total_entries") or 0) > SEARCH_CAP:
                progress["capped"] = True
            yield from records
            last_page = min(total_pages, SEARCH_CAP // PAGE_SIZE)
            if len(records) < PAGE_SIZE or progress["page"] >= last_page:
                break
            progress["page"] += 1
        progress["done"] = True

    def _changed_rows(
        self, records: Iterable[dict], render: Callable[[dict], dict], kind: str
    ) -> Iterator[dict]:
        """Render each record and yield the rows whose fingerprint changed.

        The fingerprint hashes the whole row, so it follows what is rendered: a renamed account
        changes the text of its contacts and re-emits them, and so would a new node set.
        """
        for record in records:
            row: dict[str, Any] = {
                **render(record),
                NODE_SET_COLUMN: _node_set(self.state["team_id"], kind),
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
    def _close_cycle(self) -> None:
        """Queue ids not seen in a complete cycle, then start a fresh one."""
        walks = self.state.get("walks") or {}
        capped = any(progress.get("capped") for progress in walks.values())
        if capped:
            # A capped listing is incomplete, so nothing missing can be trusted as deleted.
            self.stats["failed"] += 1
            self.stats["failed_cap"] = 1
            logger.warning(
                "Apollo search is capped at %d records; narrow the sync with stage or list "
                "filters. Deletions are not detected until then.",
                SEARCH_CAP,
            )
        else:
            self.state["pending_deletes"] = self._due(set(self.fingerprints) - set(self.seen))
        self.state["walks"] = {}
        self.seen.clear()

    def _due(self, missing: set[str]) -> list[str]:
        """The missing ids to check this cycle. A record in back-off sits the cycle out."""
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
        """Whether the record's deletion check is left alone until its wait is over.

        A back-off record is not a failure, so ``failed`` stays 0 for it and a sync that only
        holds records back reads as healthy; ``held_back`` says otherwise. A record of a kind
        left out of ``include`` is never held: it is forgotten without a request, so back-off
        has nothing to protect.
        """
        streak = self.check_failures.get(row_id)
        return bool(streak and streak["wait"] > 0 and self._included(row_id))

    def _included(self, row_id: str) -> bool:
        """Whether the record's kind is still synced."""
        return f"{row_id.split(':', 1)[0]}s" in self.include

    def _forget(self) -> Iterator[dict]:
        pending = self.state.setdefault("pending_deletes", [])
        while pending:
            row_id = pending[0]
            if self._should_forget(row_id):
                self.fingerprints.pop(row_id, None)
                if row_id.startswith("account:"):
                    # A forgotten account must not go on naming the contacts that point at it.
                    self.account_names.pop(row_id.removeprefix("account:"), None)
                self.stats["deleted"] += 1
                yield {"id": row_id, "_deleted": True}
            pending.pop(0)

    def _should_forget(self, row_id: str) -> bool:
        """Whether the record is gone. A check that fails keeps the record and counts."""
        self.checked.add(row_id)
        try:
            gone = self._gone(row_id)
        except ApolloAccessError as exc:
            self._check_failed(row_id, "failed_access", exc)
            return False
        except ApolloAPIError as exc:
            self._check_failed(row_id, "failed_deletion_check", exc)
            return False
        self.check_failures.pop(row_id, None)
        return gone

    def _check_failed(self, row_id: str, reason: str, exc: Exception) -> None:
        self.stats["failed"] += 1
        self.stats[reason] = self.stats.get(reason, 0) + 1
        streak = self.check_failures.setdefault(row_id, {"failed": 0, "wait": 0})
        streak["failed"] += 1
        if streak["failed"] >= _MAX_CHECK_FAILURES:
            streak["wait"] = _BACKOFF_CYCLES
        # The message names the status and endpoint, never the record id.
        logger.warning(
            "Apollo deletion check failed for one record, %d in a row; it is kept: %s",
            streak["failed"],
            exc,
        )

    def _gone(self, row_id: str) -> bool:
        kind, record_id = row_id.split(":", 1)
        if not self._included(row_id):
            return True  # deselected kinds leave memory
        if kind == "sequence":
            return True  # the sequence list is read in full every run
        path = _record_path(f"{kind}s", record_id)
        try:
            record = self._call("GET", path).get(kind)
        except ApolloNotFoundError:
            return True
        if not isinstance(record, dict):
            raise ApolloAPIError(f"Apollo sent no {kind} for {_endpoint(path)}")
        # Still in Apollo: forget it only if it left the selected stages or lists.
        return not self._in_scope(kind, record)

    def _in_scope(self, kind: str, record: dict) -> bool:
        stage_ids = self.filters.get(f"{kind}_stage_ids")
        label_ids = self.filters.get(f"{kind}_label_ids")
        if stage_ids and record.get(f"{kind}_stage_id") not in stage_ids:
            return False
        if not label_ids:
            return True
        return bool(set(record.get("label_ids") or []) & set(label_ids))

    def _record_account_titles(self, accounts: Iterable[dict]) -> Iterator[dict]:
        """Pass the accounts of a listing on, recording each one's title and domain first."""
        for account in accounts:
            self.account_names[account["id"]] = [
                _account_title(account),
                _text(account.get("domain")),
            ]
            yield account

    # -- the run -----------------------------------------------------------
    def rows(self) -> Iterator[dict]:
        sequences = self._load_lookups()
        self._check_filters()
        # Finish deletions a cut run left behind before starting a new listing.
        yield from self._forget()
        if "sequences" in self.include:
            yield from self._changed_rows(sequences, render_sequence, "sequences")
        # Accounts come before contacts, so every contact is rendered with the titles of the
        # accounts this cycle listed.
        if "accounts" in self.include:
            yield from self._changed_rows(
                self._record_account_titles(self._walk("accounts")),
                lambda account: render_account(account, self.lookups),
                "accounts",
            )
        if "contacts" in self.include:
            yield from self._changed_rows(
                self._walk("contacts"),
                lambda contact: render_contact(contact, self.lookups),
                "contacts",
            )
        self._close_cycle()
        yield from self._forget()


def _iter_rows(
    client: Any,
    state: dict,
    stats: dict[str, int],
    *,
    include: tuple[str, ...] = KINDS,
    filters: dict[str, list[str]] | None = None,
    custom_field_ids: frozenset[str] = frozenset(),
    max_requests: int = DEFAULT_MAX_REQUESTS,
    check_active: Callable[[], None] | None = None,
) -> Iterator[dict]:
    """Yield the changed rows of a team. Pure of dlt, so tests drive it with a dict."""
    walker = _Walker(
        client,
        state,
        stats,
        include=include,
        filters=filters or {},
        custom_field_ids=custom_field_ids,
        max_requests=max_requests,
        check_active=check_active,
    )
    try:
        yield from walker.rows()
    except _CutShortError as cut:
        stats["failed"] += 1
        stats[f"failed_{cut.reason}"] = 1
        logger.warning("Apollo sync stopped early (%s); it resumes on the next run.", cut.reason)
    except ApolloSourceError as exc:
        # A lookup or a listing the key cannot call, or a key Apollo rejects before any row, ends
        # the run with a raise instead of a cut: a cut run keeps its state, so a listing already
        # marked done stays done, the refused one never finishes and the cycle never closes, so
        # no later delete is seen. The host reads these counts after the raise: dlt drops the
        # state, not this object. Every raise counts as a failure; only an error with a count key
        # of its own also sets that key.
        stats["failed"] += 1
        reason = next((k for t, k in _RAISED_REASONS.items() if isinstance(exc, t)), None)
        if reason:
            stats[reason] = stats.get(reason, 0) + 1
        raise


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------
# What GET /fields lists: the modality, a dot and the field's id. Only the modalities this
# source renders; "deal.<id>" and the like name fields it never reads.
_CUSTOM_FIELD_ID = re.compile(r"(?:contact|account)\..+")


def _id_list(name: str, ids: list[str] | tuple[str, ...] | None) -> list[str]:
    """``ids`` sorted and without repeats; ``None`` is none.

    Anything but a list or tuple of strings is refused. A bare string would be read as one id
    per letter, and a filter that matches nothing makes the walk forget every record it
    synced. The order is dropped because it is part of the search-body hash that resumes a
    cut listing: the same filter in another order would start over at page 1.
    """
    if ids is None:
        return []
    if not isinstance(ids, (list, tuple)) or not all(isinstance(i, str) for i in ids):
        raise ValueError(f'{name} must be a list or tuple of id strings, like ["<id>"].')
    return sorted(set(ids))


def _custom_field_id_list(ids: list[str] | tuple[str, ...] | None) -> frozenset[str]:
    """The custom field ids to render, each as ``GET /fields?source=custom`` lists it."""
    fields = _id_list("custom_field_ids", ids)
    bad_shape = [field_id for field_id in fields if not _CUSTOM_FIELD_ID.fullmatch(field_id)]
    if bad_shape:
        raise ValueError(
            f"custom_field_ids must be ids as GET /fields?source=custom lists them, like "
            f"contact.<id> or account.<id>; got {bad_shape}."
        )
    return frozenset(fields)


def apollo_source(
    *,
    resource_name: str,
    api_key: str | None = None,
    service: Any = None,
    check_active: Callable[[], None] | None = None,
    include: tuple[str, ...] | None = None,
    contact_stage_ids: list[str] | None = None,
    contact_label_ids: list[str] | None = None,
    account_stage_ids: list[str] | None = None,
    account_label_ids: list[str] | None = None,
    custom_field_ids: list[str] | None = None,
    max_requests: int = DEFAULT_MAX_REQUESTS,
):
    """Return a ``dlt`` resource that yields an Apollo team's documents for ``remember``.

    Args:
        resource_name: Stable name of this connection, one per Apollo team. It scopes
            the sync state and the cleanup of deleted records, so keep it fixed across
            runs. There is no default: two teams sharing a name would forget each
            other's records, and a run refuses state written for another Apollo team.
        api_key: Apollo API key. Used to build the client when ``service`` is omitted.
        service: Pre-built client (see :func:`build_apollo_service`). Mainly an
            injection point for tests and hosts.
        check_active: Optional host authorization checkpoint, called before each
            request.
        include: Which of ``"contacts"``, ``"accounts"`` and ``"sequences"`` to sync.
            Defaults to all three. A kind left out is forgotten, and the endpoints only
            it needs are not called.
        contact_stage_ids, contact_label_ids: Only sync contacts in these stages or lists.
        account_stage_ids, account_label_ids: Only sync accounts in these stages or lists.
            A filter id Apollo does not list fails the run before it emits or forgets anything.
        custom_field_ids: Custom fields to render, as ``GET /fields?source=custom`` lists their
            ids (``contact.<id>`` or ``account.<id>``); any other shape is refused. None by
            default: teams keep phone numbers and personal emails in custom fields. Do not list
            picklist or multi-select fields: Apollo returns option ids for them, so they render
            as ids. Changing the list re-renders every record.
        max_requests: Budget of API calls in one run. A call the client retries counts once.
            A bigger workspace finishes over several runs.

    Returns:
        A ``dlt`` resource configured with ``primary_key="id"``,
        ``write_disposition="merge"`` and a ``_deleted`` hard-delete column.
    """
    import dlt

    if getattr(dlt_utils, "DOCUMENT_SYNC_VERSION", 0) < 2:
        raise RuntimeError(
            "Apollo sync requires a Cognee build that reads per-row node sets "
            "(DOCUMENT_SYNC_VERSION >= 2). Upgrade Cognee before syncing."
        )
    if not isinstance(resource_name, str) or not resource_name.strip():
        raise ValueError("apollo_source needs a resource_name, one per Apollo team")
    if service is None and not api_key:
        raise ValueError("apollo_source needs a service or an api_key")
    kinds = KINDS if include is None else tuple(include)
    if not kinds or set(kinds) - set(KINDS):
        raise ValueError(f"include must be a non-empty subset of {KINDS}")

    # Built here so a malformed key fails at construction and the closure holds no raw key.
    client = service if service is not None else ApolloClient(api_key or "")
    filters = {
        "contact_stage_ids": _id_list("contact_stage_ids", contact_stage_ids),
        "contact_label_ids": _id_list("contact_label_ids", contact_label_ids),
        "account_stage_ids": _id_list("account_stage_ids", account_stage_ids),
        "account_label_ids": _id_list("account_label_ids", account_label_ids),
    }
    custom_fields = _custom_field_id_list(custom_field_ids)
    stats: dict[str, int] = {}

    @dlt.resource(
        name=resource_name,
        primary_key="id",
        write_disposition="merge",
        # cognee_node_set needs no json hint here: ingest_dlt_source applies
        # DOCUMENT_COLUMN_HINTS to every document source, merged with this one.
        columns={"_deleted": {"data_type": "bool", "hard_delete": True}},
    )
    def apollo_team():
        stats.clear()
        # The quota a reused client saw last run is stale once its windows reset.
        client.rate_limit = {}
        rows = _iter_rows(
            client,
            dlt.current.resource_state(),
            stats,
            include=kinds,
            filters=filters,
            custom_field_ids=custom_fields,
            max_requests=max_requests,
            check_active=check_active,
        )
        yield from dlt_utils.guarded_rows(rows, check_active)

    resource = apollo_team()
    setattr(resource, DOCUMENT_SOURCE_ATTR, "apollo")
    setattr(resource, dlt_utils.PIPELINE_SCOPE_ATTR, resource_name)
    resource.cognee_sync_stats = stats
    return resource
