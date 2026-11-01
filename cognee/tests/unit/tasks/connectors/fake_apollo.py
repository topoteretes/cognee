# Adapted from topoteretes/cognee-community (Apache-2.0), PR #304, commit 83a980e.
# Modified to ship in the Cognee SDK with SDK-local imports and test paths.
"""In-memory stand-in for the Apollo REST API, shaped like its real responses.

An endpoint the connector does not use, the activity feed included, is not answered: a
call to it raises ``AssertionError`` and fails the test that made it.
"""

import math

from cognee.tasks.ingestion.connectors.apollo import (
    BASE_URL,
    ApolloNotFoundError,
    _endpoint,
)

CONTACT_STAGES = [{"id": "cs1", "name": "Cold"}, {"id": "cs2", "name": "Interested"}]
ACCOUNT_STAGES = [{"id": "as1", "name": "Prospect"}, {"id": "as2", "name": "Customer"}]
LABELS = [{"id": "l1", "name": "Champions"}, {"id": "l2", "name": "Key Accounts"}]
# /fields prefixes ids with the modality; records key their values by the bare id. Real
# picklist and multi-select fields answer option ids, not labels (see apollo.md): text values
# stand in for them in these tests.
FIELDS = [
    {"id": "contact.f1", "label": "Messaging Angle"},
    {"id": "contact.f2", "label": "Seniority"},
    {"id": "account.f3", "label": "Tier"},
]


class FakeApollo:
    def __init__(self, team_id: str | None = "team1"):
        # None builds stages that carry no team id at all
        self.team_id = team_id
        self.contacts: dict[str, dict] = {}
        self.accounts: dict[str, dict] = {}
        self.sequences: dict[str, dict] = {}
        # still in apollo but missing from search results, like a record skipped by a page shift
        self.hidden: set[str] = set()
        self.total_override: int | None = None
        self.fail_after: int | None = None
        self.fail_with: Exception | None = None
        self.fail_paths: dict[str, Exception] = {}
        # paths answered with this payload instead, like a 200 with an empty body
        self.answers: dict[str, object] = {}
        self.no_pagination: set[str] = set()
        self.bodies: list[tuple[str, dict]] = []
        self.calls: list[tuple[str, str]] = []
        # what the client reports per endpoint label, see ApolloClient.rate_limit
        self.rate_limit: dict[str, dict[str, int | None]] = {}
        self._clock = 0

    # -- seeding -----------------------------------------------------------
    def _stamp(self) -> str:
        self._clock += 1
        return f"2026-10-01T10:{self._clock // 60:02d}:{self._clock % 60:02d}.000Z"

    def add_contact(self, contact_id: str, name: str, **fields) -> dict:
        self.contacts[contact_id] = {
            "id": contact_id,
            "name": name,
            "title": "Engineer",
            # distinct from every account name the tests use, so a text that names the account
            # cannot have got the name from here
            "organization_name": "Acme Inc",
            "contact_stage_id": "cs1",
            "label_ids": [],
            "typed_custom_fields": {},
            "contact_campaign_statuses": [],
            "emailer_campaign_ids": [],
            "last_activity_date": None,
            "created_at": self._stamp(),
            "updated_at": "2026-10-01T09:00:00.000Z",
            # enrichment and channels the connector must never render
            "email": f"{contact_id}@example.com",
            "phone_numbers": [{"raw_number": "+1 555 0100"}],
            "linkedin_url": f"https://linkedin.com/in/{contact_id}-enriched",
            "headline": "Enriched headline",
            "email_status": "verified",
            "organization": {"name": "Acme", "industry": "Enriched industry"},
            **fields,
        }
        return self.contacts[contact_id]

    def add_account(self, account_id: str, name: str, **fields) -> dict:
        # apollo returns no updated_at on accounts
        self.accounts[account_id] = {
            "id": account_id,
            "name": name,
            "domain": f"{account_id}.example.com",
            "account_stage_id": "as1",
            "label_ids": [],
            "typed_custom_fields": {},
            "created_at": self._stamp(),
            "linkedin_url": f"https://linkedin.com/company/{account_id}-enriched",
            "industry": "Enriched industry",
            "estimated_num_employees": 5000,
            **fields,
        }
        return self.accounts[account_id]

    def add_sequence(self, sequence_id: str, name: str, **fields) -> dict:
        self.sequences[sequence_id] = {
            "id": sequence_id,
            "name": name,
            "active": True,
            "archived": False,
            "num_steps": 3,
            "unique_opened": 7,
            "open_rate": 0.4,
            **fields,
        }
        return self.sequences[sequence_id]

    def enroll(self, contact_id: str, sequence_id: str, *, status: str = "active") -> None:
        """Enroll a contact the way apollo does: statuses change, updated_at does not."""
        contact = self.contacts[contact_id]
        contact["contact_campaign_statuses"] = [
            {
                "id": f"ccs-{contact_id}",
                "emailer_campaign_id": sequence_id,
                "status": status,
                "current_step_position": 1,
            }
        ]
        contact["emailer_campaign_ids"] = [sequence_id]
        contact["last_activity_date"] = self._stamp()

    def advance_step(self, contact_id: str) -> None:
        """Send the next step of a sequence: only activity moves, not the status."""
        contact = self.contacts[contact_id]
        for status in contact["contact_campaign_statuses"]:
            status["current_step_position"] += 1
        contact["last_activity_date"] = self._stamp()

    # -- the api -----------------------------------------------------------
    def count(self, path: str) -> int:
        return sum(1 for _, called in self.calls if called == path)

    def _stages(self, stages: list[dict]) -> list[dict]:
        if self.team_id is None:
            return [dict(stage) for stage in stages]
        return [{**stage, "team_id": self.team_id} for stage in stages]

    def request(self, method: str, path: str, *, body=None, params=None):
        self.calls.append((method, path))
        self.bodies.append((path, dict(body or {})))
        if path in self.fail_paths:
            raise self.fail_paths[path]
        if self.fail_after is not None and len(self.calls) > self.fail_after:
            raise self.fail_with
        if path in self.answers:
            return self.answers[path]
        body = body or {}
        if path == "/contact_stages":
            return {"contact_stages": self._stages(CONTACT_STAGES)}
        if path == "/account_stages":
            return {"account_stages": self._stages(ACCOUNT_STAGES)}
        if path == "/labels":
            return LABELS
        if path == "/fields":
            return {"fields": FIELDS, "field_groups": []}
        if path == "/emailer_campaigns/search":
            sequences = list(self.sequences.values())
            return self._page(path, "emailer_campaigns", sequences, params or {})
        if path == "/contacts/search":
            return self._page(path, "contacts", self._search(self.contacts, "contact", body), body)
        if path == "/accounts/search":
            return self._page(path, "accounts", self._search(self.accounts, "account", body), body)
        for kind, store in (("contact", self.contacts), ("account", self.accounts)):
            prefix = f"/{kind}s/"
            if method == "GET" and path.startswith(prefix):
                record = store.get(path[len(prefix) :])
                if record is None:
                    raise ApolloNotFoundError("Apollo record not found (HTTP 422)")
                return {kind: dict(record)}
        raise AssertionError(f"unexpected call {method} {path}")

    def _search(self, store: dict, kind: str, body: dict) -> list[dict]:
        stage_ids = body.get(f"{kind}_stage_ids")
        label_ids = body.get(f"{kind}_label_ids")
        records = [
            r
            for r in store.values()
            if r["id"] not in self.hidden
            and (not stage_ids or r[f"{kind}_stage_id"] in stage_ids)
            and (not label_ids or set(r["label_ids"]) & set(label_ids))
        ]
        assert body.get("sort_by_field") == f"{kind}_created_at"
        return sorted(records, key=lambda r: r["created_at"])

    def _page(self, path: str, key: str, records: list[dict], body: dict) -> dict:
        page, per_page = body.get("page", 1), body.get("per_page", 100)
        total = self.total_override if self.total_override is not None else len(records)
        chunk = records[(page - 1) * per_page : page * per_page]
        if path in self.no_pagination:
            return {key: [dict(record) for record in chunk]}
        return {
            key: [dict(record) for record in chunk],
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total_entries": total,
                "total_pages": math.ceil(total / per_page) if per_page else 0,
            },
        }


class _HTTPResponse:
    def __init__(self, status: int, payload, minute_left: int):
        self.status_code = status
        self._payload = payload
        self.headers = {"x-minute-requests-left": str(minute_left), "x-hourly-requests-left": "199"}

    def json(self):
        return self._payload


def serve_over_http(
    fake: FakeApollo, *, keys: list[str] | None = None, windows: dict | None = None
):
    """A stand-in for ``requests.request`` that answers from ``fake`` the way Apollo's HTTP API does.

    ``keys`` collects the ``x-api-key`` of every request. ``windows`` maps an endpoint label
    (``/contacts/{id}``) to the calls left in its minute window: every answer for that endpoint,
    a 404 included, carries what is left in its headers and spends one, and an endpoint at 0
    answers 429. An endpoint not listed reports 49 and never runs out.
    """
    windows = {} if windows is None else windows

    def request(method, url, *, headers, json=None, params=None, timeout=None, **_):
        if keys is not None:
            keys.append(headers["x-api-key"])
        path = url.removeprefix(BASE_URL)
        label = _endpoint(path)
        left = windows.get(label)
        if left == 0:
            return _HTTPResponse(429, {}, 0)
        if left is not None:
            windows[label] = left = left - 1
        minute = 49 if left is None else left
        try:
            payload = fake.request(method, path, body=json, params=params)
        except ApolloNotFoundError:
            return _HTTPResponse(422, {"error": "This contact has been deleted."}, minute)
        return _HTTPResponse(200, payload, minute)

    return request
