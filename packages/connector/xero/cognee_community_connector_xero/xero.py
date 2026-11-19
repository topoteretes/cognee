"""Xero data-source connector for cognee — invoices & contacts as memory.

Using a Xero *organisation* you own, this connector turns invoices (and
optionally contacts) into documents for cognee's cognify pipeline: "who owes us
money, for what, and when it was last touched".

Security model
--------------
* **OAuth 2.0 with rotating refresh tokens.** You run one login flow
  (:func:`xero_authenticate`) that stores a short-lived access token plus a
  refresh token in a local JSON file (``XERO_TOKEN_PATH``). Every sync refreshes
  tokens as needed (Xero rotates refresh tokens on each refresh), so the file on
  disk is always current.
* **Read-only, single tenant.** Requests carry your ``Xero-Tenant-Id`` header
  (auto-discovered from ``/connections`` when ``XERO_TENANT_ID`` is not set);
  the connector never writes to Xero.
* **Data minimisation.** Invoices are rendered from an allow-list
  (customer, reference, dates, status, totals, line-item descriptions and
  amounts). Contacts are **opt-in**, and contact documents only carry name,
  email, and the contact person. Nothing else is pulled into memory.

Full-snapshot syncing
---------------------
Each run is a full snapshot of the invoice (and, if enabled, contact) lists
(``write_disposition="replace"``). Xero keeps no deletion feed and bulk-updating
invoices is unusual, so a doc disappearing from the listing surface is absent
from the snapshot and cognee's ``orphan_cleanup`` forgets it on the next sync.
Unchanged rows keep a stable content-hash data_id. A fetch error aborts the run
instead of letting a partial snapshot forget anything.

Note: Xero's older ``If-Modified-Since`` incremental feed is not used because it
does not surface deletions — the full-snapshot model is committed to instead,
which is strictly simpler and correct for forget-on-delete.
"""

import json
import logging
import os
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.parse import parse_qs, urlencode

import httpx

logger = logging.getLogger("xero_connector")

XERO_SOURCE_NAME = "xero"
XERO_TABLE_INVOICES = "xero_invoices"
XERO_TABLE_CONTACTS = "xero_contacts"

_TOKEN_URL = "https://identity.xero.com/connect/token"
_AUTH_URL = "https://login.xero.com/identity/connect/authorize"
_CONNECTIONS_URL = "https://api.xero.com/connections"
_API_BASE = "https://api.xero.com/api.xro/2.0"

_DEFAULT_REDIRECT = "http://localhost:8642/callback"
_DEFAULT_SCOPES = "accounting.transactions offline_access"

_EXTRA_HINT = (
    'The Xero connector requires the "xero" extra: '
    'pip install "cognee[xero]" (provides dlt, httpx, cognee).'
)


class XeroTokenError(RuntimeError):
    """Raised when no usable OAuth token(s) are available."""


class XeroAPIError(httpx.HTTPStatusError):
    """Raised for Xero API errors after the refresh-and-retry is exhausted."""


def _basic_auth(client_id: str, client_secret: str) -> str:
    import base64

    raw = f"{client_id}:{client_secret}".encode("ascii")
    return f"Basic {base64.b64encode(raw).decode('ascii')}"


def _load_tokens(path: str) -> dict[str, Any]:
    if not path or not os.path.exists(path):
        raise XeroTokenError(
            f"No Xero tokens at {path or 'XERO_TOKEN_PATH'}. Run `xero_authenticate()` "
            "once (or see examples/example.py) to log in."
        )
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if not data.get("access_token"):
        raise XeroTokenError(f"Token file {path} contains no access_token.")
    return data


def _save_tokens(path: str, data: dict[str, Any]) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _env_credentials(client_id, client_secret):
    cid = client_id or os.environ.get("XERO_CLIENT_ID")
    secret = client_secret or os.environ.get("XERO_CLIENT_SECRET")
    if not cid or not secret:
        raise ValueError(
            "Xero OAuth app credentials required: set XERO_CLIENT_ID and "
            "XERO_CLIENT_SECRET (create an app at https://developer.xero.com/app/manage)"
            " or pass them to `xero_authenticate(...)`."
        )
    return cid, secret


class _CallbackServer:
    """Tiny local HTTP server that captures the OAuth ``code`` redirect."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8642) -> None:
        self.code: str | None = None
        self.error: str | None = None
        self._done = threading.Event()
        self._httpd = HTTPServer((host, port), self._handler_factory())

    def _handler_factory(self):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                query = parse_qs(self.path.split("?", 1)[1] if "?" in self.path else "")
                if "code" in query:
                    outer.code = query["code"][0]
                elif "error" in query:
                    outer.error = query["error"][0]
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"<h1>cognee xero</h1><p>You can close this tab.</p>")
                outer._done.set()

            def log_message(self, *args):  # silence default stderr logging
                del args

        return Handler

    def wait(self, timeout: float = 300.0) -> str:
        try:
            self._httpd.handle_request()
            if self.code is None and self.error is None and not self._done.is_set():
                self.wait(timeout=timeout)
        except KeyboardInterrupt:
            raise
        finally:
            self._httpd.server_close()
        if self.error:
            raise XeroTokenError(f"Xero auth denied: {self.error}")
        if not self.code:
            raise XeroTokenError("Timed out waiting for the Xero authorization callback.")
        return self.code


def xero_authenticate(
    token_path: str | None = None,
    *,
    client_id: str | None = None,
    client_secret: str | None = None,
    redirect_uri: str | None = None,
    scopes: str | None = None,
) -> str:
    """Run the OAuth2 authorization-code flow and persist tokens.

    Opens the browser, listens for the redirect on a local port, exchanges the
    authorization code for an access + refresh token pair, and stores them
    (``access_token``, ``refresh_token``, ``expires_at``) in ``token_path``
    (env ``XERO_TOKEN_PATH``, default ``xero_tokens.json``).

    Returns the path of the written token file.
    """
    cid, secret = _env_credentials(client_id, client_secret)
    redirect = redirect_uri or os.environ.get("XERO_REDIRECT_URI") or _DEFAULT_REDIRECT
    out_path = token_path or os.environ.get("XERO_TOKEN_PATH") or "xero_tokens.json"
    scope = scopes or os.environ.get("XERO_SCOPES") or _DEFAULT_SCOPES

    state = os.urandom(16).hex()
    query = urlencode(
        {
            "response_type": "code",
            "client_id": cid,
            "redirect_uri": redirect,
            "scope": scope,
            "state": state,
        }
    )
    server = _CallbackServer()
    webbrowser.open(f"{_AUTH_URL}?{query}")
    try:
        code = server.wait()
    except XeroTokenError:
        raise

    resp = httpx.post(
        _TOKEN_URL,
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect,
        },
        headers={"Authorization": _basic_auth(cid, secret)},
    )
    resp.raise_for_status()
    body = resp.json()
    _save_tokens(
        out_path,
        {
            "access_token": body["access_token"],
            "refresh_token": body["refresh_token"],
            "token_type": body.get("token_type", "Bearer"),
            "expires_at": time.time(),
        },
    )
    logger.info("Xero: tokens stored at %s", out_path)
    return out_path


class XeroClient:
    """Bearer-token client for a single Xero organisation."""

    def __init__(
        self,
        token_path: str,
        *,
        tenant_id: str | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._token_path = token_path
        self._token = _load_tokens(token_path)
        self._tenant_id = tenant_id
        self._http = httpx.Client(transport=transport)

    def close(self) -> None:
        self._http.close()

    @property
    def tenant_id(self) -> str | None:
        return self._tenant_id

    def ensure_tenant(self) -> str:
        """Pick a tenant (from env, or discover the first via /connections)."""
        if self._tenant_id:
            return self._tenant_id
        resp = self._http.get(_CONNECTIONS_URL, headers=self._bearer(tenant=False))
        resp.raise_for_status()
        connections = resp.json()
        if not connections:
            raise XeroTokenError("No Xero connections found — is the app connected to an org?")
        self._tenant_id = connections[0]["tenantId"]
        return self._tenant_id

    def _bearer(self, *, tenant: bool) -> dict[str, str]:
        scheme = self._token.get("token_type", "Bearer")
        headers = {"Authorization": f"{scheme} {self._token['access_token']}"}
        if tenant:
            headers["Xero-Tenant-Id"] = self.ensure_tenant()
        return headers

    def _refresh(self) -> None:
        """Refresh with rotation (Xero issues a new refresh token every time)."""
        resp = self._http.post(
            _TOKEN_URL,
            data={
                "grant_type": "refresh_token",
                "refresh_token": self._token["refresh_token"],
            },
            headers={"Authorization": _basic_auth(*self._env_creds())},
        )
        resp.raise_for_status()
        body = resp.json()
        self._token = {
            "access_token": body["access_token"],
            "refresh_token": body.get("refresh_token") or self._token["refresh_token"],
            "token_type": body.get("token_type", "Bearer"),
            "expires_at": time.time(),
        }
        _save_tokens(self._token_path, self._token)
        logger.info("Xero: refreshed OAuth token (rotated).")

    def _env_creds(self) -> tuple[str, str]:
        return _env_credentials(None, None)

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = _API_BASE + path
        for attempt in range(2):
            resp = self._http.get(url, params=params, headers=self._bearer(tenant=True))
            if resp.status_code == 401 and attempt == 0:
                self._refresh()
                continue
            if resp.status_code < 400:
                return resp.json()
            self._raise_if_error(resp)
        raise RuntimeError("unreachable")

    @staticmethod
    def _raise_if_error(resp: httpx.Response) -> None:
        try:
            body = resp.json()
        except ValueError:
            body = resp.text
        raise XeroAPIError(
            f"Xero API error {resp.status_code}: {body}",
            request=resp.request,
            response=resp,
        )


def _iter_items(client, path: str, item_key: str, statuses: list[str] | None = None):
    """Page through an item list (Xero returns ≤100 items per page)."""
    params: dict[str, Any] = {}
    if statuses:
        params["Statuses"] = ",".join(statuses)
    page = 1
    while True:
        payload = client.get(path, {**params, "page": page})
        items = payload.get(item_key) or []
        yield from items
        if len(items) < 100:
            return
        page += 1


def _line_items(invoice: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for line in invoice.get("LineItems") or []:
        description = (line.get("Description") or "").strip()
        quantity = line.get("Quantity")
        unit = line.get("UnitAmount")
        amount = line.get("LineAmount")
        parts = []
        if description:
            parts.append(description)
        if quantity is not None and unit is not None:
            parts.append(f"{quantity} x {unit:.2f}")
        if amount is not None:
            parts.append(f"= {amount:.2f}")
        if parts:
            lines.append("- " + " ".join(parts))
    return lines


def _customer(invoice: dict[str, Any]) -> dict[str, Any]:
    contact = invoice.get("Contact") or {}
    return contact


def _invoice_to_row(invoice: dict[str, Any]) -> dict[str, Any]:
    """Render an invoice to a document row via an allow-list only."""
    fields: list[str] = []
    for label, key in (
        ("Status", "Status"),
        ("Type", "Type"),
        ("Reference", "Reference"),
        ("Currency", "CurrencyCode"),
        ("Date", "Date"),
        ("Due date", "DueDate"),
        ("Currency rate", "CurrencyRate"),
    ):
        value = invoice.get(key)
        if value:
            fields.append(f"{label}: {value}")

    customer = _customer(invoice)
    if customer.get("Name"):
        fields.append(f"Customer: {customer['Name']}")
    if customer.get("EmailAddress"):
        fields.append(f"Customer email: {customer['EmailAddress']}")

    for label, key in (("Subtotal", "SubTotal"), ("Tax", "TaxAmount"), ("Total", "Total")):
        value = invoice.get(key)
        if value is not None:
            fields.append(f"{label}: {value:.2f}")

    items = _line_items(invoice)
    if items:
        fields.append("Line items:")
        fields.extend(items)

    invoice_number = invoice.get("InvoiceNumber") or str(invoice.get("InvoiceID"))
    return {
        "id": invoice.get("InvoiceID"),
        "url": f"https://go.xero.com/invoice/{invoice.get('InvoiceID')}",
        "title": f"Invoice {invoice_number}",
        "content": "\n".join(fields),
    }


def _contact_to_row(contact: dict[str, Any]) -> dict[str, Any]:
    fields: list[str] = []
    for label, key in (
        ("Name", "Name"),
        ("Contact person", "ContactFirstName"),
        ("Last name", "ContactLastName"),
        ("Email", "EmailAddress"),
    ):
        value = contact.get(key)
        if value:
            fields.append(f"{label}: {value}")
    return {
        "id": contact.get("ContactID"),
        "url": f"https://go.xero.com/contact/{contact.get('ContactID')}",
        "title": f"Contact {contact.get('Name')}",
        "content": "\n".join(fields),
    }


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def xero_source(
    client: XeroClient | None = None,
    *,
    token_path: str | None = None,
    tenant_id: str | None = None,
    include_contacts: bool = False,
    invoice_statuses: list[str] | None = None,
):
    """Create a dlt source that yields Xero documents for ``remember``.

    Args:
        client: Pre-built :class:`XeroClient` (test injection point); when
            omitted one is built from a token file. See
            :func:`xero_authenticate` for the one-time login.
        token_path: Path to the OAuth token file (env ``XERO_TOKEN_PATH``,
            default ``xero_tokens.json``).
        tenant_id: Xero tenant (organisation) id (env ``XERO_TENANT_ID``);
            auto-discovered from ``/connections`` if omitted.
        include_contacts: When False (default) contacts are not synced. Contact
            documents carry name/email/person only.
        invoice_statuses: Restrict to e.g. ``["AUTHORISED"]``; ``None`` = all.

    Returns:
        A dlt source suitable for ``cognee.add(...)`` / ``cognee.remember(...)``.
        Resources: ``xero_invoices`` and (only when ``include_contacts``)
        ``xero_contacts``.
    """
    try:
        import dlt
    except ImportError as exc:
        raise ImportError(_EXTRA_HINT) from exc

    if client is None:
        token_path = token_path or os.environ.get("XERO_TOKEN_PATH") or "xero_tokens.json"
        tenant_id = tenant_id or os.environ.get("XERO_TENANT_ID")
        client = XeroClient(token_path, tenant_id=tenant_id)
    client.ensure_tenant()

    @dlt.resource(name=XERO_TABLE_INVOICES, primary_key="id", write_disposition="replace")
    def xero_invoices():
        count = 0
        for invoice in _iter_items(client, "/Invoices", "Invoices", invoice_statuses):
            yield _invoice_to_row(invoice)
            count += 1
        logger.info("Xero: synced %d invoice(s).", count)

    @dlt.resource(name=XERO_TABLE_CONTACTS, primary_key="id", write_disposition="replace")
    def xero_contacts():
        count = 0
        for contact in _iter_items(client, "/Contacts", "Contacts"):
            yield _contact_to_row(contact)
            count += 1
        logger.info("Xero: synced %d contact(s).", count)

    @dlt.source(name=XERO_SOURCE_NAME)
    def _xero():
        if include_contacts:
            return [xero_invoices, xero_contacts]
        return [xero_invoices]

    source = _xero()
    from cognee.tasks.ingestion.dlt_utils import DOCUMENT_SOURCE_ATTR

    setattr(source, DOCUMENT_SOURCE_ATTR, XERO_SOURCE_NAME)
    return source


__all__ = [
    "XERO_SOURCE_NAME",
    "XERO_TABLE_CONTACTS",
    "XERO_TABLE_INVOICES",
    "XeroAPIError",
    "XeroClient",
    "XeroTokenError",
    "xero_authenticate",
    "xero_source",
]
