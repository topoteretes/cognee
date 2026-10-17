"""Unit tests for the Xero dlt connector.

Two layers, all runnable in CI without a Xero org:

* DB-free tests for rendering (allow-list, line-item spelling), paging, token
  loading/rotate-on-401 (httpx MockTransport), and the document-source marker.
* dlt-pipeline tests (temp sqlite destination) covering initial load, edit
  re-sync, forget-on-delete, the contacts opt-in gate, and credential guards.
"""

import httpx
import pytest
from cognee.tasks.ingestion.dlt_utils import document_source_tag

from cognee_community_connector_xero.xero import (
    XERO_SOURCE_NAME,
    XERO_TABLE_CONTACTS,
    XERO_TABLE_INVOICES,
    XeroAPIError,
    XeroClient,
    XeroTokenError,
    _invoice_to_row,
    _iter_items,
    _load_tokens,
    _save_tokens,
)


def _invoice(invoice_id, number="INV-001", customer="Acme", **extra):
    invoice = {
        "InvoiceID": invoice_id,
        "InvoiceNumber": number,
        "Type": "ACCREC",
        "Status": "AUTHORISED",
        "Reference": "PO-123",
        "CurrencyCode": "USD",
        "Date": "2026-01-01",
        "DueDate": "2026-02-01",
        "SubTotal": 21.0,
        "TaxAmount": 2.1,
        "Total": 23.1,
        "Contact": {
            "ContactID": "contact_1",
            "Name": customer,
            "EmailAddress": "billing@acme.test",
        },
        "LineItems": [
            {
                "Description": "Consulting",
                "Quantity": 2,
                "UnitAmount": 10.5,
                "LineAmount": 21.0,
            }
        ],
    }
    invoice.update(extra)
    return invoice


def _contact(contact_id, name="Acme", email="hi@acme.test"):
    return {"ContactID": contact_id, "Name": name, "EmailAddress": email}


class FakeXeroClient:
    """Stand-in for XeroClient backed by in-memory fixtures."""

    def __init__(self, invoices=None, contacts=None, tenant_id="tenant_1"):
        self._invoices = invoices or []
        self._contacts = contacts or []
        self.tenant_id = tenant_id
        self.calls = []

    def ensure_tenant(self):
        return self.tenant_id

    def get(self, path, params=None):
        self.calls.append((path, params or {}))
        page = (params or {}).get("page", 1)
        if path == "/Invoices":
            payload = self._invoices
        elif path == "/Contacts":
            payload = self._contacts
        else:
            payload = []
        start = (page - 1) * 100
        chunk = payload[start : start + 100]
        key = "Invoices" if path == "/Invoices" else "Contacts"
        return {key: chunk}


# ---------------------------------------------------------------------------
# Rendering (DB-free)
# ---------------------------------------------------------------------------


def test_invoice_row_renders_allowlist():
    row = _invoice_to_row(_invoice("inv_1", number="INV-001", customer="Acme"))
    assert row["id"] == "inv_1"
    assert row["title"] == "Invoice INV-001"
    assert "Customer: Acme" in row["content"]
    assert "Customer email: billing@acme.test" in row["content"]
    assert "Reference: PO-123" in row["content"]
    assert "Status: AUTHORISED" in row["content"]
    assert "Total: 23.10" in row["content"]
    assert "- Consulting 2 x 10.50 = 21.00" in row["content"]


def test_invoice_row_omits_non_allowlist_fields():
    row = _invoice_to_row(
        _invoice(
            "inv_2",
            BankAccountDetails={"AccountNumber": "11111111"},
            Contact={"Name": "Acme", "EmailAddress": "b@acme.test"},
        )
    )
    assert "11111111" not in row["content"]
    assert "AccountNumber" not in row["content"]


def test_line_items_spelling():
    row = _invoice_to_row(
        _invoice(
            "inv_3",
            LineItems=[
                {
                    "Description": "Build memories",
                    "Quantity": 1,
                    "UnitAmount": 9.0,
                    "LineAmount": 9.0,
                }
            ],
        )
    )
    assert "- Build memories 1 x 9.00 = 9.00" in row["content"]


def test_iter_items_pages_while_full_pages_returned():
    invoices = [_invoice(f"inv_{i}") for i in range(120)]
    client = FakeXeroClient(invoices=invoices)
    collected = list(_iter_items(client, "/Invoices", "Invoices"))
    assert len(collected) == 120
    pages = [params.get("page") for _, params in client.calls]
    assert pages == [1, 2]


# ---------------------------------------------------------------------------
# Token lifecycle
# ---------------------------------------------------------------------------


def test_client_requires_token_file(tmp_path):
    with pytest.raises(XeroTokenError, match="No Xero tokens"):
        XeroClient(str(tmp_path / "missing.json"))


def test_client_rotates_refresh_token_on_401(tmp_path, monkeypatch):
    monkeypatch.setenv("XERO_CLIENT_ID", "demo-id")
    monkeypatch.setenv("XERO_CLIENT_SECRET", "demo-secret")
    token_path = tmp_path / "tokens.json"
    _save_tokens(
        token_path,
        {
            "access_token": "old-access",
            "refresh_token": "old-refresh",
            "token_type": "Bearer",
            "expires_at": 0,
        },
    )

    state = {"refreshed": False}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/connect/token":
            state["refreshed"] = True
            return httpx.Response(
                200,
                json={
                    "access_token": "new-access",
                    "refresh_token": "new-refresh",
                    "token_type": "Bearer",
                },
            )
        if request.headers.get("Authorization") == "Bearer old-access":
            return httpx.Response(401, json={"message": "token expired"})
        return httpx.Response(200, json={"Invoices": [{"InvoiceID": "inv_1"}]})

    client = XeroClient(
        str(token_path),
        tenant_id="tenant_1",
        transport=httpx.MockTransport(handler),
    )
    payload = client.get("/Invoices", {"page": 1})
    assert payload["Invoices"][0]["InvoiceID"] == "inv_1"
    assert state["refreshed"] is True

    saved = _load_tokens(str(token_path))
    assert saved["access_token"] == "new-access"
    assert saved["refresh_token"] == "new-refresh"  # rotated on disk


def test_client_gives_up_after_401_retry(tmp_path, monkeypatch):
    monkeypatch.setenv("XERO_CLIENT_ID", "demo-id")
    monkeypatch.setenv("XERO_CLIENT_SECRET", "demo-secret")
    token_path = tmp_path / "tokens.json"
    _save_tokens(
        token_path,
        {
            "access_token": "t1",
            "refresh_token": "r1",
            "token_type": "Bearer",
            "expires_at": 0,
        },
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/connect/token":
            return httpx.Response(200, json={"access_token": "t2", "refresh_token": "r2"})
        return httpx.Response(401, json={"message": "unauthorized"})

    with pytest.raises(XeroAPIError):
        XeroClient(
            str(token_path),
            tenant_id="tenant_1",
            transport=httpx.MockTransport(handler),
        ).get("/Invoices", {"page": 1})


# ---------------------------------------------------------------------------
# Source wiring
# ---------------------------------------------------------------------------


def test_xero_source_declares_document_marker():
    from cognee_community_connector_xero.xero import xero_source

    source = xero_source(client=FakeXeroClient())
    assert XERO_SOURCE_NAME == "xero"
    assert document_source_tag(source) == "xero"


def test_xero_authenticate_requires_credentials(monkeypatch):
    from cognee_community_connector_xero.xero import xero_authenticate

    monkeypatch.delenv("XERO_CLIENT_ID", raising=False)
    monkeypatch.delenv("XERO_CLIENT_SECRET", raising=False)
    with pytest.raises(ValueError, match="credentials"):
        xero_authenticate()


def _run_sync(dlt, tmp_path, fake_client, **kwargs):
    from cognee_community_connector_xero.xero import xero_source

    db_path = (tmp_path / "xero.db").as_posix()
    pipeline = dlt.pipeline(
        pipeline_name="xero_test",
        destination=dlt.destinations.sqlalchemy(f"sqlite:///{db_path}"),
        dataset_name="xero_ds",
        pipelines_dir=str(tmp_path / "state"),
    )
    pipeline.run(xero_source(client=fake_client, **kwargs))
    return pipeline


def _read_table(pipeline, table):
    with pipeline.sql_client() as client:
        rows = client.execute_sql(f"SELECT id, title, content FROM {table}")
    return {row[0]: {"id": row[0], "title": row[1], "content": row[2]} for row in rows}


@pytest.fixture
def dlt_mod():
    return pytest.importorskip("dlt")


def test_first_sync_loads_invoices(dlt_mod, tmp_path):
    client = FakeXeroClient(invoices=[_invoice("inv_1", number="INV-001")])
    pipeline = _run_sync(dlt_mod, tmp_path, client)

    invoices = _read_table(pipeline, XERO_TABLE_INVOICES)
    assert set(invoices) == {"inv_1"}
    assert "Acme" in invoices["inv_1"]["content"]


def test_contacts_are_opt_in(dlt_mod, tmp_path):
    client = FakeXeroClient(invoices=[_invoice("inv_1")], contacts=[_contact("c1")])
    pipeline = _run_sync(dlt_mod, tmp_path, client, include_contacts=False)

    with pipeline.sql_client() as conn:
        found = conn.execute_sql(
            f"SELECT name FROM sqlite_master WHERE type='table' AND name='{XERO_TABLE_CONTACTS}'"
        )
    assert found == []


def test_contacts_sync_when_opted_in(dlt_mod, tmp_path):
    client = FakeXeroClient(invoices=[_invoice("inv_1")], contacts=[_contact("c1")])
    pipeline = _run_sync(dlt_mod, tmp_path, client, include_contacts=True)

    contacts = _read_table(pipeline, XERO_TABLE_CONTACTS)
    assert set(contacts) == {"c1"}
    assert "Acme" in contacts["c1"]["content"]


def test_edit_is_reflected_on_resync(dlt_mod, tmp_path):
    client = FakeXeroClient(invoices=[_invoice("inv_1", number="INV-001", customer="Acme")])
    _run_sync(dlt_mod, tmp_path, client)

    edited = FakeXeroClient(invoices=[_invoice("inv_1", number="INV-001", customer="Globex")])
    pipeline = _run_sync(dlt_mod, tmp_path, edited)

    rows = _read_table(pipeline, XERO_TABLE_INVOICES)
    assert rows["inv_1"]["content"].count("Globex") == 1
    assert "Acme" not in rows["inv_1"]["content"]


def test_vanished_invoice_is_removed_on_resync(dlt_mod, tmp_path):
    client = FakeXeroClient(invoices=[_invoice("inv_1"), _invoice("inv_2")])
    _run_sync(dlt_mod, tmp_path, client)

    vanished = FakeXeroClient(invoices=[_invoice("inv_2")])
    pipeline = _run_sync(dlt_mod, tmp_path, vanished)

    rows = _read_table(pipeline, XERO_TABLE_INVOICES)
    assert "inv_1" not in rows
    assert "inv_2" in rows


def test_fetch_failure_aborts_sync(dlt_mod, tmp_path):
    from cognee_community_connector_xero.xero import xero_source

    class BoomClient:
        def ensure_tenant(self):
            return "tenant_1"

        def get(self, path, params=None):
            raise RuntimeError("boom")

    db_path = (tmp_path / "boom.db").as_posix()
    pipeline = dlt_mod.pipeline(
        pipeline_name="xero_boom",
        destination=dlt_mod.destinations.sqlalchemy(f"sqlite:///{db_path}"),
        dataset_name="xero_ds",
        pipelines_dir=str(tmp_path / "state"),
    )
    with pytest.raises(Exception):  # noqa: B017 - dlt wraps the source error
        pipeline.run(xero_source(client=BoomClient()))
