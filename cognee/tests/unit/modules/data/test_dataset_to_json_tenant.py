"""Regression test: Dataset.to_json() must serialize a null tenant_id as JSON
null, not the string "None".

`tenant_id` is a nullable column, so in single-tenant deployments it is None.
`str(None)` yields the literal "None", leaking a bogus string into the payload.
The same method already handles the nullable `updated_at` correctly with
`... if ... else None`; this pins tenant_id to that same convention.
"""

from datetime import datetime, timezone

from cognee.modules.data.models.Dataset import Dataset


def _make_dataset(tenant_id):
    d = Dataset()
    d.id = "00000000-0000-0000-0000-000000000001"
    d.name = "ds"
    d.created_at = datetime(2024, 1, 1, tzinfo=timezone.utc)
    d.updated_at = None
    d.owner_id = "00000000-0000-0000-0000-000000000002"
    d.tenant_id = tenant_id
    d.data = []
    return d


def test_null_tenant_serializes_as_none_not_string():
    payload = _make_dataset(None).to_json()
    assert payload["tenantId"] is None  # was the literal string "None" before the fix


def test_present_tenant_still_serialized_as_string():
    tid = "00000000-0000-0000-0000-000000000003"
    payload = _make_dataset(tid).to_json()
    assert payload["tenantId"] == str(tid)
