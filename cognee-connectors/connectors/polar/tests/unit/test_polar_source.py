import pytest
from cognee_connector_polar.source import polar_source, _PolarClient, PolarAPIError


class FakeResponse:
    def __init__(self, status_code: int, payload: dict, text: str = ""):
        self.status_code = status_code
        self._payload = payload
        self.text = text or str(payload)
        self.headers = {}

    def json(self):
        return self._payload


class FakeHTTPClient:
    def __init__(self):
        self.requests = []

    def request(self, method: str, url: str, params=None, json=None, headers=None):
        self.requests.append((method, url, params, json, headers))
        if "customers" in url:
            return FakeResponse(
                200,
                {
                    "items": [
                        {"id": "cust_123", "email": "test@example.com", "name": "Test User"}
                    ],
                    "pagination": {"max_page": 1},
                },
            )
        if "subscriptions" in url:
            return FakeResponse(
                200,
                {
                    "items": [
                        {
                            "id": "sub_456",
                            "status": "active",
                            "customer_id": "cust_123",
                            "product": {"name": "Pro Plan"},
                        }
                    ],
                    "pagination": {"max_page": 1},
                },
            )
        if "customer-portal/benefit-grants" in url:
            return FakeResponse(
                200,
                {
                    "items": [
                        {
                            "id": "grant_789",
                            "benefit_id": "ben_1",
                            "customer_id": "cust_123",
                            "benefit": {"type": "discord"},
                            "is_granted": True,
                        }
                    ],
                    "pagination": {"max_page": 1},
                },
            )
        return FakeResponse(404, {"detail": "Not found"})


def test_polar_source_token_missing():
    with pytest.raises(ValueError, match="Polar access token required"):
        polar_source(token="", root_page_ids=None)


def test_polar_source_extraction():
    fake_http = FakeHTTPClient()
    resource = polar_source(
        token="polar_pat_12345",
        organization_id="org_test",
        http_client=fake_http,
    )

    rows = list(resource)
    assert len(rows) == 3

    customer_doc = next(r for r in rows if r["id"] == "polar_customer_cust_123")
    assert "Test User" in customer_doc["title"]
    assert "test@example.com" in customer_doc["content"]

    sub_doc = next(r for r in rows if r["id"] == "polar_subscription_sub_456")
    assert "Pro Plan" in sub_doc["title"]
    assert "active" in sub_doc["content"]

    grant_doc = next(r for r in rows if r["id"] == "polar_grant_grant_789")
    assert "discord" in grant_doc["title"]
