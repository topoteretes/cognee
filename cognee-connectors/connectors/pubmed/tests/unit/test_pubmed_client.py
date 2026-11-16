"""The E-utilities client: spacing, errors, retries and what never leaks.

The client is driven against ``requests.request`` patched to a fake NCBI over HTTP, so
throttling, redirects, 429s, the rejected-key 400 and 5xx retries are observed the way
the walker would see them — with no network.
"""

import pytest
from fake_pubmed import FakePubMed, serve_over_http

from cognee_connector_pubmed.source import (
    EUTILS_URL,
    PubMedAPIError,
    PubMedAuthError,
    PubMedClient,
    PubMedRateLimitedError,
)

KEY = "SECRET-pubmed-key-123"


class _HTTPResponse:
    def __init__(self, status: int, payload=None, text: str = ""):
        self.status_code = status
        self._payload = payload
        self.text = text
        self.headers = {}

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def _client(**kwargs) -> PubMedClient:
    return PubMedClient(KEY, sleep=lambda _: None, monotonic=_Clock(), **kwargs)


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_the_key_is_never_in_the_representation_and_a_bad_key_is_refused():
    client = _client()
    assert KEY not in repr(client)
    with pytest.raises(ValueError, match="invalid format"):
        PubMedClient("not a key")


def test_a_request_sends_the_key_as_a_query_param(monkeypatch):
    keys = []
    monkeypatch.setattr("requests.request", serve_over_http(FakePubMed(), keys=keys))

    _client().request("GET", "/esearch.fcgi", params={"db": "pubmed"})

    assert keys == [KEY]


def test_the_client_spaces_requests_by_the_key_it_holds(monkeypatch):
    sleeps: list[float] = []
    clock = _Clock()
    calls = []

    def sleep(seconds):
        sleeps.append(seconds)
        clock.now += seconds

    def fake_request(method, url, **kwargs):
        calls.append(clock.now)
        return _HTTPResponse(200, {"esearchresult": {"count": "0", "idlist": []}})

    monkeypatch.setattr("requests.request", fake_request)
    without = PubMedClient(sleep=sleep, monotonic=clock)
    with_key = PubMedClient(KEY, sleep=sleep, monotonic=clock)

    without.request("GET", "/esearch.fcgi")
    with_key.request("GET", "/esearch.fcgi")
    without.request("GET", "/esearch.fcgi")

    assert sleeps == [pytest.approx(0.4), pytest.approx(0.12), pytest.approx(0.4)]
    assert calls == [pytest.approx(0.4), pytest.approx(0.52), pytest.approx(0.92)]


def test_a_429_raises_the_rate_limited_error(monkeypatch):
    def fake_request(method, url, **kwargs):
        return _HTTPResponse(429, {"error": "rate limit"})

    monkeypatch.setattr("requests.request", fake_request)

    with pytest.raises(PubMedRateLimitedError, match="429"):
        _client().request("GET", "/esearch.fcgi")


def test_a_400_naming_the_api_key_is_an_auth_error(monkeypatch):
    def fake_request(method, url, **kwargs):
        return _HTTPResponse(400, {"error": "API key invalid"})

    monkeypatch.setattr("requests.request", fake_request)

    with pytest.raises(PubMedAuthError, match="API key"):
        _client().request("GET", "/esearch.fcgi")


def test_a_400_without_the_key_in_its_message_is_a_plain_api_error(monkeypatch):
    def fake_request(method, url, **kwargs):
        return _HTTPResponse(400, {"error": "bad term"})

    monkeypatch.setattr("requests.request", fake_request)

    with pytest.raises(PubMedAPIError, match="HTTP 400") as excinfo:
        _client().request("GET", "/esearch.fcgi")
    assert "SECRET" not in str(excinfo.value)


def test_a_redirect_is_an_error_and_is_not_followed(monkeypatch):
    seen = []

    def fake_request(method, url, **kwargs):
        seen.append((url, kwargs.get("allow_redirects")))
        return _HTTPResponse(302, None)

    monkeypatch.setattr("requests.request", fake_request)

    with pytest.raises(PubMedAPIError, match="redirect"):
        _client().request("GET", "/esearch.fcgi")

    assert seen == [(EUTILS_URL + "/esearch.fcgi", False)]


def test_a_5xx_is_retried_then_reported(monkeypatch):
    statuses = iter([500, 502, 503])
    sleeps: list[float] = []

    def fake_request(method, url, **kwargs):
        return _HTTPResponse(next(statuses), None)

    monkeypatch.setattr("requests.request", fake_request)
    client = PubMedClient(KEY, sleep=sleeps.append, monotonic=_Clock())

    with pytest.raises(PubMedAPIError, match="HTTP 503"):
        client.request("GET", "/esearch.fcgi")
    assert sleeps[1::2] == [1.0, 2.0], "backoff 1s then 2s between the three attempts"
    assert len(sleeps) == 5, "every attempt is also spaced by the throttle"


def test_a_network_error_is_reported_by_type(monkeypatch):
    import requests

    def fake_request(method, url, **kwargs):
        raise requests.ConnectionError("boom")

    monkeypatch.setattr("requests.request", fake_request)

    with pytest.raises(PubMedAPIError, match="ConnectionError"):
        _client().request("GET", "/esearch.fcgi")


def test_a_500_on_the_last_retry_is_parsed_like_any_other_answer(monkeypatch):
    # A 5xx that survives every attempt is read once more at raise time; a readable error
    # body there is still an API error, not a crash.
    def fake_request(method, url, **kwargs):
        return _HTTPResponse(500, {"error": "upstream"})

    monkeypatch.setattr("requests.request", fake_request)
    client = PubMedClient(KEY, sleep=lambda _: None, monotonic=_Clock())

    with pytest.raises(PubMedAPIError, match="HTTP 500"):
        client.request("GET", "/esearch.fcgi")


def test_efetch_asks_for_the_xml_as_text(monkeypatch):
    fake = FakePubMed()
    fake.add_article("100", "Sleep and memory")
    texts = []

    def fake_request(method, url, **kwargs):
        payload = serve_over_http(fake)(method, url, **kwargs)
        texts.append(payload.text)
        return payload

    monkeypatch.setattr("requests.request", fake_request)

    payload = _client().request(
        "GET", "/efetch.fcgi", params={"db": "pubmed", "id": "100", "retmode": "xml"}, as_text=True
    )

    assert "<PubmedArticle>" in payload
    assert texts and "<PubmedArticle>" in texts[0]
