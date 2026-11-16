"""The REST client: key and subdomain validation, pacing, and status mapping.

Nothing here hits the network: ``requests.request`` is replaced with a recorder.
"""

import base64

import pytest
import requests

from cognee_connector_bamboohr.source import (
    _MIN_REQUEST_INTERVAL,
    BASE_URL,
    BambooHRAccessError,
    BambooHRAPIError,
    BambooHRAuthError,
    BambooHRClient,
    BambooHRNotFoundError,
    BambooHRRateLimitedError,
    BambooHRResponseError,
)


class _Answer:
    def __init__(self, status: int, payload=None, bad_json: bool = False):
        self.status_code = status
        self.headers = {}
        self._payload = payload
        self._bad_json = bad_json

    def json(self):
        if self._bad_json:
            raise ValueError("no json")
        return self._payload


class _Recorder:
    """The ``requests.request`` stand-in: queues answers, records calls."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = []

    def __call__(self, method, url, **kwargs):
        self.calls.append({"method": method, "url": url, **kwargs})
        if not self.answers:
            raise AssertionError("the client made more requests than the test queued")
        failure = self.answers.pop(0)
        if isinstance(failure, Exception):
            raise failure
        return failure


class _Clock:
    """A monotonic clock the pacing sleeps advance, so waits are exact."""

    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def client(monkeypatch):
    recorder = _Recorder(_Answer(200, {}))
    monkeypatch.setattr(requests, "request", recorder)
    clock = _Clock()
    bamboohr = BambooHRClient("testkey", "acme", sleep=clock.sleep, monotonic=clock.monotonic)
    return bamboohr, recorder, clock


# -- construction -----------------------------------------------------------
@pytest.mark.parametrize(
    "key",
    ["", "   ", "key with spaces", "key\nwith\nnewlines", "k€y"],
)
def test_an_empty_or_unprintable_key_is_refused(key):
    with pytest.raises(ValueError, match="API key"):
        BambooHRClient(key, "acme")


@pytest.mark.parametrize(
    "subdomain",
    ["", "Acme", "ac me", "acme.example.com", "acme/x", "-acme-", "acme_"],
)
def test_an_invalid_subdomain_is_refused(subdomain):
    with pytest.raises(ValueError, match="subdomain"):
        BambooHRClient("testkey", subdomain)


def test_a_valid_key_and_subdomain_build():
    client = BambooHRClient("testkey", "acme-inc2")
    assert repr(client) == "BambooHRClient(api_key=<redacted>)"


# -- the request --------------------------------------------------------------
def test_the_key_goes_out_as_basic_auth_and_never_into_logs(client):
    bamboohr, recorder, _ = client
    bamboohr.request("GET", "/meta/fields")
    headers = recorder.calls[0]["headers"]
    expected = "Basic " + base64.b64encode(b"testkey:x").decode()
    assert headers["Authorization"] == expected
    assert "testkey" not in repr(bamboohr)


def test_the_url_names_the_subdomain_and_the_path(client):
    bamboohr, recorder, _ = client
    bamboohr.request("GET", "/meta/fields", params={"a": "1"})
    call = recorder.calls[0]
    assert call["url"] == f"{BASE_URL}/acme/v1/meta/fields"
    assert call["params"] == {"a": "1"}
    # A redirect is an error, never followed: the answer may name another host.
    assert call["allow_redirects"] is False
    assert call["timeout"] == (10, 30)


def test_a_200_returns_the_json_payload(client):
    bamboohr, _ = recorder_with(client, _Answer(200, {"ok": True}))
    assert bamboohr.request("GET", "/meta/fields") == {"ok": True}


def test_a_200_without_json_is_a_response_error(client):
    bamboohr, _ = recorder_with(client, _Answer(200, bad_json=True))
    with pytest.raises(BambooHRResponseError, match="HTTP 200 without JSON"):
        bamboohr.request("GET", "/meta/fields")


@pytest.mark.parametrize(
    ("status", "error"),
    [
        (302, BambooHRAPIError),
        (400, BambooHRAPIError),
        (401, BambooHRAuthError),
        (403, BambooHRAccessError),
        (404, BambooHRNotFoundError),
        (429, BambooHRRateLimitedError),
    ],
)
def test_statuses_map_to_errors_that_carry_status_and_endpoint(client, status, error):
    bamboohr, _ = recorder_with(client, _Answer(status, {"message": "boom"}))
    with pytest.raises(error) as raised:
        bamboohr.request("GET", "/meta/fields")
    assert raised.value.status == status
    assert "/meta/fields" in str(raised.value)
    assert "boom" not in str(raised.value)


def test_an_error_message_never_carries_the_key(client):
    bamboohr, _ = recorder_with(client, _Answer(400, {"message": "testkey is wrong"}))
    with pytest.raises(BambooHRAPIError) as raised:
        bamboohr.request("GET", "/meta/fields")
    assert "testkey" not in str(raised.value)


def _no_wait_client(monkeypatch, *answers):
    """A client on a fake clock: requests never wait, and answers are queued."""
    recorder = _Recorder(*answers)
    monkeypatch.setattr(requests, "request", recorder)
    clock = _Clock()
    return BambooHRClient("testkey", "acme", sleep=clock.sleep, monotonic=clock.monotonic), recorder


def test_a_5xx_is_retried_then_reported_by_status(client, monkeypatch):
    bamboohr, recorder, _ = client
    recorder.answers = [
        _Answer(500, {"error": "x"}),
        _Answer(502, {"error": "x"}),
        _Answer(503, {"error": "x"}),
    ]
    with pytest.raises(BambooHRAPIError) as raised:
        bamboohr.request("GET", "/meta/fields")
    assert raised.value.status == 503
    assert len(recorder.calls) == 3


def test_a_network_error_is_retried_then_reported_without_an_answer(client):
    bamboohr, recorder, _ = client
    recorder.answers = [
        requests.ConnectionError("down"),
        requests.Timeout("slow"),
        requests.ConnectionError("down"),
    ]
    with pytest.raises(BambooHRAPIError) as raised:
        bamboohr.request("GET", "/meta/fields")
    assert raised.value.status == 0
    assert "no answer" in str(raised.value)
    assert len(recorder.calls) == 3


def test_a_4xx_is_not_retried(client):
    bamboohr, recorder, _ = client
    recorder.answers = [_Answer(403, {})]
    with pytest.raises(BambooHRAccessError):
        bamboohr.request("GET", "/meta/fields")
    assert len(recorder.calls) == 1


# -- pacing -------------------------------------------------------------------
def test_requests_are_paced_to_one_per_interval(monkeypatch):
    recorder = _Recorder(_Answer(200, {}), _Answer(200, {}), _Answer(200, {}))
    monkeypatch.setattr(requests, "request", recorder)
    clock = _Clock()
    bamboohr = BambooHRClient("testkey", "acme", sleep=clock.sleep, monotonic=clock.monotonic)

    bamboohr.request("GET", "/meta/fields")
    assert clock.sleeps == []  # the first request is not paced

    bamboohr.request("GET", "/meta/fields")
    assert clock.sleeps == [pytest.approx(_MIN_REQUEST_INTERVAL)]

    clock.now += 0.5  # 0.5s already passed since the last request
    bamboohr.request("GET", "/meta/fields")
    assert clock.sleeps[-1] == pytest.approx(_MIN_REQUEST_INTERVAL - 0.5)


def recorder_with(client, *answers):
    """Swap the queued answers; pytest's ``client`` fixture queues one 200 already."""
    bamboohr, recorder, _ = client
    recorder.answers = list(answers)
    return bamboohr, recorder
