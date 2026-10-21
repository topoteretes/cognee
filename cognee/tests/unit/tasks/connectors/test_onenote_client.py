# Adapted from topoteretes/cognee-community (Apache-2.0), PR #301, commit 4339aa4.
# Modified to ship in the Cognee SDK with SDK-local imports and test paths.
"""The Graph client: retries, wait budget, redirects, unsafe links and what an error may carry.

Every case drives the real client through ``httpx.MockTransport``, so the token and URL handling
that builds an error message is the code under test. No network.
"""

import httpx
import pytest
from fake_onenote import Graph

from cognee.tasks.ingestion.connectors import list_onenote_notebooks, onenote
from cognee.tasks.ingestion.connectors.onenote import OneNoteError


def test_notebook_listing_uses_delegated_token_provider():
    graph = Graph()
    with graph.client() as client:
        result = list_onenote_notebooks(lambda: "offline-token", http_client=client)
    assert {item["id"] for item in result} == {"a", "b"}


@pytest.mark.parametrize("status", [429, 503])
def test_retry_after_and_five_total_attempts(monkeypatch, status):
    from cognee.tasks.ingestion.connectors import onenote

    sleeps = []
    monkeypatch.setattr(onenote.time, "sleep", sleeps.append)
    requests = []

    def throttled(request):
        requests.append(request)
        return httpx.Response(
            status, headers={"Retry-After": "2"}, json={"error": {"code": "throttled"}}
        )

    with (
        httpx.Client(transport=httpx.MockTransport(throttled)) as client,
        pytest.raises(OneNoteError),
    ):
        list_onenote_notebooks("offline-token", http_client=client)
    assert len(requests) == 5
    assert sleeps == [2] * 4


def test_retry_recovers_and_token_is_not_in_error(monkeypatch):
    from cognee.tasks.ingestion.connectors import onenote

    monkeypatch.setattr(onenote.time, "sleep", lambda _: None)
    requests = []

    def response(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(200, json={"value": []})

    with httpx.Client(transport=httpx.MockTransport(response)) as client:
        assert list_onenote_notebooks("offline-token", http_client=client) == []
    assert len(requests) == 2
    with (
        httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(401))) as client,
        pytest.raises(OneNoteError, match="reauthenticate") as failure,
    ):
        list_onenote_notebooks("secret-token", http_client=client)
    assert "secret-token" not in str(failure.value)


def test_transport_failure_has_bounded_backoff_and_redacted_diagnostics(monkeypatch):
    from cognee.tasks.ingestion.connectors import onenote

    requests, sleeps = [], []
    monkeypatch.setattr(onenote.time, "sleep", sleeps.append)

    def timeout(request):
        requests.append(request)
        raise httpx.ReadTimeout("secret-token", request=request)

    with (
        httpx.Client(transport=httpx.MockTransport(timeout)) as client,
        pytest.raises(OneNoteError) as failure,
    ):
        list_onenote_notebooks("secret-token", http_client=client)
    assert len(requests) == 5
    assert sleeps == [1, 2, 4, 8]
    assert "secret-token" not in str(failure.value)


@pytest.mark.parametrize("retry_after", ["bad", "-1", "inf", "nan"])
def test_invalid_retry_after_uses_bounded_fallback(monkeypatch, retry_after):
    from cognee.tasks.ingestion.connectors import onenote

    sleeps = []
    monkeypatch.setattr(onenote.time, "sleep", sleeps.append)
    with (
        httpx.Client(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(429, headers={"Retry-After": retry_after})
            )
        ) as client,
        pytest.raises(OneNoteError),
    ):
        list_onenote_notebooks("offline-token", http_client=client)
    assert sleeps == [1, 2, 4, 8]


@pytest.mark.parametrize(
    "next_link",
    [
        "http://graph.microsoft.com/v1.0/me/onenote/notebooks",
        "https://evil.example/v1.0/me/onenote/notebooks",
        "https://graph.microsoft.com.evil.example/v1.0/me/onenote/notebooks",
        "https://user:password@graph.microsoft.com/v1.0/me/onenote/notebooks",
    ],
)
def test_unsafe_pagination_link_never_receives_token(next_link):
    requests = []

    def response(request):
        requests.append(request)
        return httpx.Response(200, json={"value": [{"id": "a"}], "@odata.nextLink": next_link})

    with (
        httpx.Client(transport=httpx.MockTransport(response)) as client,
        pytest.raises(OneNoteError, match="Unsafe"),
    ):
        list_onenote_notebooks("offline-token", http_client=client)
    assert len(requests) == 1


def test_injected_redirect_enabled_client_still_does_not_follow_redirects():
    requests = []

    def redirect(request):
        requests.append(request)
        return httpx.Response(302, headers={"Location": "https://evil.example"})

    with (
        httpx.Client(transport=httpx.MockTransport(redirect), follow_redirects=True) as client,
        pytest.raises(OneNoteError),
    ):
        list_onenote_notebooks("offline-token", http_client=client)
    assert len(requests) == 1


@pytest.mark.parametrize("duplicate", [False, True])
def test_next_link_cycle_and_duplicate_page_fail(duplicate):
    requests = []

    def response(request):
        requests.append(request)
        item_id = "a" if duplicate else str(len(requests))
        next_link = (
            "https://graph.microsoft.com/v1.0/me/onenote/notebooks?$skiptoken=next"
            if duplicate
            else str(request.url)
        )
        return httpx.Response(200, json={"value": [{"id": item_id}], "@odata.nextLink": next_link})

    with (
        httpx.Client(transport=httpx.MockTransport(response)) as client,
        pytest.raises(OneNoteError),
    ):
        list_onenote_notebooks("offline-token", http_client=client)
    assert len(requests) == (2 if duplicate else 1)


def test_successful_next_link_is_followed():
    requests = []

    def response(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(
                200,
                json={
                    "value": [{"id": "a"}],
                    "@odata.nextLink": "https://graph.microsoft.com/v1.0/me/onenote/"
                    "notebooks?$skiptoken=next",
                },
            )
        assert request.url.params["$skiptoken"] == "next"
        return httpx.Response(200, json={"value": [{"id": "b"}]})

    with httpx.Client(transport=httpx.MockTransport(response)) as client:
        assert [
            row["id"] for row in list_onenote_notebooks("offline-token", http_client=client)
        ] == [
            "a",
            "b",
        ]


def test_long_retry_after_stops_without_retrying_early(monkeypatch):
    from cognee.tasks.ingestion.connectors import onenote

    delays = []
    monkeypatch.setattr(onenote.time, "sleep", delays.append)
    client = httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(429, headers={"Retry-After": "301"}))
    )
    with client, pytest.raises(OneNoteError):
        list_onenote_notebooks("offline-token", http_client=client)
    assert delays == []


def test_cumulative_retry_wait_budget_never_retries_early(monkeypatch):
    from cognee.tasks.ingestion.connectors import onenote

    delays, requests = [], []
    monkeypatch.setattr(onenote.time, "sleep", delays.append)

    def response(request):
        requests.append(request)
        return httpx.Response(429, headers={"Retry-After": "40"})

    with (
        httpx.Client(transport=httpx.MockTransport(response)) as client,
        pytest.raises(OneNoteError),
    ):
        list_onenote_notebooks("offline-token", http_client=client)
    assert delays == [40]
    assert len(requests) == 2


def test_retry_wait_budget_constant_bounds_the_cumulative_sleep(monkeypatch):
    monkeypatch.setattr(onenote, "RETRY_WAIT_BUDGET_SECONDS", 5)
    delays, requests = [], []
    monkeypatch.setattr(onenote.time, "sleep", delays.append)

    def response(request):
        requests.append(request)
        return httpx.Response(429, headers={"Retry-After": "3"})

    with (
        httpx.Client(transport=httpx.MockTransport(response)) as client,
        pytest.raises(OneNoteError),
    ):
        list_onenote_notebooks("offline-token", http_client=client)
    assert delays == [3]
    assert len(requests) == 2
    assert onenote.RETRY_WAIT_BUDGET_SECONDS == 5
