"""Unit tests for FigmaClient: validation, authentication, rate limits, and budgets."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from cognee.tasks.ingestion.connectors.figma import (
    FigmaAPIError,
    FigmaAuthError,
    FigmaBudgetExceededError,
    FigmaClient,
    FigmaNotFoundError,
    FigmaRateLimitedError,
)


def test_token_validation_rejects_empty_and_whitespace():
    with pytest.raises(ValueError, match="empty or has an invalid format"):
        FigmaClient("")
    with pytest.raises(ValueError, match="empty or has an invalid format"):
        FigmaClient("   ")
    with pytest.raises(ValueError, match="empty or has an invalid format"):
        FigmaClient("token with spaces")
    with pytest.raises(ValueError, match="empty or has an invalid format"):
        FigmaClient("token\twith\nwhitespace")


def test_token_validation_rejects_non_ascii():
    with pytest.raises(ValueError, match="empty or has an invalid format"):
        FigmaClient("token_with_unicode_ñ")


def test_client_repr_redacts_token():
    client = FigmaClient("figd_super_secret_token_12345")
    assert repr(client) == "FigmaClient(access_token=<redacted>)"
    assert "super_secret" not in repr(client)


def test_header_selection_pat_vs_bearer():
    # Personal Access Token format
    pat_client = FigmaClient("figd_pat_token_123")
    assert pat_client._headers("figd_pat_token_123")["X-Figma-Token"] == "figd_pat_token_123"
    assert "Authorization" not in pat_client._headers("figd_pat_token_123")

    # OAuth / standard bearer format
    oauth_client = FigmaClient("oauth_access_token_456")
    assert (
        oauth_client._headers("oauth_access_token_456")["Authorization"]
        == "Bearer oauth_access_token_456"
    )
    assert "X-Figma-Token" not in oauth_client._headers("oauth_access_token_456")

    # Explicit auth_type override
    forced_pat = FigmaClient("custom_pat", auth_type="pat")
    assert forced_pat._headers("custom_pat")["X-Figma-Token"] == "custom_pat"


@patch("requests.request")
def test_client_request_success(mock_request):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"name": "My File"}
    mock_request.return_value = mock_resp

    client = FigmaClient("figd_test_token")
    result = client.request("GET", "/v1/files/abc123")
    assert result == {"name": "My File"}
    mock_request.assert_called_once()


@patch("requests.request")
def test_client_handles_401_403_as_auth_error(mock_request):
    mock_resp = MagicMock()
    mock_resp.status_code = 403
    mock_resp.json.return_value = {"err": "Invalid token"}
    mock_request.return_value = mock_resp

    client = FigmaClient("figd_test_token")
    with pytest.raises(FigmaAuthError, match="HTTP 401"):
        client.request("GET", "/v1/files/abc123")


@patch("requests.request")
def test_client_handles_404_as_not_found(mock_request):
    mock_resp = MagicMock()
    mock_resp.status_code = 404
    mock_resp.json.return_value = {"err": "File not found"}
    mock_request.return_value = mock_resp

    client = FigmaClient("figd_test_token")
    with pytest.raises(FigmaNotFoundError, match="HTTP 404"):
        client.request("GET", "/v1/files/missing")


@patch("requests.request")
def test_client_handles_429_as_rate_limited(mock_request):
    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_resp.json.return_value = {"err": "Rate limit exceeded"}
    mock_request.return_value = mock_resp

    client = FigmaClient("figd_test_token")
    with pytest.raises(FigmaRateLimitedError, match="HTTP 429"):
        client.request("GET", "/v1/files/abc123")


@patch("requests.request")
def test_client_retries_transient_500_errors(mock_request):
    error_resp = MagicMock()
    error_resp.status_code = 502
    success_resp = MagicMock()
    success_resp.status_code = 200
    success_resp.json.return_value = {"name": "Recovered"}
    mock_request.side_effect = [error_resp, success_resp]

    sleeps = []
    client = FigmaClient("figd_test_token", sleep=sleeps.append)
    result = client.request("GET", "/v1/files/abc123")

    assert result == {"name": "Recovered"}
    assert len(sleeps) == 1
    assert sleeps[0] == 1  # 2 ** (1 - 1) = 1


def test_client_budget_exceeded():
    client = FigmaClient("figd_test_token")
    stats = {"requests": 5}
    with pytest.raises(FigmaBudgetExceededError, match="budget exhausted"):
        client.request("GET", "/v1/files/abc123", stats=stats, max_requests=5)


def test_client_calls_check_active():
    active_checks = []
    client = FigmaClient("figd_test_token")
    with patch("requests.request") as mock_req:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {}
        mock_req.return_value = mock_resp

        client.request("GET", "/v1/files/abc123", check_active=lambda: active_checks.append(1))
        assert active_checks == [1]
