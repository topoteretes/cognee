"""Unit tests for the Twilio data-source connector."""

from unittest.mock import MagicMock

import pytest

from cognee.tasks.ingestion.connectors.twilio import (
    TwilioAPIError,
    TwilioAuthError,
    TwilioClient,
    TwilioRateLimitError,
    clean_phone_for_node,
    iter_twilio_rows,
    mask_phone,
    render_call_document,
    render_message_document,
    render_transcription_document,
    twilio_source,
)
from cognee.tasks.ingestion.dlt_utils import DOCUMENT_SOURCE_ATTR


def test_client_validation():
    """Client should validate Account SID and Auth Token format."""
    with pytest.raises(ValueError, match="Invalid Twilio Account SID"):
        TwilioClient(account_sid="XYZ123", auth_token="token")

    with pytest.raises(ValueError, match="Invalid Twilio Account SID"):
        TwilioClient(account_sid="", auth_token="token")

    with pytest.raises(ValueError, match="Twilio Auth Token is required"):
        TwilioClient(account_sid="AC_MOCK_ACCOUNT_SID_001", auth_token="")


def test_client_repr_redaction():
    """Client __repr__ should never leak the Auth Token."""
    client = TwilioClient(
        account_sid="AC_MOCK_ACCOUNT_SID_001", auth_token="super_secret_token_value"
    )
    repr_str = repr(client)
    assert "super_secret_token_value" not in repr_str
    assert "AC_MOC..." in repr_str


def test_mask_phone():
    """Test masking phone numbers for PII compliance."""
    assert mask_phone("+15551234567") == "+1******4567"
    assert mask_phone("+442071838750") == "+4*******8750"
    assert mask_phone("12345") == "12345"
    assert mask_phone("") == "Unknown"
    assert mask_phone(None) == "Unknown"


def test_clean_phone_for_node():
    """Test phone sanitization for node set tags."""
    assert clean_phone_for_node("+15551234567") == "+15551234567"
    assert clean_phone_for_node(" +1 (555) 123-4567 ") == "+15551234567"
    assert clean_phone_for_node("") == "unknown"
    assert clean_phone_for_node(None) == "unknown"


def test_render_message_document():
    """Test rendering of an SMS message record."""
    msg = {
        "sid": "SM1234567890abcdef",
        "from": "+15551112222",
        "to": "+15553334444",
        "direction": "inbound",
        "status": "received",
        "date_sent": "2026-10-20T12:00:00Z",
        "body": "Hello, I want to reschedule my appointment to Thursday at 3 PM.",
        "num_segments": 1,
        "price": "-0.0075",
        "price_unit": "USD",
    }

    doc = render_message_document(msg, mask_numbers=False)
    assert doc["id"] == "twilio:msg:SM1234567890abcdef"
    assert "SM1234567890abcdef" in doc["title"]
    assert "+15551112222" in doc["title"]
    assert "reschedule my appointment" in doc["content"]
    assert doc["_deleted"] is False
    assert "twilio" in doc["cognee_node_set"]
    assert "twilio:sms" in doc["cognee_node_set"]
    assert "phone:+15551112222" in doc["cognee_node_set"]
    assert "phone:+15553334444" in doc["cognee_node_set"]

    # Masked variant
    doc_masked = render_message_document(msg, mask_numbers=True)
    assert "+1******2222" in doc_masked["title"]
    assert "+1******2222" in doc_masked["content"]


def test_render_call_document():
    """Test rendering of a Voice Call record."""
    call = {
        "sid": "CA1234567890abcdef",
        "from": "+15551112222",
        "to": "+15553334444",
        "direction": "outbound-api",
        "status": "completed",
        "duration": "142",
        "start_time": "2026-10-20T12:05:00Z",
        "end_time": "2026-10-20T12:07:22Z",
        "answered_by": "human",
    }

    doc = render_call_document(call, mask_numbers=False)
    assert doc["id"] == "twilio:call:CA1234567890abcdef"
    assert "142s" in doc["title"]
    assert "completed" in doc["title"]
    assert "Duration**: 142 seconds" in doc["content"]
    assert doc["_deleted"] is False
    assert "twilio:call" in doc["cognee_node_set"]
    assert "phone:+15551112222" in doc["cognee_node_set"]


def test_render_transcription_document():
    """Test rendering of an audio transcription record."""
    transcript = {
        "sid": "TR1234567890abcdef",
        "recording_sid": "RE9876543210fedcba",
        "status": "completed",
        "duration": "45",
        "date_created": "2026-10-20T12:10:00Z",
        "transcription_text": "Hi, this is Alice calling about contract pricing details.",
    }

    doc = render_transcription_document(transcript)
    assert doc["id"] == "twilio:transcript:TR1234567890abcdef"
    assert "RE9876543210fedcba" in doc["title"]
    assert "contract pricing details" in doc["content"]
    assert "twilio:transcript" in doc["cognee_node_set"]
    assert "recording:RE9876543210fedcba" in doc["cognee_node_set"]


def test_client_auth_failure(monkeypatch):
    """Client should raise TwilioAuthError on HTTP 401."""
    import requests

    client = TwilioClient(
        account_sid="AC_MOCK_ACCOUNT_SID_001",
        auth_token="bad_token",
        sleep=lambda _: None,
    )

    mock_resp = MagicMock()
    mock_resp.status_code = 401
    monkeypatch.setattr(requests, "get", lambda *a, **k: mock_resp)

    with pytest.raises(TwilioAuthError, match="credentials rejected"):
        client.get("/test")


def test_client_rate_limit(monkeypatch):
    """Client should raise TwilioRateLimitError on repeated HTTP 429."""
    import requests

    client = TwilioClient(
        account_sid="AC_MOCK_ACCOUNT_SID_001",
        auth_token="token",
        sleep=lambda _: None,
    )

    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_resp.headers = {"Retry-After": "3"}
    monkeypatch.setattr(requests, "get", lambda *a, **k: mock_resp)

    with pytest.raises(TwilioRateLimitError, match="rate limit exceeded"):
        client.get("/test")


def test_client_server_error_retry(monkeypatch):
    """Client should retry on 500 server errors and succeed if later attempt passes."""
    import requests

    client = TwilioClient(
        account_sid="AC_MOCK_ACCOUNT_SID_001",
        auth_token="token",
        sleep=lambda _: None,
    )

    attempts = 0

    def mock_get(*a, **k):
        nonlocal attempts
        attempts += 1
        resp = MagicMock()
        if attempts == 1:
            resp.status_code = 503
            return resp
        resp.status_code = 200
        resp.json.return_value = {"messages": []}
        return resp

    monkeypatch.setattr(requests, "get", mock_get)
    result = client.get("/test")
    assert result == {"messages": []}
    assert attempts == 2


def test_iter_twilio_rows_all_streams():
    """Test full iteration over messages, calls, and transcriptions with fake client."""
    fake_client = MagicMock()
    fake_client.account_sid = "AC_MOCK_ACCOUNT_SID_001"

    def fake_get(url, params=None):
        if "Messages.json" in url:
            return {
                "messages": [
                    {
                        "sid": "SM001",
                        "from": "+15551111111",
                        "to": "+15552222222",
                        "body": "Test SMS message 1",
                        "date_sent": "2026-10-20T10:00:00Z",
                        "status": "delivered",
                    }
                ],
                "next_page_uri": None,
            }
        elif "Calls.json" in url:
            return {
                "calls": [
                    {
                        "sid": "CA001",
                        "from": "+15551111111",
                        "to": "+15552222222",
                        "status": "completed",
                        "duration": "60",
                        "start_time": "2026-10-20T10:05:00Z",
                    }
                ],
                "next_page_uri": None,
            }
        elif "Transcriptions.json" in url:
            return {
                "transcriptions": [
                    {
                        "sid": "TR001",
                        "recording_sid": "RE001",
                        "status": "completed",
                        "duration": "30",
                        "transcription_text": "Customer left voice message.",
                    }
                ],
                "next_page_uri": None,
            }
        return {}

    fake_client.get.side_effect = fake_get

    state = {}
    stats = {}
    rows = list(
        iter_twilio_rows(
            client=fake_client,
            state=state,
            stats=stats,
            include_messages=True,
            include_calls=True,
            include_transcriptions=True,
        )
    )

    assert len(rows) == 3
    assert rows[0]["id"] == "twilio:msg:SM001"
    assert rows[1]["id"] == "twilio:call:CA001"
    assert rows[2]["id"] == "twilio:transcript:TR001"
    assert stats["scanned"] == 3
    assert stats["yielded"] == 3
    assert state["last_message_date"] == "2026-10-20T10:00:00Z"
    assert state["last_call_date"] == "2026-10-20T10:05:00Z"


def test_iter_twilio_rows_pagination():
    """Test pagination across multiple pages following next_page_uri."""
    fake_client = MagicMock()
    fake_client.account_sid = "AC_MOCK_ACCOUNT_SID_001"

    call_count = 0

    def fake_get(url, params=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {
                "messages": [
                    {
                        "sid": "SM_P1",
                        "from": "+15551111111",
                        "to": "+15552222222",
                        "body": "Page 1 message",
                        "date_sent": "2026-10-20T10:00:00Z",
                    }
                ],
                "next_page_uri": "/2010-04-01/Accounts/AC.../Messages.json?Page=1&PageToken=abc",
            }
        elif call_count == 2:
            return {
                "messages": [
                    {
                        "sid": "SM_P2",
                        "from": "+15551111111",
                        "to": "+15552222222",
                        "body": "Page 2 message",
                        "date_sent": "2026-10-20T10:01:00Z",
                    }
                ],
                "next_page_uri": None,
            }
        return {"calls": [], "transcriptions": []}

    fake_client.get.side_effect = fake_get

    state = {}
    stats = {}
    rows = list(
        iter_twilio_rows(
            client=fake_client,
            state=state,
            stats=stats,
            include_messages=True,
            include_calls=False,
            include_transcriptions=False,
        )
    )

    assert len(rows) == 2
    assert rows[0]["id"] == "twilio:msg:SM_P1"
    assert rows[1]["id"] == "twilio:msg:SM_P2"
    assert state["last_message_date"] == "2026-10-20T10:01:00Z"


def test_iter_twilio_rows_incremental_filter():
    """Test that existing floor is passed in query parameters on subsequent runs."""
    fake_client = MagicMock()
    fake_client.account_sid = "AC_MOCK_ACCOUNT_SID_001"
    captured_params = {}

    def fake_get(url, params=None):
        nonlocal captured_params
        if params:
            captured_params = dict(params)
        return {"messages": [], "calls": [], "transcriptions": []}

    fake_client.get.side_effect = fake_get

    state = {"last_message_date": "2026-10-20T08:00:00Z"}
    stats = {}
    list(
        iter_twilio_rows(
            client=fake_client,
            state=state,
            stats=stats,
            include_messages=True,
            include_calls=False,
            include_transcriptions=False,
        )
    )

    assert captured_params.get("DateSent>") == "2026-10-20T08:00:00Z"


def test_twilio_source_factory():
    """Test twilio_source returns a valid dlt resource with DOCUMENT_SOURCE_ATTR."""
    fake_client = MagicMock()
    fake_client.account_sid = "AC_MOCK_ACCOUNT_SID_001"

    res = twilio_source(
        account_sid="AC_MOCK_ACCOUNT_SID_001",
        auth_token="dummy_token",
        client=fake_client,
    )

    assert res.name == "twilio_ount_sid_001"
    assert res.write_disposition == "merge"
    assert getattr(res, DOCUMENT_SOURCE_ATTR, None) == "twilio"
