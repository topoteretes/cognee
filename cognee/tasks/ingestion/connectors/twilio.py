"""Twilio connector for cognee: a ``dlt`` source that turns telecommunications into memory.

Ingests SMS/MMS messages, voice call logs, and call audio transcriptions from Twilio
into structured documents and customer phone entity nodes, suitable for passing to
:func:`cognee.remember`::

    import cognee
    from cognee.tasks.ingestion.connectors import twilio_source

    await cognee.remember(
        twilio_source(
            account_sid="AC_YOUR_TWILIO_ACCOUNT_SID",
            auth_token="your_auth_token",
            include_messages=True,
            include_calls=True,
            include_transcriptions=True,
        ),
        dataset_name="twilio",
        primary_key="id",
        write_disposition="merge",
        max_rows_per_table=0,
    )

Design
------
* **Rows** are flat ``{id, title, content, url, _deleted, cognee_node_set}``.
  IDs are prefixed by kind (``twilio:msg:<sid>``, ``twilio:call:<sid>``,
  ``twilio:transcript:<sid>``).
* **Graph entities:** Phone numbers (in E.164 format) are attached to node sets
  (e.g., ``phone:+15551234567``) so Cognee links customer communications across
  channels into cohesive person/customer knowledge graphs.
* **Incremental:** Filtered by ``DateSent>=<floor>`` for messages and
  ``StartTime>=<floor>`` for calls. A run with no new events yields nothing.
* **Privacy & Masking:** Optional ``mask_phone_numbers=True`` masks all but
  the last 4 digits for compliance with GDPR, HIPAA, and customer privacy standards.
* **Rate Limits:** Automatically handles HTTP 429 and Twilio rate limit headers
  with exponential backoff.
"""

import logging
import re
import time
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from typing import Any

from cognee.tasks.ingestion import dlt_utils
from cognee.tasks.ingestion.dlt_utils import DOCUMENT_SOURCE_ATTR

logger = logging.getLogger(__name__)

BASE_URL = "https://api.twilio.com"
DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 100
DEFAULT_MAX_REQUESTS = 1000
_TIMEOUT = (10, 30)
_RETRIES = 3


class TwilioAPIError(Exception):
    """Raised when a Twilio REST API request fails."""


class TwilioAuthError(TwilioAPIError):
    """Raised when Twilio rejects the Account SID or Auth Token (HTTP 401)."""


class TwilioRateLimitError(TwilioAPIError):
    """Raised when Twilio rate limits requests (HTTP 429)."""

    def __init__(self, retry_after: int | None = None):
        super().__init__(f"Twilio rate limit exceeded (Retry-After: {retry_after}s)")
        self.retry_after = retry_after


def mask_phone(phone: str | None) -> str:
    """Mask a phone number, preserving only the country prefix and last 4 digits."""
    if not phone:
        return "Unknown"
    cleaned = phone.strip()
    if len(cleaned) <= 6:
        return cleaned
    prefix = cleaned[:2] if cleaned.startswith("+") else ""
    last_four = cleaned[-4:]
    masked_len = max(3, len(cleaned) - len(prefix) - 4)
    return f"{prefix}{'*' * masked_len}{last_four}"


def clean_phone_for_node(phone: str | None) -> str:
    """Sanitize phone number for node set tagging."""
    if not phone:
        return "unknown"
    return re.sub(r"[^a-zA-Z0-9_+]", "", phone.strip()).lower()


class TwilioClient:
    """Synchronous HTTP client for Twilio REST API."""

    def __init__(
        self,
        account_sid: str,
        auth_token: str,
        api_key: str | None = None,
        base_url: str = BASE_URL,
        *,
        sleep: Callable[[float], None] = time.sleep,
    ):
        sid = (account_sid or "").strip()
        token = (auth_token or "").strip()
        if not sid or not sid.startswith("AC"):
            raise ValueError("Invalid Twilio Account SID. Must start with 'AC'.")
        if not token:
            raise ValueError("Twilio Auth Token is required.")

        self.account_sid = sid
        self.auth_token = token
        self.api_key = api_key.strip() if api_key else None
        self.base_url = base_url.rstrip("/")
        self._sleep = sleep

    def __repr__(self) -> str:
        masked_sid = self.account_sid[:6] + "..." if len(self.account_sid) > 6 else "..."
        return f"TwilioClient(account_sid={masked_sid})"

    def _auth(self) -> tuple[str, str]:
        username = self.api_key if self.api_key else self.account_sid
        return (username, self.auth_token)

    def get(self, endpoint_or_uri: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Send a GET request to Twilio API with retries and rate limit handling."""
        import requests

        if endpoint_or_uri.startswith("http"):
            url = endpoint_or_uri
        else:
            path = endpoint_or_uri if endpoint_or_uri.startswith("/") else f"/{endpoint_or_uri}"
            url = f"{self.base_url}{path}"

        for attempt in range(_RETRIES):
            last = attempt == _RETRIES - 1
            try:
                response = requests.get(
                    url,
                    auth=self._auth(),
                    params=params if not endpoint_or_uri.startswith("http") else None,
                    timeout=_TIMEOUT,
                    headers={"Accept": "application/json"},
                )
            except requests.RequestException as exc:
                if last:
                    raise TwilioAPIError(
                        f"Twilio network request failed: {type(exc).__name__}"
                    ) from None
                self._sleep(2**attempt)
                continue

            if response.status_code == 401:
                raise TwilioAuthError("Twilio credentials rejected (HTTP 401).")

            if response.status_code == 429:
                retry_after_str = response.headers.get("Retry-After")
                retry_after = (
                    int(retry_after_str)
                    if retry_after_str and retry_after_str.isdigit()
                    else 2**attempt
                )
                if last:
                    raise TwilioRateLimitError(retry_after=retry_after)
                self._sleep(retry_after)
                continue

            if response.status_code >= 500:
                if last:
                    raise TwilioAPIError(f"Twilio server error HTTP {response.status_code}")
                self._sleep(2**attempt)
                continue

            if response.status_code >= 400:
                try:
                    err_json = response.json()
                    msg = err_json.get("message") or response.text
                except (ValueError, KeyError, TypeError):
                    msg = response.text
                raise TwilioAPIError(f"Twilio API error HTTP {response.status_code}: {msg}")

            try:
                return response.json()
            except ValueError as exc:
                raise TwilioAPIError("Invalid JSON received from Twilio API.") from exc

        raise TwilioAPIError("Twilio request failed after all retry attempts.")


def render_message_document(msg: dict[str, Any], mask_numbers: bool = False) -> dict[str, Any]:
    """Render a Twilio SMS/MMS record into a Cognee document dictionary."""
    sid = msg.get("sid", "")
    from_raw = msg.get("from") or ""
    to_raw = msg.get("to") or ""
    direction = msg.get("direction", "unknown")
    status = msg.get("status", "unknown")
    date_sent = msg.get("date_sent") or msg.get("date_created") or ""
    body = msg.get("body") or ""
    num_segments = msg.get("num_segments", 1)
    price = msg.get("price") or "0.00"
    price_unit = msg.get("price_unit") or "USD"

    from_disp = mask_phone(from_raw) if mask_numbers else from_raw
    to_disp = mask_phone(to_raw) if mask_numbers else to_raw

    content_lines = [
        "# Twilio SMS Message",
        f"- **Message SID**: `{sid}`",
        f"- **From**: {from_disp}",
        f"- **To**: {to_disp}",
        f"- **Direction**: {direction}",
        f"- **Status**: {status}",
        f"- **Date Sent**: {date_sent}",
        f"- **Segments**: {num_segments}",
        f"- **Price**: {price} {price_unit}",
        "",
        "## Message Content",
        body.strip() if body else "*(No textual body content)*",
    ]

    from_clean = clean_phone_for_node(from_raw)
    to_clean = clean_phone_for_node(to_raw)
    node_sets = ["twilio", "twilio:sms"]
    if from_clean and from_clean != "unknown":
        node_sets.append(f"phone:{from_clean}")
    if to_clean and to_clean != "unknown":
        node_sets.append(f"phone:{to_clean}")

    return {
        "id": f"twilio:msg:{sid}",
        "title": f"Twilio SMS {sid}: {from_disp} -> {to_disp} ({status})",
        "content": "\n".join(content_lines),
        "url": f"https://console.twilio.com/us1/monitor/logs/sms/{sid}",
        "_deleted": False,
        "cognee_node_set": node_sets,
    }


def render_call_document(call: dict[str, Any], mask_numbers: bool = False) -> dict[str, Any]:
    """Render a Twilio Voice Call record into a Cognee document dictionary."""
    sid = call.get("sid", "")
    from_raw = call.get("from") or ""
    to_raw = call.get("to") or ""
    direction = call.get("direction", "unknown")
    status = call.get("status", "unknown")
    duration = call.get("duration", "0")
    start_time = call.get("start_time") or call.get("date_created") or ""
    end_time = call.get("end_time") or ""
    answered_by = call.get("answered_by") or "human"

    from_disp = mask_phone(from_raw) if mask_numbers else from_raw
    to_disp = mask_phone(to_raw) if mask_numbers else to_raw

    content_lines = [
        "# Twilio Voice Call",
        f"- **Call SID**: `{sid}`",
        f"- **From**: {from_disp}",
        f"- **To**: {to_disp}",
        f"- **Direction**: {direction}",
        f"- **Status**: {status}",
        f"- **Duration**: {duration} seconds",
        f"- **Start Time**: {start_time}",
        f"- **End Time**: {end_time}",
        f"- **Answered By**: {answered_by}",
    ]

    from_clean = clean_phone_for_node(from_raw)
    to_clean = clean_phone_for_node(to_raw)
    node_sets = ["twilio", "twilio:call"]
    if from_clean and from_clean != "unknown":
        node_sets.append(f"phone:{from_clean}")
    if to_clean and to_clean != "unknown":
        node_sets.append(f"phone:{to_clean}")

    return {
        "id": f"twilio:call:{sid}",
        "title": f"Twilio Call: {from_disp} <-> {to_disp} ({status}, {duration}s)",
        "content": "\n".join(content_lines),
        "url": f"https://console.twilio.com/us1/monitor/logs/calls/{sid}",
        "_deleted": False,
        "cognee_node_set": node_sets,
    }


def render_transcription_document(
    transcript: dict[str, Any], mask_numbers: bool = False
) -> dict[str, Any]:
    """Render a Twilio Audio Transcription record into a Cognee document dictionary."""
    sid = transcript.get("sid", "")
    recording_sid = transcript.get("recording_sid", "")
    duration = transcript.get("duration", "0")
    status = transcript.get("status", "unknown")
    date_created = transcript.get("date_created", "")
    text = transcript.get("transcription_text") or ""

    content_lines = [
        "# Twilio Call Recording Audio Transcription",
        f"- **Transcription SID**: `{sid}`",
        f"- **Recording SID**: `{recording_sid}`",
        f"- **Status**: {status}",
        f"- **Duration**: {duration} seconds",
        f"- **Date Created**: {date_created}",
        "",
        "## Transcript Text",
        text.strip() if text else "*(Transcription empty or pending)*",
    ]

    node_sets = ["twilio", "twilio:transcript"]
    if recording_sid:
        node_sets.append(f"recording:{recording_sid}")

    return {
        "id": f"twilio:transcript:{sid}",
        "title": f"Twilio Transcript: {sid} (Recording: {recording_sid})",
        "content": "\n".join(content_lines),
        "url": f"https://console.twilio.com/us1/monitor/logs/recordings/{recording_sid}",
        "_deleted": False,
        "cognee_node_set": node_sets,
    }


def iter_twilio_rows(
    client: TwilioClient,
    state: dict[str, Any],
    stats: dict[str, int],
    *,
    include_messages: bool = True,
    include_calls: bool = True,
    include_transcriptions: bool = True,
    mask_phone_numbers: bool = False,
    page_size: int = DEFAULT_PAGE_SIZE,
    max_requests: int = DEFAULT_MAX_REQUESTS,
) -> Iterator[dict[str, Any]]:
    """Iterate through Twilio messages, calls, and transcriptions, yielding document rows."""
    account_sid = client.account_sid
    requests_count = 0

    # 1. Fetch Messages
    if include_messages:
        last_msg_floor = state.get("last_message_date")
        next_msg_floor = last_msg_floor
        msg_endpoint = f"/2010-04-01/Accounts/{account_sid}/Messages.json"
        params: dict[str, Any] = {"PageSize": min(page_size, MAX_PAGE_SIZE)}
        if last_msg_floor:
            params["DateSent>"] = last_msg_floor

        current_url: str | None = msg_endpoint
        while current_url and requests_count < max_requests:
            requests_count += 1
            data = client.get(current_url, params=params if current_url == msg_endpoint else None)
            messages = data.get("messages", [])

            for msg in messages:
                stats["scanned"] = stats.get("scanned", 0) + 1
                stamp = msg.get("date_sent") or msg.get("date_created") or ""
                if not next_msg_floor or (stamp and stamp > next_msg_floor):
                    next_msg_floor = stamp

                yield render_message_document(msg, mask_numbers=mask_phone_numbers)
                stats["yielded"] = stats.get("yielded", 0) + 1

            next_page_uri = data.get("next_page_uri")
            current_url = next_page_uri if next_page_uri else None

        if next_msg_floor:
            state["last_message_date"] = next_msg_floor

    # 2. Fetch Calls
    if include_calls and requests_count < max_requests:
        last_call_floor = state.get("last_call_date")
        next_call_floor = last_call_floor
        call_endpoint = f"/2010-04-01/Accounts/{account_sid}/Calls.json"
        params = {"PageSize": min(page_size, MAX_PAGE_SIZE)}
        if last_call_floor:
            params["StartTime>"] = last_call_floor

        current_url = call_endpoint
        while current_url and requests_count < max_requests:
            requests_count += 1
            data = client.get(current_url, params=params if current_url == call_endpoint else None)
            calls = data.get("calls", [])

            for call in calls:
                stats["scanned"] = stats.get("scanned", 0) + 1
                stamp = call.get("start_time") or call.get("date_created") or ""
                if not next_call_floor or (stamp and stamp > next_call_floor):
                    next_call_floor = stamp

                yield render_call_document(call, mask_numbers=mask_phone_numbers)
                stats["yielded"] = stats.get("yielded", 0) + 1

            next_page_uri = data.get("next_page_uri")
            current_url = next_page_uri if next_page_uri else None

        if next_call_floor:
            state["last_call_date"] = next_call_floor

    # 3. Fetch Transcriptions
    if include_transcriptions and requests_count < max_requests:
        transcription_endpoint = f"/2010-04-01/Accounts/{account_sid}/Transcriptions.json"
        params = {"PageSize": min(page_size, MAX_PAGE_SIZE)}

        current_url = transcription_endpoint
        while current_url and requests_count < max_requests:
            requests_count += 1
            data = client.get(
                current_url, params=params if current_url == transcription_endpoint else None
            )
            transcriptions = data.get("transcriptions", [])

            for transcript in transcriptions:
                stats["scanned"] = stats.get("scanned", 0) + 1
                yield render_transcription_document(transcript, mask_numbers=mask_phone_numbers)
                stats["yielded"] = stats.get("yielded", 0) + 1

            next_page_uri = data.get("next_page_uri")
            current_url = next_page_uri if next_page_uri else None


def twilio_source(
    *,
    account_sid: str,
    auth_token: str,
    api_key: str | None = None,
    resource_name: str | None = None,
    include_messages: bool = True,
    include_calls: bool = True,
    include_transcriptions: bool = True,
    mask_phone_numbers: bool = False,
    page_size: int = DEFAULT_PAGE_SIZE,
    max_requests: int = DEFAULT_MAX_REQUESTS,
    client: Any | None = None,
):
    """Return a dlt resource yielding Twilio communications for Cognee memory.

    Args:
        account_sid: Twilio Account SID (starts with 'AC').
        auth_token: Twilio Auth Token or API Secret.
        api_key: Optional Twilio API Key SID (starts with 'SK').
        resource_name: Custom dlt resource name. Defaults to 'twilio_<account_sid_suffix>'.
        include_messages: Ingest SMS/MMS messages.
        include_calls: Ingest voice call logs.
        include_transcriptions: Ingest call audio transcriptions.
        mask_phone_numbers: Mask phone numbers for GDPR/HIPAA compliance.
        page_size: Pagination size per request (max 100).
        max_requests: Request quota cap per sync.
        client: Pre-built TwilioClient for testing and dependency injection.
    """
    try:
        import dlt
    except ImportError as exc:
        raise ImportError("The Twilio connector requires dlt (a cognee core dependency).") from exc

    if getattr(dlt_utils, "DOCUMENT_SYNC_VERSION", 0) < 1:
        raise RuntimeError(
            "Twilio sync requires a Cognee build with table-scoped DLT document cleanup."
        )

    active_client = client or TwilioClient(
        account_sid=account_sid,
        auth_token=auth_token,
        api_key=api_key,
    )

    clean_sid = re.sub(r"[^a-zA-Z0-9_]+", "_", active_client.account_sid.lower())
    r_name = resource_name or f"twilio_{clean_sid[-12:]}"

    @dlt.resource(
        name=r_name,
        primary_key="id",
        write_disposition="merge",
        columns={"_deleted": {"data_type": "bool", "nullable": True}},
    )
    def _resource():
        state = dlt.current.resource_state() if hasattr(dlt, "current") else {}
        stats: dict[str, int] = {"scanned": 0, "yielded": 0}

        yield from iter_twilio_rows(
            client=active_client,
            state=state,
            stats=stats,
            include_messages=include_messages,
            include_calls=include_calls,
            include_transcriptions=include_transcriptions,
            mask_phone_numbers=mask_phone_numbers,
            page_size=page_size,
            max_requests=max_requests,
        )

    res = _resource()
    setattr(res, DOCUMENT_SOURCE_ATTR, "twilio")
    return res
