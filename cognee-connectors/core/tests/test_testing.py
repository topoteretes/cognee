"""Unit tests for cognee_connectors_core.testing — the helpers connector suites build on."""

import pytest

from cognee_connectors_core.descriptor import ConnectorDescriptor
from cognee_connectors_core.errors import RateLimitedError
from cognee_connectors_core.settings import require
from cognee_connectors_core.signatures import HmacVerifier, SignatureScheme
from cognee_connectors_core.testing import (
    FakeCredential,
    FakeHttpClient,
    FakeSettings,
    RecordedCall,
    sign_body,
    webhook_headers,
)

_URL = "https://testprov.example/oauth/token"


# ---- FakeHttpClient ------------------------------------------------------


@pytest.mark.asyncio
async def test_a_keyed_response_is_returned_for_its_method_and_url():
    http = FakeHttpClient({("POST", _URL): (200, {"ok": True})})

    assert await http.post_form(_URL, {"a": "b"}) == (200, {"ok": True})


@pytest.mark.asyncio
async def test_a_keyed_list_is_consumed_in_order_and_its_last_entry_repeats():
    http = FakeHttpClient({("GET", _URL): [(500, "down"), (200, {"ok": True})]})

    assert await http.get_json(_URL) == (500, "down")
    assert await http.get_json(_URL) == (200, {"ok": True})
    assert await http.get_json(_URL) == (200, {"ok": True})


@pytest.mark.asyncio
async def test_add_scripts_after_construction_and_ignores_method_case():
    http = FakeHttpClient().add("post", _URL, (201, {}))

    assert await http.post_form(_URL, {}) == (201, {})


@pytest.mark.asyncio
async def test_an_ordered_queue_answers_whatever_the_request():
    http = FakeHttpClient([(200, {"first": 1}), (404, {"second": 2})])

    assert await http.post_form(_URL, {}) == (200, {"first": 1})
    assert await http.get_json("https://elsewhere.example") == (404, {"second": 2})


@pytest.mark.asyncio
async def test_keyed_responses_win_over_the_queue():
    http = FakeHttpClient([(200, {"queued": True})]).add("GET", _URL, (200, {"keyed": True}))

    assert await http.get_json(_URL) == (200, {"keyed": True})
    assert await http.get_json("https://other.example") == (200, {"queued": True})


@pytest.mark.asyncio
async def test_the_default_answers_anything_unscripted():
    http = FakeHttpClient(default=(204, None))

    assert await http.get_json(_URL) == (204, None)


@pytest.mark.asyncio
async def test_an_unscripted_request_fails_the_test():
    with pytest.raises(AssertionError, match="no scripted response for GET"):
        await FakeHttpClient().get_json(_URL)


@pytest.mark.asyncio
async def test_an_exception_response_is_raised():
    http = FakeHttpClient([RateLimitedError(retry_after=2.5)])

    with pytest.raises(RateLimitedError) as excinfo:
        await http.get_json(_URL)

    assert excinfo.value.retry_after == 2.5


@pytest.mark.asyncio
async def test_calls_are_recorded_with_their_form_data_and_headers():
    http = FakeHttpClient(default=(200, {}))

    await http.post_form(_URL, {"code": "c"}, {"Accept": "application/json"})
    await http.get_json("https://api.example/me", {"Authorization": "Bearer t"})

    assert http.calls == [
        RecordedCall("POST", _URL, {"code": "c"}, {"Accept": "application/json"}),
        RecordedCall("GET", "https://api.example/me", None, {"Authorization": "Bearer t"}),
    ]
    assert http.last_call.url == "https://api.example/me"
    assert http.calls_to("post", _URL) == [http.calls[0]]


def test_last_call_before_any_call_is_an_assertion():
    with pytest.raises(AssertionError, match="received no calls"):
        _ = FakeHttpClient().last_call


# ---- FakeSettings / FakeCredential ---------------------------------------


def test_fake_settings_carries_the_shared_fields_and_extras():
    settings = FakeSettings(webhook_secret="s", signing_secret=None)

    assert settings.client_id == "client-id"
    assert settings.client_secret == "client-secret"
    assert settings.redirect_uri.endswith("/callback")
    assert settings.webhook_secret == "s"
    assert settings.signing_secret is None
    assert require(settings, "client_id") == "client-id"


def test_fake_settings_prefix_is_configurable():
    with pytest.raises(RuntimeError, match="NOTION_CLIENT_ID"):
        require(FakeSettings(client_id="", env_prefix="NOTION_"), "client_id")


def test_fake_credential_defaults():
    credential = FakeCredential()

    assert credential.provider == "testprov"
    assert credential.provider_account_id == "acct-1"
    assert credential.provider_metadata == {}
    assert credential.token_expires_at is None
    assert credential.auth_type == "oauth2"


# ---- sign_body / webhook_headers -----------------------------------------


def _descriptor(**overrides) -> ConnectorDescriptor:
    base = {
        "provider": "testprov",
        "display_name": "Test Provider",
        "authorize_url": "https://testprov.example/oauth/authorize",
        "token_url": _URL,
        "account_id_path": "account.id",
        "signature_scheme": SignatureScheme.SHA256_PREFIX,
        "signature_header": "X-Hub-Signature-256",
        "webhook_account_id_path": "account.id",
        "webhook_event_header": "X-GitHub-Event",
    }
    return ConnectorDescriptor(**{**base, **overrides})


@pytest.mark.parametrize(
    ("scheme", "timestamp_header"),
    [
        (SignatureScheme.SHA256_PREFIX, None),
        (SignatureScheme.BARE_HEX, None),
        (SignatureScheme.SLACK_V0, "x-ts"),
        (SignatureScheme.TIMESTAMPED, None),
    ],
    ids=["sha256_prefix", "bare_hex", "slack_v0", "timestamped"],
)
@pytest.mark.asyncio
async def test_sign_body_produces_what_the_verifier_accepts(scheme, timestamp_header):
    verifier = HmacVerifier(
        provider="testprov",
        scheme=scheme,
        header="x-sig",
        timestamp_header=timestamp_header,
        secret_provider=lambda: "secret",
    )
    body = b'{"id": 1}'
    timestamp = 1_700_000_000
    headers = {"x-sig": sign_body(scheme, "secret", body, timestamp=timestamp)}
    if timestamp_header:
        headers[timestamp_header] = str(timestamp)

    import time

    # The timestamped schemes enforce a replay window; sign with the present.
    if scheme in (SignatureScheme.SLACK_V0, SignatureScheme.TIMESTAMPED):
        timestamp = int(time.time())
        headers = {"x-sig": sign_body(scheme, "secret", body, timestamp=timestamp)}
        if timestamp_header:
            headers[timestamp_header] = str(timestamp)

    assert await verifier.verify(body, headers) == body


def test_sign_body_needs_a_timestamp_for_the_timestamped_schemes():
    with pytest.raises(ValueError, match="needs a timestamp"):
        sign_body(SignatureScheme.SLACK_V0, "s", b"x")


@pytest.mark.asyncio
async def test_webhook_headers_fill_everything_the_descriptor_verifier_reads():
    descriptor = _descriptor()
    body = b'{"account": {"id": "A1"}}'

    headers = webhook_headers(descriptor, "secret", body, event="push")

    assert set(headers) == {"X-Hub-Signature-256", "X-GitHub-Event"}
    assert headers["X-GitHub-Event"] == "push"
    verifier = HmacVerifier(
        provider="testprov",
        scheme=SignatureScheme.SHA256_PREFIX,
        header="x-hub-signature-256",
        secret_provider=lambda: "secret",
    )
    assert await verifier.verify(body, headers) == body


@pytest.mark.asyncio
async def test_webhook_headers_include_the_timestamp_for_a_slack_descriptor():
    descriptor = _descriptor(
        signature_scheme=SignatureScheme.SLACK_V0,
        signature_header="X-Slack-Signature",
        signature_timestamp_header="X-Slack-Request-Timestamp",
    )
    body = b"payload"

    headers = webhook_headers(descriptor, "secret", body)

    assert set(headers) == {"X-Slack-Signature", "X-Slack-Request-Timestamp"}
    verifier = HmacVerifier(
        provider="testprov",
        scheme=SignatureScheme.SLACK_V0,
        header="x-slack-signature",
        timestamp_header="x-slack-request-timestamp",
        secret_provider=lambda: "secret",
    )
    assert await verifier.verify(body, headers) == body


def test_webhook_headers_refuse_a_descriptor_without_webhooks():
    descriptor = _descriptor(
        signature_scheme=None,
        signature_header=None,
        webhook_account_id_path=None,
        webhook_event_header=None,
        poll_interval_seconds=60,
    )

    with pytest.raises(ValueError, match="declares no webhook signature"):
        webhook_headers(descriptor, "secret", b"x")
