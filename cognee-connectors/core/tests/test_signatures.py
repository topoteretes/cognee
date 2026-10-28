"""Unit tests for cognee_connectors_core.signatures.

Ported from cloud-backend tests/modules/integrations/test_framework_signatures.py
(cognee-saas-backend, 2026-10-26); the FastAPI request stand-in became the
``(body, headers)`` pair the host-agnostic verifier takes, and ``HTTPException``
became ``WebhookVerificationError`` carrying the same status codes. The
stream-level cases (a chunked body, a client disconnecting mid-upload) are a
host's to test: the host reads the body, this verifier only measures it.

This is the contract's highest-stakes surface: a mistake here is a silent
authentication bypass on every connector at once, not a visible bug in one. So
each scheme is tested against the shapes that must pass and — more importantly
— the shapes that must not: a tampered body, a wrong secret, a missing header,
a replayed timestamp, and an unconfigured secret.

The unconfigured case is the one worth being loudest about. An HMAC keyed with
an empty string is a perfectly valid HMAC, so a verifier that does not
special-case a missing secret accepts anything an attacker signs the same way
— a config gap silently becoming an open endpoint.
"""

import hashlib
import hmac
import time

import pytest

from cognee_connectors_core.errors import ConnectorError, WebhookVerificationError
from cognee_connectors_core.signatures import (
    MAX_BODY_BYTES,
    HmacVerifier,
    SignatureScheme,
)

_SECRET = "77e5c0f7ab0d4d8f9c1b2a3e4f5d6c7b"


def _hex(secret: str, message: bytes) -> str:
    return hmac.new(secret.encode(), message, hashlib.sha256).hexdigest()


def _verifier(scheme: SignatureScheme, *, secret: str | None = _SECRET, **kwargs) -> HmacVerifier:
    return HmacVerifier(
        provider="testprov",
        scheme=scheme,
        secret_provider=lambda: secret,
        **kwargs,
    )


# ---- sha256_prefix (GitHub, Jira) --------------------------------------


def _sha256_prefix_verifier(**kwargs) -> HmacVerifier:
    return _verifier(SignatureScheme.SHA256_PREFIX, header="x-hub-signature-256", **kwargs)


@pytest.mark.asyncio
async def test_sha256_prefix_valid_signature_returns_raw_body():
    body = b'{"action":"push","installation":{"id":42}}'
    headers = {"X-Hub-Signature-256": f"sha256={_hex(_SECRET, body)}"}

    # Returning the raw bytes is the contract, not an implementation detail:
    # handlers must parse these rather than a re-serialized body, or the HMAC
    # they were verified against no longer describes what gets parsed.
    assert await _sha256_prefix_verifier().verify(body, headers) == body


@pytest.mark.asyncio
async def test_sha256_prefix_tampered_body_rejected():
    signed_body = b'{"repo":"safe"}'
    delivered_body = b'{"repo":"evil"}'
    headers = {"X-Hub-Signature-256": f"sha256={_hex(_SECRET, signed_body)}"}

    with pytest.raises(WebhookVerificationError) as excinfo:
        await _sha256_prefix_verifier().verify(delivered_body, headers)
    assert excinfo.value.status_code == 401


@pytest.mark.asyncio
async def test_sha256_prefix_wrong_secret_rejected():
    body = b"payload"
    headers = {"X-Hub-Signature-256": f"sha256={_hex('other', body)}"}

    with pytest.raises(WebhookVerificationError):
        await _sha256_prefix_verifier().verify(body, headers)


@pytest.mark.asyncio
async def test_sha256_prefix_missing_prefix_rejected():
    """A bare hex digest must not pass a scheme that specifies the prefix."""
    body = b"payload"

    with pytest.raises(WebhookVerificationError):
        await _sha256_prefix_verifier().verify(body, {"X-Hub-Signature-256": _hex(_SECRET, body)})


@pytest.mark.asyncio
async def test_missing_signature_header_rejected():
    with pytest.raises(WebhookVerificationError):
        await _sha256_prefix_verifier().verify(b"payload", {})


@pytest.mark.asyncio
async def test_non_ascii_signature_header_rejected_cleanly():
    """A non-ASCII header must 401, not blow up ``hmac.compare_digest``.

    ``compare_digest`` raises ``TypeError`` on a ``str`` containing a
    non-ASCII character — real ASGI header decoding admits obs-text, so a
    forged byte like ``\\xff`` reaches ``verify()`` as a genuine character
    rather than being rejected earlier.
    """
    with pytest.raises(WebhookVerificationError) as excinfo:
        await _sha256_prefix_verifier().verify(b"payload", {"X-Hub-Signature-256": "sha256=ab\xff"})
    assert excinfo.value.status_code == 401


@pytest.mark.asyncio
async def test_unconfigured_secret_rejects_rather_than_signing_with_empty_key():
    """The bypass this guards: HMAC("" , body) is a valid HMAC.

    Without the explicit empty-secret check, a deployment that forgot the
    webhook secret would accept any payload signed with an empty key — an
    open endpoint presenting as a working one.
    """
    body = b"payload"
    forged = f"sha256={_hex('', body)}"

    with pytest.raises(WebhookVerificationError):
        await _sha256_prefix_verifier(secret="").verify(body, {"X-Hub-Signature-256": forged})


@pytest.mark.asyncio
async def test_a_secret_provider_returning_none_rejects_too():
    body = b"payload"

    with pytest.raises(WebhookVerificationError):
        await _sha256_prefix_verifier(secret=None).verify(
            body, {"X-Hub-Signature-256": f"sha256={_hex('', body)}"}
        )


# ---- header matching ---------------------------------------------------


@pytest.mark.parametrize(
    "header_name",
    ["x-hub-signature-256", "X-Hub-Signature-256", "X-HUB-SIGNATURE-256"],
    ids=["lower", "canonical", "upper"],
)
@pytest.mark.asyncio
async def test_headers_are_matched_case_insensitively(header_name):
    """Hosts keep headers in whatever case their framework does.

    Starlette lower-cases, a raw environ upper-cases, a test writes the
    provider's own spelling — a verifier matching only one would 401 every
    delivery for the others with the cause nowhere near the symptom.
    """
    body = b"payload"

    assert (
        await _sha256_prefix_verifier().verify(body, {header_name: f"sha256={_hex(_SECRET, body)}"})
        == body
    )


@pytest.mark.asyncio
async def test_any_mapping_of_headers_is_accepted():
    """A host may hand over its framework's own header mapping, not a dict."""

    class _Headers:
        def __init__(self, raw):
            self._raw = raw

        def items(self):
            return self._raw.items()

    body = b"payload"
    headers = _Headers({"X-Hub-Signature-256": f"sha256={_hex(_SECRET, body)}"})

    assert await _sha256_prefix_verifier().verify(body, headers) == body


# ---- bare_hex (Linear) -------------------------------------------------


@pytest.mark.asyncio
async def test_bare_hex_valid_signature_passes():
    body = b'{"organizationId":"org-1"}'
    verifier = _verifier(SignatureScheme.BARE_HEX, header="linear-signature")

    assert await verifier.verify(body, {"Linear-Signature": _hex(_SECRET, body)}) == body


@pytest.mark.asyncio
async def test_bare_hex_rejects_prefixed_signature():
    body = b"payload"
    verifier = _verifier(SignatureScheme.BARE_HEX, header="linear-signature")

    with pytest.raises(WebhookVerificationError):
        await verifier.verify(body, {"Linear-Signature": f"sha256={_hex(_SECRET, body)}"})


# ---- slack_v0 ----------------------------------------------------------


def _slack_verifier(**kwargs) -> HmacVerifier:
    return _verifier(
        SignatureScheme.SLACK_V0,
        header="x-slack-signature",
        timestamp_header="x-slack-request-timestamp",
        **kwargs,
    )


def _slack_signature(body: bytes, timestamp: int, secret: str = _SECRET) -> str:
    return "v0=" + _hex(secret, f"v0:{timestamp}:".encode() + body)


@pytest.mark.asyncio
async def test_slack_v0_valid_signature_passes():
    body = b"command=%2Fcognee-ask&team_id=T1"
    now = int(time.time())
    headers = {
        "X-Slack-Signature": _slack_signature(body, now),
        "X-Slack-Request-Timestamp": str(now),
    }

    assert await _slack_verifier().verify(body, headers) == body


@pytest.mark.asyncio
@pytest.mark.parametrize("offset", [-600, 600], ids=["stale", "future_dated"])
async def test_slack_v0_replay_window_enforced_in_both_directions(offset):
    """A future-dated timestamp is as suspect as a stale one.

    Checking only the past leaves an attacker free to sign far-future
    timestamps and replay them indefinitely once captured.
    """
    body = b"payload"
    timestamp = int(time.time()) + offset
    headers = {
        "X-Slack-Signature": _slack_signature(body, timestamp),
        "X-Slack-Request-Timestamp": str(timestamp),
    }

    with pytest.raises(WebhookVerificationError):
        await _slack_verifier().verify(body, headers)


@pytest.mark.parametrize(
    "raw_timestamp",
    ["inf", "-inf", "Infinity", "nan", "1e400", "-1e400"],
    ids=["inf", "neg_inf", "infinity_word", "nan", "overflow_exp", "neg_overflow_exp"],
)
@pytest.mark.asyncio
async def test_slack_v0_non_finite_timestamp_is_rejected_not_crashed(raw_timestamp):
    """These parse as floats but overflow ``int()``, raising ``OverflowError``.

    That is neither ``TypeError`` nor ``ValueError``, so a naive parser lets it
    escape as an unhandled 500 — reachable before the HMAC comparison, so by
    any anonymous caller. The contract is 401 for every bad input, without
    exception.
    """
    headers = {"X-Slack-Signature": "v0=irrelevant", "X-Slack-Request-Timestamp": raw_timestamp}

    with pytest.raises(WebhookVerificationError) as excinfo:
        await _slack_verifier().verify(b"payload", headers)
    assert excinfo.value.status_code == 401


@pytest.mark.parametrize(
    "raw_timestamp",
    ["9" * 400, "-" + "9" * 400, str(10**11 + 1)],
    ids=["giant", "giant_negative", "just_past_bound"],
)
@pytest.mark.asyncio
async def test_slack_v0_implausibly_large_timestamp_is_rejected_not_crashed(raw_timestamp):
    """Python ints are arbitrary precision, so a run of digits parses cleanly.

    It only explodes later, when the replay-window check subtracts it from a
    float — ``OverflowError``, uncaught, before the HMAC is ever compared. The
    non-finite-float guard does not cover this: the value never becomes a
    float on the way in.
    """
    headers = {"X-Slack-Signature": "v0=irrelevant", "X-Slack-Request-Timestamp": raw_timestamp}

    with pytest.raises(WebhookVerificationError) as excinfo:
        await _slack_verifier().verify(b"payload", headers)
    assert excinfo.value.status_code == 401


@pytest.mark.asyncio
async def test_timestamped_implausibly_large_timestamp_is_rejected_not_crashed():
    """Same trap on the compound-header path."""
    with pytest.raises(WebhookVerificationError) as excinfo:
        await _timestamped_verifier().verify(
            b"payload", {"X-Signature": f"t={'9' * 400},v1=deadbeef"}
        )
    assert excinfo.value.status_code == 401


@pytest.mark.asyncio
async def test_timestamped_non_finite_timestamp_is_rejected_not_crashed():
    """Same trap on the compound-header path."""
    with pytest.raises(WebhookVerificationError) as excinfo:
        await _timestamped_verifier().verify(b"payload", {"X-Signature": "t=1e400,v1=deadbeef"})
    assert excinfo.value.status_code == 401


@pytest.mark.asyncio
async def test_slack_v0_signs_the_raw_timestamp_string():
    """The signed message uses the header verbatim, not a reparsed copy.

    The sender signed the bytes it sent. A fractional timestamp survives
    ``int(float(x))`` for the replay check but would not survive being
    reformatted back into the signed string — and the resulting mismatch
    presents as "bad signature" on every delivery, nowhere near its cause.
    """
    body = b"payload"
    raw = f"{int(time.time())}.500"
    expected = "v0=" + _hex(_SECRET, f"v0:{raw}:".encode() + body)
    headers = {"X-Slack-Signature": expected, "X-Slack-Request-Timestamp": raw}

    assert await _slack_verifier().verify(body, headers) == body


@pytest.mark.asyncio
async def test_slack_v0_unparseable_timestamp_rejected():
    body = b"payload"
    headers = {
        "X-Slack-Signature": _slack_signature(body, int(time.time())),
        "X-Slack-Request-Timestamp": "not-a-number",
    }

    with pytest.raises(WebhookVerificationError):
        await _slack_verifier().verify(body, headers)


@pytest.mark.asyncio
async def test_slack_v0_missing_timestamp_header_rejected():
    body = b"payload"

    with pytest.raises(WebhookVerificationError):
        await _slack_verifier().verify(
            body, {"X-Slack-Signature": _slack_signature(body, int(time.time()))}
        )


def test_a_plain_string_scheme_refuses_construction():
    """``SignatureScheme`` is a str enum, so a plain string like
    ``"sha256_prefix"`` imports and equality-compares as if it were the real
    member — but ``verify()``'s dispatch is `is`-based, which a raw string
    never matches. Left unchecked, every delivery would 401 forever with
    "unsupported scheme", even under a genuinely correct signature, and the
    cause would be nowhere near the symptom. Caught at construction instead
    of at the first verified-but-misdispatched request.
    """
    with pytest.raises(ValueError, match="SignatureScheme member"):
        HmacVerifier(
            provider="testprov",
            scheme="sha256_prefix",
            header="x-signature",
            secret_provider=lambda: _SECRET,
        )


def test_a_plain_string_slack_v0_scheme_raises_value_error_not_attribute_error():
    """The scheme-type guard must run before the SLACK_V0 misconfiguration
    check touches ``scheme.value`` — a plain str has no ``.value``, so
    without the type guard running first this same bad input would crash
    with ``AttributeError`` instead of the ``ValueError`` the misconfiguration
    check is supposed to raise.
    """
    with pytest.raises(ValueError, match="SignatureScheme member"):
        HmacVerifier(
            provider="testprov",
            scheme="slack_v0",
            header="x-slack-signature",
            secret_provider=lambda: _SECRET,
        )


def test_slack_v0_without_timestamp_header_refuses_construction():
    """Misconfiguration fails at wiring time, not by silently weakening the check.

    A timestamped scheme with no timestamp header would verify the HMAC and
    quietly skip replay protection — serving a weaker guarantee than the
    scheme name promises, with nothing in the logs to say so.
    """
    with pytest.raises(ValueError, match="requires timestamp_header"):
        _verifier(SignatureScheme.SLACK_V0, header="x-slack-signature")


# ---- timestamped (Stripe-shaped) ---------------------------------------


def _timestamped_verifier(**kwargs) -> HmacVerifier:
    return _verifier(SignatureScheme.TIMESTAMPED, header="x-signature", **kwargs)


@pytest.mark.asyncio
async def test_timestamped_valid_compound_header_passes():
    body = b'{"id":"evt_1"}'
    now = int(time.time())
    digest = _hex(_SECRET, f"{now}.".encode() + body)

    assert await _timestamped_verifier().verify(body, {"X-Signature": f"t={now},v1={digest}"}) == (
        body
    )


@pytest.mark.asyncio
async def test_timestamped_tolerates_unknown_header_components():
    """Providers add scheme versions over time; an unknown one must not break v1."""
    body = b"payload"
    now = int(time.time())
    digest = _hex(_SECRET, f"{now}.".encode() + body)
    headers = {"X-Signature": f"t={now},v1={digest},v2=ignored"}

    assert await _timestamped_verifier().verify(body, headers) == body


@pytest.mark.asyncio
async def test_timestamped_malformed_header_rejected():
    with pytest.raises(WebhookVerificationError):
        await _timestamped_verifier().verify(b"payload", {"X-Signature": "garbage-without-parts"})


@pytest.mark.asyncio
async def test_timestamped_stale_timestamp_rejected():
    body = b"payload"
    stale = int(time.time()) - 600
    digest = _hex(_SECRET, f"{stale}.".encode() + body)

    with pytest.raises(WebhookVerificationError):
        await _timestamped_verifier().verify(body, {"X-Signature": f"t={stale},v1={digest}"})


@pytest.mark.asyncio
async def test_timestamped_accepts_the_first_of_two_v1_values():
    """Stripe's documented rotation behavior: during a signing-secret
    rotation a delivery carries one ``v1`` per active secret, and is valid if
    ANY of them matches. ``dict(piece.split("=", 1) ...)`` keeps only the
    LAST ``v1=`` in the header — this checks that the FIRST one, which would
    be silently dropped by that bug, is still honored.
    """
    body = b'{"id":"evt_1"}'
    now = int(time.time())
    valid_digest = _hex(_SECRET, f"{now}.".encode() + body)
    headers = {"X-Signature": f"t={now},v1={valid_digest},v1={'0' * 64}"}

    assert await _timestamped_verifier().verify(body, headers) == body


@pytest.mark.asyncio
async def test_timestamped_accepts_the_second_of_two_v1_values():
    """The mirror of the case above — whichever position the matching secret
    lands in, not just the first.
    """
    body = b'{"id":"evt_1"}'
    now = int(time.time())
    valid_digest = _hex(_SECRET, f"{now}.".encode() + body)
    headers = {"X-Signature": f"t={now},v1={'0' * 64},v1={valid_digest}"}

    assert await _timestamped_verifier().verify(body, headers) == body


@pytest.mark.asyncio
async def test_timestamped_rejects_when_no_v1_value_matches():
    body = b'{"id":"evt_1"}'
    now = int(time.time())
    headers = {"X-Signature": f"t={now},v1={'0' * 64},v1={'1' * 64}"}

    with pytest.raises(WebhookVerificationError):
        await _timestamped_verifier().verify(body, headers)


# ---- Cross-scheme ------------------------------------------------------


@pytest.mark.asyncio
async def test_rejection_detail_is_uniform_across_failure_modes():
    """Every rejection says the same thing, whatever actually failed.

    Distinguishable messages hand a caller a free oracle for probing which
    check it tripped. The specific reason belongs in the log, which is useful
    to us and not readable by the sender.
    """
    verifier = _sha256_prefix_verifier()
    details = set()

    for headers in (
        {},
        {"X-Hub-Signature-256": "sha256=deadbeef"},
        {"X-Hub-Signature-256": "malformed"},
    ):
        with pytest.raises(WebhookVerificationError) as excinfo:
            await verifier.verify(b"payload", headers)
        details.add(excinfo.value.detail)

    assert details == {"Invalid signature"}


@pytest.mark.asyncio
async def test_the_specific_reason_goes_to_the_log(caplog):
    with caplog.at_level("WARNING"), pytest.raises(WebhookVerificationError):
        await _sha256_prefix_verifier().verify(b"payload", {})

    assert "Rejected testprov webhook: missing x-hub-signature-256" in caplog.text


def test_a_rejection_is_a_connector_error_with_a_status_and_detail():
    error = WebhookVerificationError()

    assert isinstance(error, ConnectorError)
    assert error.status_code == 401
    assert error.detail == "Invalid signature"
    assert str(error) == "Invalid signature"


def test_scheme_members_spell_as_their_value():
    """Descriptors and catalogs serialize the scheme; the name must not leak in."""
    assert str(SignatureScheme.SHA256_PREFIX) == "sha256_prefix"
    assert f"{SignatureScheme.SLACK_V0}" == "slack_v0"
    assert SignatureScheme("bare_hex") is SignatureScheme.BARE_HEX
    assert SignatureScheme.TIMESTAMPED == "timestamped"


def test_the_verifier_has_no_instance_dict():
    """``WebhookVerifier.__slots__ = ()`` is what makes the subclass's slots real."""
    verifier = _sha256_prefix_verifier()

    assert not hasattr(verifier, "__dict__")


# ---- body size cap -----------------------------------------------------


@pytest.mark.asyncio
async def test_an_oversized_content_length_is_rejected_before_the_signature_check():
    """The common case, and the cheap one: a truthful, oversized header.

    No signature can be computed here to make this pass — the point is that
    it never has to be. A host that streams the body can stop at the same
    number (``MAX_BODY_BYTES`` is public for that) before spending memory on it.
    """
    headers = {"Content-Length": str(MAX_BODY_BYTES + 1), "X-Hub-Signature-256": "x"}

    with pytest.raises(WebhookVerificationError) as excinfo:
        await _sha256_prefix_verifier().verify(b"payload", headers)

    assert excinfo.value.status_code == 413


@pytest.mark.asyncio
async def test_an_oversized_body_is_rejected_even_without_a_truthful_content_length():
    """`Content-Length` is caller-supplied and optional; the cap must not trust it.

    A chunked delivery carries no header at all, so the bytes themselves are
    measured — the header-only check above would wave this straight through.
    """
    body = b"a" * (MAX_BODY_BYTES + 1)

    with pytest.raises(WebhookVerificationError) as excinfo:
        await _sha256_prefix_verifier().verify(body, {"X-Hub-Signature-256": "x"})

    assert excinfo.value.status_code == 413


@pytest.mark.asyncio
async def test_a_malformed_content_length_falls_through_to_the_signature_check():
    """Not a size problem — the signature check is what should refuse this."""
    with pytest.raises(WebhookVerificationError) as excinfo:
        await _sha256_prefix_verifier().verify(b"payload", {"Content-Length": "not-a-number"})

    assert excinfo.value.status_code == 401


@pytest.mark.asyncio
async def test_a_body_within_the_cap_still_verifies():
    body = b"small"
    headers = {
        "Content-Length": str(len(body)),
        "X-Hub-Signature-256": f"sha256={_hex(_SECRET, body)}",
    }

    assert await _sha256_prefix_verifier().verify(body, headers) == body
