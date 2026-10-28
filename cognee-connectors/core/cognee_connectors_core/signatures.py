"""Webhook signature schemes, as data rather than per-provider code.

Moved from cloud-backend ``modules.integrations.framework.signatures``
(cognee-saas-backend, 2026-10-26) and re-expressed on the host-agnostic
:class:`~.base.WebhookVerifier`: the verifier takes the raw body and the
request headers instead of a FastAPI request, and refuses a delivery with
:class:`~.errors.WebhookVerificationError` instead of an ``HTTPException``.

Across the roadmap's providers there are only a handful of distinct HMAC
constructions, and every provider reuses one of them. So a connector names a
scheme instead of writing verification code — which matters because this is
the one place in a connector where a subtle mistake is a silent
authentication bypass rather than a visible bug. Written once, tested once
(``core/tests/test_signatures.py``), reused 50 times.

The schemes in use:

============== ================================================= ==============
Scheme         Signed string                                     Providers
============== ================================================= ==============
sha256_prefix  ``{body}``, header ``sha256=<hex>``               GitHub, Jira
bare_hex       ``{body}``, header ``<hex>``                      Linear
slack_v0       ``v0:{timestamp}:{body}``, header ``v0=<hex>``    Slack
timestamped    ``{timestamp}.{body}``, header ``t=..,v1=<hex>``  Stripe-shaped
============== ================================================= ==============

Two invariants hold for all of them, and both are load-bearing:

**Verify over raw bytes, before parsing.** Re-serializing a parsed body
changes key order and escaping and breaks every scheme here. Verifiers
therefore return the raw bytes for the handler to parse.

**Compare in constant time.** ``hmac.compare_digest``, never ``==``: a
short-circuiting comparison leaks how much of a candidate signature was
correct, which is enough to forge one byte at a time.

Only ``slack_v0`` and ``timestamped`` carry a timestamp, so only they can
enforce a replay window. For the others, replay protection is the handler
being idempotent — which is why :meth:`~.base.Connector.handle_webhook`
requires it.
"""

import hashlib
import hmac
import logging
import math
import time
from collections.abc import Callable, Mapping
from enum import Enum

from cognee_connectors_core.base import WebhookVerifier
from cognee_connectors_core.errors import WebhookVerificationError

logger = logging.getLogger(__name__)

# Reject a timestamped delivery skewed further than this from local time.
# Slack's documented window; reused for the Stripe-shaped scheme, whose own
# tolerance is advisory rather than specified.
_MAX_TIMESTAMP_SKEW_SECONDS = 60 * 5

# Any unix timestamp a real provider sends is far inside this; anything beyond
# it is a probe, not a clock. The bound exists so an arbitrarily large integer
# never reaches the float arithmetic in the replay-window check, where it would
# raise OverflowError before the signature is ever compared. ~year 5138.
_MAX_PLAUSIBLE_TIMESTAMP = 10**11

# No real delivery from any provider on the roadmap comes close to this — a
# webhook body is an event description, not a file upload. The route a host
# mounts this on is unauthenticated until the signature check below passes, so
# nothing else bounds how much of a request an anonymous caller can make a
# process buffer before that check runs. Public so a host that streams the
# body can stop reading at the same point this verifier would refuse it; a
# ``Content-Length`` lie or a chunked body with no length header at all means
# the header alone cannot be trusted either, so the bytes are checked too.
MAX_BODY_BYTES = 5_000_000


class SignatureScheme(str, Enum):
    """How a provider signs its webhooks. See the module docstring.

    A ``str`` mixin (rather than ``enum.StrEnum``, which is 3.11+) so a member
    compares equal to its value and serializes as it; ``__str__`` is pinned to
    the value so the two spell the same on every supported Python.
    """

    SHA256_PREFIX = "sha256_prefix"
    BARE_HEX = "bare_hex"
    SLACK_V0 = "slack_v0"
    TIMESTAMPED = "timestamped"

    def __str__(self) -> str:
        return str(self.value)


def _digest(secret: str, message: bytes) -> str:
    return hmac.new(secret.encode(), message, hashlib.sha256).hexdigest()


def _reject(provider: str, reason: str) -> WebhookVerificationError:
    """A uniform 401 for every rejection.

    The response text is deliberately identical whatever went wrong — a missing
    header, a bad timestamp, a wrong digest. Telling a caller *which* check it
    failed hands it a free oracle for probing the rest. The specific reason goes
    to the log, where it is useful and not attacker-readable.
    """
    logger.warning("Rejected %s webhook: %s", provider, reason)
    return WebhookVerificationError("Invalid signature", status_code=401)


def _parse_timestamp(raw: str | None) -> int | None:
    """Parse a delivery timestamp, returning ``None`` for anything unusable.

    Reachable before the HMAC comparison, so any anonymous caller can steer it
    and it must not raise for *any* header value. Two distinct traps, both of
    which turn a would-be 401 into an unhandled 500 (and an on-call page):

    * ``float()`` accepts ``inf``, ``nan`` and overflowing exponents like
      ``1e400``; ``int()`` on those raises ``OverflowError``, which is neither
      of the exceptions a naive ``except (TypeError, ValueError)`` catches.
    * Python ints are arbitrary precision, so a plain run of digits — ``"9"``
      four hundred times — parses cleanly here and only explodes later, when
      the replay-window check subtracts it from a float.

    The magnitude bound is what closes the second one, and it is a real check
    rather than defensive noise: a unix timestamp outside it is not a clock
    skew, it is a value no provider would ever send.
    """
    if raw is None:
        return None

    parsed: int | None = None
    try:
        parsed = int(raw)
    except (TypeError, ValueError):
        # A provider sending fractional seconds still has a usable timestamp.
        try:
            as_float = float(raw)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(as_float):
            return None
        parsed = int(as_float)

    if abs(parsed) > _MAX_PLAUSIBLE_TIMESTAMP:
        return None
    return parsed


def _lower_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Header names folded to lower case, so lookups are case-insensitive.

    Hosts keep headers in whatever case their framework does (Starlette lower-
    cases, a raw WSGI environ upper-cases, a test writes them canonically), and
    a verifier that matched only one spelling would 401 every delivery for the
    others with the cause nowhere near the symptom.
    """
    return {str(name).lower(): value for name, value in headers.items()}


class HmacVerifier(WebhookVerifier):
    """Verifies a delivery against one of the :class:`SignatureScheme` shapes.

    The secret is resolved per request through ``secret_provider`` rather than
    captured at construction, so a rotated secret takes effect without
    re-registering the connector — as soon as the settings object the provider
    reads from sees the new value. ``secret_provider`` returning an empty value
    refuses every delivery; it never degrades into an HMAC under an empty key.
    """

    __slots__ = ("_secret_provider", "header", "provider", "scheme", "timestamp_header")

    def __init__(
        self,
        *,
        provider: str,
        scheme: SignatureScheme,
        header: str,
        secret_provider: Callable[[], str | None],
        timestamp_header: str | None = None,
    ) -> None:
        if not isinstance(scheme, SignatureScheme):
            # A plain string like "sha256_prefix" imports cleanly and equality-
            # compares as if it were fine, but verify()'s dispatch is `is`-based
            # against the enum members — a raw string never matches any branch
            # and falls through to "unsupported scheme", 401ing every delivery
            # forever even with a genuinely correct signature. Checked first,
            # before scheme.value is ever touched below, or this same bad input
            # crashes the SLACK_V0 guard with AttributeError instead of the
            # ValueError it is supposed to raise.
            raise ValueError(f"scheme must be a SignatureScheme member, got {scheme!r}")
        if scheme in (SignatureScheme.SLACK_V0,) and not timestamp_header:
            # A timestamped scheme without its timestamp header silently loses
            # replay protection, so refuse the misconfiguration at wiring time
            # rather than serving a weaker check than the scheme name promises.
            raise ValueError(f"{scheme.value} requires timestamp_header")

        self.provider = provider
        self.scheme = scheme
        self.header = header.lower()
        self.timestamp_header = timestamp_header.lower() if timestamp_header else None
        self._secret_provider = secret_provider

    async def verify(self, body: bytes, headers: Mapping[str, str]) -> bytes:
        headers = _lower_headers(headers)

        # Checked twice, for two different liars. A `Content-Length` past the
        # cap is rejected first — the common case, and free. But the header is
        # caller-supplied and optional (a chunked body carries none at all),
        # so the bytes themselves are measured too, regardless of what, if
        # anything, `Content-Length` claims.
        declared = headers.get("content-length")
        if declared is not None:
            try:
                if int(declared) > MAX_BODY_BYTES:
                    raise WebhookVerificationError("Payload too large", status_code=413)
            except ValueError:
                pass  # Malformed, not oversized — the signature check rejects it.
        if len(body) > MAX_BODY_BYTES:
            raise WebhookVerificationError("Payload too large", status_code=413)

        signature = headers.get(self.header)
        if not signature:
            raise _reject(self.provider, f"missing {self.header}")
        if not signature.isascii():
            # hmac.compare_digest raises TypeError on a str containing a
            # non-ASCII character — an unhandled 500 instead of the 401 this
            # method promises, from an anonymous caller on an unauthenticated
            # route. ASGI header decoding (Starlette/h11) admits obs-text, so
            # a byte like `\xff` in the header reaches here as a real
            # character rather than being rejected earlier. Every scheme's
            # signature is ASCII by construction (hex digest, optionally
            # prefixed), so this can only ever reject forged input.
            raise _reject(self.provider, "signature mismatch")

        secret = self._secret_provider()
        if not secret:
            # Never fall through to an HMAC against an empty key: that would
            # accept any payload whose signature the sender computed the same
            # way, turning a config gap into an open endpoint.
            raise _reject(self.provider, "webhook secret is not configured")

        if self.scheme is SignatureScheme.SHA256_PREFIX:
            expected = f"sha256={_digest(secret, body)}"
        elif self.scheme is SignatureScheme.BARE_HEX:
            expected = _digest(secret, body)
        elif self.scheme is SignatureScheme.SLACK_V0:
            expected = self._slack_v0_expected(headers, body, secret)
        elif self.scheme is SignatureScheme.TIMESTAMPED:
            expected, signature = self._timestamped_expected(signature, body, secret)
        else:  # pragma: no cover — Enum membership is exhaustive above.
            raise _reject(self.provider, f"unsupported scheme {self.scheme}")

        if not hmac.compare_digest(expected, signature):
            raise _reject(self.provider, "signature mismatch")

        return body

    def _check_replay_window(self, raw_timestamp: str | None) -> None:
        """Reject a delivery whose timestamp is outside the replay window.

        ``abs()``: a future-dated timestamp is as suspect as a stale one, and
        checking only one direction leaves the other wide open to a captured
        delivery being replayed indefinitely.
        """
        timestamp = _parse_timestamp(raw_timestamp)
        if timestamp is None:
            raise _reject(self.provider, "missing or unparseable timestamp")
        if abs(time.time() - timestamp) > _MAX_TIMESTAMP_SKEW_SECONDS:
            raise _reject(self.provider, "timestamp outside replay window")

    def _slack_v0_expected(self, headers: Mapping[str, str], raw_body: bytes, secret: str) -> str:
        raw_timestamp = headers.get(self.timestamp_header or "")
        self._check_replay_window(raw_timestamp)

        # The *raw* header string goes into the signed message, never a parsed
        # and reformatted copy of it. The sender signed the bytes it sent, so
        # any round-trip that is not byte-exact (a fractional component, a
        # leading zero, padding) computes a different digest and rejects every
        # delivery from that provider with "signature mismatch" — a failure
        # whose cause is nowhere near its symptom. Matches the precedent in
        # cognee's slack/verify_slack_signature.py, which also signs the raw
        # string.
        signed = b"v0:" + str(raw_timestamp).encode() + b":" + raw_body
        return f"v0={_digest(secret, signed)}"

    def _timestamped_expected(
        self, signature: str, raw_body: bytes, secret: str
    ) -> tuple[str, str]:
        """Verify a ``t=<ts>,v1=<hex>`` header, accepting any matching ``v1``.

        Stripe's documented rotation behavior: a delivery carries one ``v1``
        per currently-active signing secret, and is valid if ANY of them
        matches — that is how a secret rotates without a gap where deliveries
        signed under the old secret are rejected before every consumer has
        picked up the new one. A ``dict()`` over ``piece.split("=", 1)`` would
        silently keep only the *last* ``v1=`` and drop the rest, so this
        collects every one and checks each in turn.

        Still returns a single ``(expected, candidate)`` pair to the caller's
        ``hmac.compare_digest`` — the caller's own timing behavior does not
        change, since ``expected`` depends only on ``raw_timestamp``/
        ``raw_body``/``secret`` and is identical for every candidate.
        """
        raw_timestamp: str | None = None
        candidates: list[str] = []
        for piece in signature.split(","):
            if "=" not in piece:
                continue
            key, value = piece.split("=", 1)
            if key == "t" and raw_timestamp is None:
                raw_timestamp = value
            elif key == "v1":
                candidates.append(value)

        if not raw_timestamp or not candidates:
            raise _reject(self.provider, "malformed compound signature header")
        self._check_replay_window(raw_timestamp)

        # Raw header string, for the reason given in _slack_v0_expected.
        signed = raw_timestamp.encode() + b"." + raw_body
        expected = _digest(secret, signed)

        for candidate in candidates:
            if hmac.compare_digest(expected, candidate):
                # Return this candidate as-is: verify() re-runs the exact same
                # compare_digest(expected, candidate) call, so the constant-time
                # property holds per-candidate the same way it always did for
                # the single-candidate case.
                return expected, candidate

        # None matched. Returning the first candidate keeps verify()'s own
        # compare_digest a real (failing) comparison rather than synthesizing
        # a bypass value — same rejection path as before, just after checking
        # every rotated secret instead of only the last one in the header.
        return expected, candidates[0]


__all__ = ["MAX_BODY_BYTES", "HmacVerifier", "SignatureScheme"]
