"""Test helpers for connector packages: a scripted HTTP client and fixtures.

Importable without pytest. A connector's tests exercise its contract object
against the provider's real token-response shape through
:class:`FakeHttpClient`, so nothing in a test suite touches the network.
"""

import hashlib
import hmac
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from cognee_connectors_core.descriptor import ConnectorDescriptor
from cognee_connectors_core.signatures import SignatureScheme

Response = tuple[int, Any]


@dataclass(slots=True)
class RecordedCall:
    """One request a :class:`FakeHttpClient` received."""

    method: str
    url: str
    data: dict[str, str] | None
    headers: dict[str, str]


class FakeHttpClient:
    """An :class:`~.http.HttpClient` that answers from a script and records calls.

    Responses are ``(status, body)`` pairs — ``body`` is whatever the real
    client would have decoded, so a ``dict`` for JSON and a ``str`` for a
    non-JSON answer. An exception instance in place of a pair is raised
    instead of returned, for transport failures.

    Two ways to script, resolved in this order:

    * **Keyed** by ``(method, url)``, each key holding one response or a list
      consumed in order whose last entry repeats — ``FakeHttpClient({("POST",
      token_url): (200, {...})})``, or :meth:`add` after construction.
    * **Ordered**, a plain sequence consumed first-in first-out whatever the
      request — ``FakeHttpClient([(200, {...}), (401, {...})])``, or
      :meth:`queue`.

    ``default`` answers anything neither script covers; without one an
    unscripted request fails the test, which is the point.
    """

    def __init__(
        self,
        responses: Mapping[tuple[str, str], Any] | Sequence[Any] | None = None,
        *,
        default: Any = None,
    ) -> None:
        self.calls: list[RecordedCall] = []
        self.default = default
        self._keyed: dict[tuple[str, str], list[Any]] = {}
        self._queue: list[Any] = []
        if isinstance(responses, Mapping):
            for (method, url), scripted in responses.items():
                if isinstance(scripted, list):
                    self.add(method, url, *scripted)
                else:
                    self.add(method, url, scripted)
        elif responses is not None:
            self.queue(*responses)

    def add(self, method: str, url: str, *responses: Any) -> "FakeHttpClient":
        """Script ``responses`` for ``method url``, in order; the last one repeats."""
        self._keyed.setdefault((method.upper(), url), []).extend(responses)
        return self

    def queue(self, *responses: Any) -> "FakeHttpClient":
        """Script ``responses`` for the next requests, whatever they are."""
        self._queue.extend(responses)
        return self

    async def post_form(
        self,
        url: str,
        data: Mapping[str, str],
        headers: Mapping[str, str] | None = None,
    ) -> Response:
        return self._respond("POST", url, dict(data), headers)

    async def get_json(
        self,
        url: str,
        headers: Mapping[str, str] | None = None,
    ) -> Response:
        return self._respond("GET", url, None, headers)

    def _respond(
        self,
        method: str,
        url: str,
        data: dict[str, str] | None,
        headers: Mapping[str, str] | None,
    ) -> Response:
        self.calls.append(RecordedCall(method, url, data, dict(headers or {})))

        scripted = self._keyed.get((method, url))
        if scripted:
            response = scripted.pop(0) if len(scripted) > 1 else scripted[0]
        elif self._queue:
            response = self._queue.pop(0)
        elif self.default is not None:
            response = self.default
        else:
            raise AssertionError(f"FakeHttpClient has no scripted response for {method} {url}")

        if isinstance(response, BaseException):
            raise response
        return response

    @property
    def last_call(self) -> RecordedCall:
        """The most recent request, for asserting on its form fields or headers."""
        if not self.calls:
            raise AssertionError("FakeHttpClient received no calls")
        return self.calls[-1]

    def calls_to(self, method: str, url: str) -> list[RecordedCall]:
        """Every request made to ``method url``, in order."""
        return [call for call in self.calls if call.method == method.upper() and call.url == url]


class FakeSettings:
    """A :class:`~.settings.ConnectorSettings` stand-in, with no environment involved.

    Carries the four shared fields with usable defaults plus whatever extra
    attributes a test passes (``webhook_secret="..."``), and a ``model_config``
    with an ``env_prefix`` so :func:`~.settings.require` names the right
    environment variable in its error.
    """

    def __init__(
        self,
        *,
        client_id: str = "client-id",
        client_secret: str = "client-secret",
        redirect_uri: str = "https://host.example/api/v1/integrations/testprov/callback",
        frontend_base_url: str = "https://app.example",
        env_prefix: str = "TESTPROV_",
        **extra: Any,
    ) -> None:
        self.model_config = {"env_prefix": env_prefix}
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.frontend_base_url = frontend_base_url
        for name, value in extra.items():
            setattr(self, name, value)


@dataclass(slots=True)
class FakeCredential:
    """A :class:`~.base.Credential` for tests; the attributes a connector may read."""

    provider: str = "testprov"
    provider_account_id: str = "acct-1"
    account_label: str | None = "Acme Corp"
    provider_metadata: dict[str, Any] | None = field(default_factory=dict)
    scopes: str | None = None
    token_expires_at: Any = None
    auth_type: str = "oauth2"


def _hex(secret: str, message: bytes) -> str:
    return hmac.new(secret.encode(), message, hashlib.sha256).hexdigest()


def sign_body(
    scheme: SignatureScheme,
    secret: str,
    body: bytes,
    *,
    timestamp: int | str | None = None,
) -> str:
    """The signature header value a provider using ``scheme`` would send for ``body``.

    ``timestamp`` is required for the timestamped schemes (``slack_v0``,
    ``timestamped``) and ignored by the others; pass the same value in the
    timestamp header (see :func:`webhook_headers`, which does both).
    """
    if scheme is SignatureScheme.SHA256_PREFIX:
        return f"sha256={_hex(secret, body)}"
    if scheme is SignatureScheme.BARE_HEX:
        return _hex(secret, body)
    if timestamp is None:
        raise ValueError(f"{scheme.value} needs a timestamp to sign")
    raw_timestamp = str(timestamp)
    if scheme is SignatureScheme.SLACK_V0:
        return "v0=" + _hex(secret, f"v0:{raw_timestamp}:".encode() + body)
    if scheme is SignatureScheme.TIMESTAMPED:
        return f"t={raw_timestamp},v1={_hex(secret, f'{raw_timestamp}.'.encode() + body)}"
    raise ValueError(f"unsupported scheme {scheme!r}")


def webhook_headers(
    descriptor: ConnectorDescriptor,
    secret: str,
    body: bytes,
    *,
    event: str | None = None,
    timestamp: int | str | None = None,
) -> dict[str, str]:
    """Headers for a delivery of ``body`` that ``descriptor``'s verifier accepts.

    Fills the signature header, the timestamp header when the scheme has one
    (``timestamp`` defaults to now), and the event header when ``event`` is
    given. Header names are the descriptor's own spelling.
    """
    scheme = descriptor.signature_scheme
    if scheme is None or not descriptor.signature_header:
        raise ValueError(f"{descriptor.provider} declares no webhook signature")
    if timestamp is None:
        timestamp = int(time.time())
    headers = {descriptor.signature_header: sign_body(scheme, secret, body, timestamp=timestamp)}
    if descriptor.signature_timestamp_header:
        headers[descriptor.signature_timestamp_header] = str(timestamp)
    if event is not None and descriptor.webhook_event_header:
        headers[descriptor.webhook_event_header] = event
    return headers


__all__ = [
    "FakeCredential",
    "FakeHttpClient",
    "FakeSettings",
    "RecordedCall",
    "sign_body",
    "webhook_headers",
]
