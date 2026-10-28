"""Slack request signature verification, on the core webhook verifier.

Moved from cognee.modules.integrations.slack.verify_slack_signature (cognee dev,
2026-10-27). The HMAC is now :class:`cognee_connectors_core.signatures.HmacVerifier`
with the ``slack_v0`` scheme, the one ``CONNECTOR`` declares, so the hand-written
Slack routes and a host serving the generic ``/events`` route verify deliveries the
same way. The FastAPI dependency that sat next to the check lives in
:mod:`cognee_connector_slack.cognee_host.verify_slack_signature`.

Every inbound Slack payload — slash commands, Events API, interactivity — is
authenticated by an HMAC over ``v0:{timestamp}:{raw_body}`` keyed with the
app's signing secret, compared against the ``X-Slack-Signature`` header
(https://docs.slack.dev/authentication/verifying-requests-from-slack).

The check MUST run over the raw request bytes, before any parsing: slash
commands arrive form-encoded, and re-serializing a parsed dict changes key
order/escaping and breaks the HMAC. Handlers therefore take the verified raw
body from the host's ``verify_slack_request`` dependency and parse it
themselves instead of declaring parsed body parameters.
"""

from collections.abc import Coroutine, Mapping
from typing import Any

from cognee_connector_slack.connector import CONNECTOR, DESCRIPTOR
from cognee_connectors_core.base import WebhookVerifier
from cognee_connectors_core.errors import WebhookVerificationError

#: The request headers Slack signs with, as the descriptor spells them (lower-case;
#: the verifier matches header names case-insensitively).
SIGNATURE_HEADER: str = DESCRIPTOR.signature_header or "x-slack-signature"
TIMESTAMP_HEADER: str = DESCRIPTOR.signature_timestamp_header or "x-slack-request-timestamp"


def slack_webhook_verifier() -> WebhookVerifier:
    """The ``slack_v0`` verifier ``CONNECTOR`` declares.

    It reads ``SLACK_SIGNING_SECRET`` off the settings object on every call, so
    a rotated secret applies to the next request; an unset secret refuses every
    delivery rather than checking against an empty key. Slack's documented
    five-minute replay window is enforced in both directions.
    """
    verifier = CONNECTOR.webhook_verifier()
    if verifier is None:  # pragma: no cover - DESCRIPTOR declares a signature scheme
        raise RuntimeError("the Slack descriptor declares no webhook signature")
    return verifier


async def verify_slack_request_body(raw_body: bytes, headers: Mapping[str, str]) -> bytes:
    """Return ``raw_body`` once its Slack signature checks out.

    Raises :class:`~cognee_connectors_core.errors.WebhookVerificationError`
    carrying the HTTP status a host should answer with: 401 for a missing or wrong
    signature or timestamp, 413 for a body over the verifier's cap. ``headers``
    may spell the header names in any case.
    """
    return await slack_webhook_verifier().verify(raw_body, headers)


def is_valid_slack_signature(raw_body: bytes, timestamp: str, signature: str) -> bool:
    """Whether ``signature`` is a current, authentic Slack signature for ``raw_body``.

    Pure function of its inputs (plus the clock): the same check as
    :func:`verify_slack_request_body`, as a boolean, for callers that hold the
    two header values rather than a request — ``timestamp`` is
    ``X-Slack-Request-Timestamp`` and ``signature`` is ``X-Slack-Signature``.
    """
    headers: dict[str, str] = {}
    if timestamp is not None:
        headers[TIMESTAMP_HEADER] = str(timestamp)
    if signature is not None:
        headers[SIGNATURE_HEADER] = str(signature)
    try:
        _complete(slack_webhook_verifier().verify(raw_body, headers))
    except WebhookVerificationError:
        return False
    return True


def _complete(coroutine: Coroutine[Any, Any, Any]) -> Any:
    """Finish a coroutine that never suspends, with no event loop involved.

    ``HmacVerifier.verify`` is ``async`` for the contract's sake but awaits
    nothing, so its first step runs it to completion. That is what keeps
    :func:`is_valid_slack_signature` synchronous, as the moved function always
    was, and callable from inside a running event loop; a verifier that did
    suspend would be a bug here, not something to wait on.
    """
    try:
        coroutine.send(None)
    except StopIteration as finished:
        return finished.value
    coroutine.close()
    raise RuntimeError("the Slack webhook verifier suspended; await verify_slack_request_body")
