"""The one place the integrations consult ``ALLOW_HTTP_REQUESTS``.

Every connector talks to its provider over HTTPS, so a set
``ALLOW_HTTP_REQUESTS`` that disables cognee's outbound HTTP disables the
connectors too. Unset (or blank), connectors work on every surface: unlike URL
ingestion and repository cloning (see
:mod:`cognee.tasks.web_scraper.ssrf_protection`, where an unset value refuses
API requests), a connector only reaches its provider's fixed hosts, so no
caller chooses where the server connects. The check sits where the
connections are made:

* :func:`client_session` is how every aiohttp session in
  ``cognee/modules/integrations`` is opened, so each provider call (OAuth
  exchange and refresh, revoke, webhook-driven posts, resource listings,
  syncs) is refused before a socket is opened, whatever path led to it.
* ``track_sync`` checks it once per integration sync (``ingestion.run_sync``,
  the GitHub repository sync, the Slack history import), so a sync a webhook
  or the resume worker starts is refused before anything is fetched, and the
  refusal is recorded as a failed run. This covers the dlt sources under
  ``cognee/tasks/ingestion/connectors`` only when an integration drives them:
  ``add(source)`` over one of them from the SDK never enters ``track_sync``
  and is not gated here.
* The router checks it on ``authorize``, ``sync`` and the resource picker so
  the caller gets a 503 instead of a background failure or an OAuth round
  trip that cannot complete.
"""

import os
from typing import Any

import aiohttp
from fastapi import status

from cognee.exceptions import CogneeConfigurationError
from cognee.tasks.web_scraper.ssrf_protection import opt_in_setting


class OutboundHttpDisabledError(CogneeConfigurationError):
    """A set ``ALLOW_HTTP_REQUESTS`` disables outbound HTTP while an integration needed it."""

    def __init__(self, provider: str):
        configured = (os.getenv("ALLOW_HTTP_REQUESTS") or "").strip()
        super().__init__(
            message=(
                f"The {provider} integration needs outbound HTTP, which "
                f"ALLOW_HTTP_REQUESTS={configured} disables on this server. Set "
                "ALLOW_HTTP_REQUESTS=true in the server's environment, or unset it, to "
                "connect or sync integrations."
            ),
            name="OutboundHttpDisabledError",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            remediation=(
                "Set ALLOW_HTTP_REQUESTS=true (only true, 1, yes or on enable it) in the "
                "server's environment, or unset it, to connect or sync integrations."
            ),
        )


def require_outbound_http(provider: str) -> None:
    """Raise :class:`OutboundHttpDisabledError` when a set ``ALLOW_HTTP_REQUESTS`` disables it.

    Unset or blank allows it: the API-request default of URL ingestion does not apply
    to connectors, which only call their provider's fixed hosts.
    """
    if opt_in_setting("ALLOW_HTTP_REQUESTS") is False:
        raise OutboundHttpDisabledError(provider)


def client_session(provider: str, **kwargs: Any) -> aiohttp.ClientSession:
    """An ``aiohttp.ClientSession`` for a provider call, or the configuration error.

    ``kwargs`` go to the session unchanged. Looked up on the ``aiohttp`` module
    at call time, so a test that patches ``aiohttp.ClientSession`` still
    intercepts every connector call.
    """
    require_outbound_http(provider)
    return aiohttp.ClientSession(**kwargs)
