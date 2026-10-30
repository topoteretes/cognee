"""The one place the integrations consult ``ALLOW_HTTP_REQUESTS``.

Every connector talks to its provider over HTTPS, so the flag that disables
cognee's outbound HTTP (honoured by URL ingestion, the crawler and repository
cloning through :mod:`cognee.tasks.web_scraper.ssrf_protection`) has to
disable the connectors too. The check sits where the connections are made:

* :func:`client_session` is how every aiohttp session in
  ``cognee/modules/integrations`` is opened, so each provider call (OAuth
  exchange and refresh, revoke, webhook-driven posts, resource listings,
  syncs) is refused before a socket is opened, whatever path led to it.
* ``track_sync`` checks it once per sync, which also covers the dlt sources
  under ``cognee/tasks/ingestion/connectors`` (``requests``/``httpx``), since
  every sync runs inside it.
* The router checks it on ``authorize``, ``sync`` and the resource picker so
  the caller gets a 503 instead of a background failure or an OAuth round
  trip that cannot complete.
"""

from typing import Any

import aiohttp
from fastapi import status

from cognee.exceptions import CogneeConfigurationError
from cognee.tasks.web_scraper.ssrf_protection import is_http_requests_allowed


class OutboundHttpDisabledError(CogneeConfigurationError):
    """``ALLOW_HTTP_REQUESTS=false`` while an integration needed the network."""

    def __init__(self, provider: str):
        super().__init__(
            message=(
                f"The {provider} integration needs outbound HTTP, "
                "which ALLOW_HTTP_REQUESTS=false disables."
            ),
            name="OutboundHttpDisabledError",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            remediation="Set ALLOW_HTTP_REQUESTS=true (the default) to connect or sync integrations.",
        )


def require_outbound_http(provider: str) -> None:
    """Raise :class:`OutboundHttpDisabledError` when outbound HTTP is switched off."""
    if not is_http_requests_allowed():
        raise OutboundHttpDisabledError(provider)


def client_session(provider: str, **kwargs: Any) -> aiohttp.ClientSession:
    """An ``aiohttp.ClientSession`` for a provider call, or the configuration error.

    ``kwargs`` go to the session unchanged. Looked up on the ``aiohttp`` module
    at call time, so a test that patches ``aiohttp.ClientSession`` still
    intercepts every connector call.
    """
    require_outbound_http(provider)
    return aiohttp.ClientSession(**kwargs)
