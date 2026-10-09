"""Server-Side Request Forgery (SSRF) protection for outbound URL fetching.

Cognee fetches user-supplied ``http(s)`` URLs server-side during ingestion
(``add()`` -> ``save_data_item_to_storage``) and web scraping. Without
validation, a caller can point Cognee at internal-only addresses -- loopback,
private ranges, link-local, or cloud metadata endpoints such as
``169.254.169.254`` -- and exfiltrate the response through the knowledge graph.
This is a classic SSRF (CWE-918).

This module centralises the outbound-URL policy so every fetch path shares the
same guard:

* ``ALLOW_HTTP_REQUESTS`` (default ``true``) gates outbound HTTP(S) fetching.
  Setting it to ``false`` disables remote URL ingestion entirely. Previously
  this variable was documented but never enforced.
* Only the ``http`` and ``https`` schemes are permitted.
* The hostname is resolved and *every* resolved address is checked. A request
  is rejected if any address is loopback, private, link-local, reserved,
  multicast or unspecified. Checking the resolved IPs (rather than the literal
  string) also blocks IP-literal bypasses (``http://0x7f.1``, ``http://[::1]``)
  and hostnames that resolve to internal addresses.
"""

import asyncio
import ipaddress
import os
import socket
from urllib.parse import urlparse

from fastapi import status

from cognee.exceptions import CogneeValidationError
from cognee.shared.logging_utils import get_logger

logger = get_logger()

ALLOWED_SCHEMES = frozenset({"http", "https"})
_FALSEY = frozenset({"false", "0", "no", "off"})


class SSRFProtectionError(CogneeValidationError):
    """Raised when an outbound URL is disallowed or targets an internal address."""

    def __init__(
        self,
        message: str = "The requested URL is not allowed.",
        name: str = "SSRFProtectionError",
        status_code=status.HTTP_403_FORBIDDEN,
    ):
        super().__init__(message, name, status_code)


def is_http_requests_allowed() -> bool:
    """Return whether outbound HTTP(S) fetching is enabled via ``ALLOW_HTTP_REQUESTS``."""
    return os.getenv("ALLOW_HTTP_REQUESTS", "true").strip().lower() not in _FALSEY


def _is_blocked_ip(ip: ipaddress._BaseAddress) -> bool:
    """Return True if ``ip`` is not a routable, public address."""
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
        # IPv6 site-local (deprecated) and IPv4-mapped IPv6 that decodes to a blocked v4.
        or getattr(ip, "is_site_local", False)
    )


async def _resolve_host_ips(host: str) -> list:
    """Resolve ``host`` to all of its IP addresses without blocking the event loop."""
    loop = asyncio.get_event_loop()
    infos = await loop.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)

    ips = []
    for info in infos:
        address = info[4][0]
        # Strip any IPv6 zone/scope id (e.g. "fe80::1%eth0").
        address = address.split("%", 1)[0]
        try:
            ips.append(ipaddress.ip_address(address))
        except ValueError:
            continue
    return ips


async def validate_outbound_url(url: str) -> None:
    """Validate a user-supplied URL before Cognee fetches it server-side.

    Raises:
        SSRFProtectionError: if outbound HTTP is disabled, the scheme is not
            ``http``/``https``, the URL has no host, the host cannot be
            resolved, or the host resolves to an internal / reserved address.
    """
    if not is_http_requests_allowed():
        raise SSRFProtectionError(
            "Outbound HTTP requests are disabled (ALLOW_HTTP_REQUESTS=false)."
        )

    parsed = urlparse(url)
    scheme = (parsed.scheme or "").lower()
    if scheme not in ALLOWED_SCHEMES:
        raise SSRFProtectionError(
            f"URL scheme '{scheme or '(none)'}' is not allowed; only http/https are permitted."
        )

    host = parsed.hostname
    if not host:
        raise SSRFProtectionError("URL does not contain a valid host.")

    # If the host is an IP literal, validate it directly. This covers cases like
    # http://169.254.169.254 and http://[::1] without a DNS lookup.
    try:
        literal_ip = ipaddress.ip_address(host)
    except ValueError:
        literal_ip = None

    if literal_ip is not None:
        if _is_blocked_ip(literal_ip):
            raise SSRFProtectionError(
                f"URL host '{host}' points to a non-public address and is blocked."
            )
        return

    try:
        resolved_ips = await _resolve_host_ips(host)
    except socket.gaierror as error:
        raise SSRFProtectionError(f"Could not resolve URL host '{host}'.") from error

    if not resolved_ips:
        raise SSRFProtectionError(f"Could not resolve URL host '{host}'.")

    for ip in resolved_ips:
        if _is_blocked_ip(ip):
            raise SSRFProtectionError(
                f"URL host '{host}' resolves to a non-public address ({ip}) and is blocked."
            )
