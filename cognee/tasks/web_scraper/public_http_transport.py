"""HTTP connections pinned to checked addresses, keeping the original Host/SNI.

Validation belongs at the socket boundary: a preflight DNS check followed by
the HTTP client's own lookup is vulnerable to rebinding. The HTTP pool remains
keyed by the original origin, so TLS connections are never shared across names
just because they resolve to the same IP. Redirects and retries create sockets
through the same guard. Environment proxies are deliberately not used: a proxy
would resolve the hostname itself, outside this boundary.
"""

import httpcore
import httpx

from .ssrf_protection import require_http_requests_allowed, validate_outbound_url


class PublicNetworkBackend(httpcore.AsyncNetworkBackend):
    def __init__(self):
        self.backend = httpcore.AnyIOBackend()

    async def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        authority = f"[{host}]" if ":" in host else host
        addresses = await validate_outbound_url(f"https://{authority}:{port}/")
        for index, address in enumerate(addresses):
            try:
                return await self.backend.connect_tcp(
                    str(address),
                    port,
                    timeout=timeout,
                    local_address=local_address,
                    socket_options=socket_options,
                )
            except (httpcore.ConnectError, httpcore.ConnectTimeout):
                if index == len(addresses) - 1:
                    raise

    async def connect_unix_socket(self, *args, **kwargs):
        raise httpcore.ConnectError("Unix sockets are not an outbound URL destination")

    async def sleep(self, seconds):
        await self.backend.sleep(seconds)


class PublicHTTPTransport(httpx.AsyncHTTPTransport):
    def __init__(self):
        super().__init__(trust_env=False)
        # HTTPX does not expose httpcore's network_backend constructor argument.
        # Keep its SSL context, limits and exception mapping; replace only DNS/TCP.
        # The transport integration tests cover this hook on dependency upgrades.
        self._pool._network_backend = PublicNetworkBackend()

    async def handle_async_request(self, request):
        require_http_requests_allowed()  # Also applies to reused connections.
        return await super().handle_async_request(request)
