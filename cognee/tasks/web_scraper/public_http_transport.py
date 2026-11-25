"""HTTP connections pinned to checked addresses, keeping the original Host/SNI.

Validation belongs at the socket boundary: a preflight DNS check followed by
the HTTP client's own lookup is vulnerable to rebinding. The HTTP pool remains
keyed by the original origin, so TLS connections are never shared across names
just because they resolve to the same IP. Redirects and retries create sockets
through the same guard. Environment proxies are deliberately not used: a proxy
would resolve the hostname itself, outside this boundary.
"""

import asyncio
import ssl

import httpcore
import httpx

from .ssrf_protection import require_http_requests_allowed, validate_outbound_url


class PublicNetworkBackend(httpcore.AsyncNetworkBackend):
    def __init__(self):
        self.backend = httpcore.AnyIOBackend()

    async def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        loop = asyncio.get_running_loop()
        deadline = None if timeout is None else loop.time() + timeout

        def remaining():
            return None if deadline is None else max(0.0, deadline - loop.time())

        authority = f"[{host}]" if ":" in host else host
        try:
            addresses = await asyncio.wait_for(
                validate_outbound_url(f"https://{authority}:{port}/"), remaining()
            )
            for index, address in enumerate(addresses):
                try:
                    return await asyncio.wait_for(
                        self.backend.connect_tcp(
                            str(address),
                            port,
                            timeout=remaining(),
                            local_address=local_address,
                            socket_options=socket_options,
                        ),
                        remaining(),
                    )
                except (httpcore.ConnectError, httpcore.ConnectTimeout):
                    if index == len(addresses) - 1:
                        raise
        except asyncio.TimeoutError as error:
            raise httpcore.ConnectTimeout("DNS/connect deadline exceeded") from error

    async def connect_unix_socket(self, *args, **kwargs):
        raise httpcore.ConnectError("Unix sockets are not an outbound URL destination")

    async def sleep(self, seconds):
        await self.backend.sleep(seconds)


class PublicHTTPTransport(httpx.AsyncHTTPTransport):
    def __init__(self, *, proxy_url: str | None = None, ca_bundle: str | None = None):
        context = ssl.create_default_context(cafile=ca_bundle) if ca_bundle else True
        super().__init__(trust_env=False, proxy=proxy_url, verify=context)
        self.proxy_url = proxy_url
        # HTTPX does not expose httpcore's network_backend constructor argument.
        # Keep its SSL context, limits and exception mapping; replace only DNS/TCP.
        # The transport integration tests cover this hook on dependency upgrades.
        if proxy_url is None:
            self._pool._network_backend = PublicNetworkBackend()

    async def handle_async_request(self, request):
        require_http_requests_allowed()  # Also applies to reused connections.
        if self.proxy_url is not None:
            # The operator-selected proxy is the DNS/egress enforcement boundary.
            # Still reject literal/private destinations visible to this process.
            try:
                await asyncio.wait_for(
                    validate_outbound_url(str(request.url)),
                    request.extensions.get("timeout", {}).get("connect"),
                )
            except asyncio.TimeoutError as error:
                raise httpx.ConnectTimeout("DNS deadline exceeded", request=request) from error
        return await super().handle_async_request(request)
