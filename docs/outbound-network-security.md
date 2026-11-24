# Outbound URL connections

The HTTP crawler (page fetches, robots.txt and Playwright subresources) uses
`PublicHTTPTransport`. Each new TCP connection resolves its host, rejects the
entire answer if any address is non-public, and connects only to a validated
numeric address. HTTP Host and TLS SNI retain the original hostname; connection
pools remain keyed by origin. Redirects, where enabled by a caller, pass through
the same connection guard. Environment HTTP proxies are ignored because their
DNS resolution would happen outside that guard.

Playwright routes HTTP requests through this client, blocks service workers and
closes WebSockets. It requires Playwright with `route_web_socket` (1.48+). The
browser is not a general sandbox for untrusted JavaScript; use network isolation
as an additional boundary. Failed request validation aborts the browser route.

HTTP(S) Git clones use validated addresses in `http.curloptResolve`, preserve TLS
verification and scoped credentials, disable proxies and refuse redirects. This
requires Git 2.37+ with libcurl support for CURLOPT_RESOLVE. The pin applies to
the specified origin; it is not a replacement for process/network isolation
against arbitrary Git helper/alternate transports. Cloud accepts only canonical
HTTPS GitHub/GitLab repository URLs and also needs its egress policy enabled.
SSH repositories retain their existing SDK behavior and are outside this HTTP
policy; Cloud does not admit SSH URLs.

`validate_outbound_url()` now returns the checked addresses. Calling it and then
letting another client independently resolve the name is still unsafe. Use the
pinned transport or pass those addresses to the connection implementation.
Private-network deployments must not weaken this boundary to reach their own
services; use dedicated trusted integration clients for those operations.

Regression tests exercise HTTPX/httpcore connections, TLS hostname preservation,
rebinding between preflight and socket creation, redirects to metadata, mixed
DNS answers, browser subresource rejection, and real Git/libcurl pinning against
a local fixture. Cloud must adopt a release containing these changes before the
hosted service gains the protection; an SDK PR alone does not deploy Cloud.
