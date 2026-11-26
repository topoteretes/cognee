# Outbound URL connections

The HTTP crawler (page fetches, robots.txt and Playwright subresources) uses
`PublicHTTPTransport`. Each new TCP connection resolves its host, rejects the
entire answer if any address is non-public, and connects only to a validated
numeric address. HTTP Host and TLS SNI retain the original hostname; connection
pools remain keyed by origin. Redirects, where enabled by a caller, pass through
the same connection guard. Environment HTTP proxies are ignored because their
DNS resolution would happen outside that guard.

Playwright routes HTTP requests through this client, blocks service workers and
closes WebSockets. Chromium launches behind a dead proxy, with its resolver
disabled and non-proxied WebRTC disabled; loopback is not exempted from the proxy. It requires Playwright with `route_web_socket` (1.48+). The
browser is not a general sandbox for untrusted JavaScript; use network isolation
as an additional boundary. Failed request validation aborts the browser route.

HTTP(S) Git clones use validated addresses in `http.curloptResolve`, preserve TLS
verification and scoped credentials, disable proxies and refuse redirects. Git
2.37+ is checked before HTTP(S) operations so older Git cannot ignore the pin.
GitHub/GitLab URLs are normalized before validation (`www.`, HTTP, and GitLab's
missing `.git` suffix). Renamed/transferred repositories require their canonical
URL; a redirect produces an explicit error. Cache paths include a URL hash;
refresh verifies the stored origin and always fetches from the validated URL.

Credential-helper settings (including host-scoped helpers), extra auth headers,
and CA bundle/path settings are copied from Git configuration. Other global/system
settings, notably `url.*.insteadOf` and proxy settings, are deliberately excluded
because they can replace the validated destination. Existing unhashed cache
directories are not reused; a new clone is created, without deleting the old one.
This requires libcurl support for CURLOPT_RESOLVE. The pin applies to
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

## Explicit self-hosted settings

`DefaultUrlCrawler` and `DefaultCrawlerConfig` accept `proxy_url`, `ca_bundle`,
`browser_route_concurrency`, and `max_browser_response_bytes`. The corresponding process settings are
`WEB_SCRAPER_PROXY_URL`, `WEB_SCRAPER_CA_BUNDLE`, and
`WEB_SCRAPER_MAX_BROWSER_RESPONSE_BYTES`. These settings belong to the operator;
never populate them from an untrusted request. Generic `HTTPS_PROXY` and
`SSL_CERT_FILE` remain ignored. An explicit PEM CA bundle preserves certificate
and hostname verification. A configured proxy becomes the DNS/egress security
boundary and **must itself block private/metadata destinations and rebinding**;
the client's preflight validation alone cannot constrain the proxy's resolver.
Keep this opt-in unset on Cloud.

Browser image/media/font requests are aborted before download. Remaining
responses stream through a separate semaphore, configured with
`browser_route_concurrency` (defaults to crawler concurrency). Fetch slots are
released before Playwright serializes the response. Renders use a dedicated
executor sized by crawler concurrency, keeping the default DNS executor free.
Cancellation stops the render on its own event loop.
The default decompressed response limit is 25 MiB (26214400 bytes), as agreed
in the PR review. An oversized route is aborted; partial content is never returned.
Set a positive `max_browser_response_bytes` to override it. DNS resolution and successive IP attempts share the
existing connect timeout; preflight lookups use the crawler request timeout.
