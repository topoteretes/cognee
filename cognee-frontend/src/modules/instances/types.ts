export interface CogneeInstance {
  name: string;
  // Wraps the shared http client (see @/services/http/client) — accepts its
  // timeoutMs override in addition to standard RequestInit, since that's the
  // only way to raise a call above the client's per-method default timeout
  // (POST default is 30s; callers like rememberData need much longer).
  fetch: (input: RequestInfo | URL, init?: RequestInit & { timeoutMs?: number }) => Promise<Response>;
}
