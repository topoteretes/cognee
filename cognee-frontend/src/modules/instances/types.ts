export interface UploadRequestOpts {
  timeoutMs?: number;
  signal?: AbortSignal;
  // Cumulative bytes of THIS request's body that have reached the network.
  onProgress?: (bytesSent: number, bytesTotal: number) => void;
}

export interface CogneeInstance {
  name: string;
  // Stable identity for the tenant/pod this instance talks to — `name` is a
  // fixed literal ("CloudCognee"/"LocalCognee") shared by every instance, so
  // it can't tell two tenants apart. Callers that cache by cogniInstance
  // (e.g. React Query keys) must key on this instead, or a switch between
  // tenants can keep serving another tenant's cached response (COG-6233).
  instanceId: string;
  // Wraps the shared http client (see @/services/http/client) — accepts its
  // timeoutMs override in addition to standard RequestInit, since that's the
  // only way to raise a call above the client's per-method default timeout
  // (POST default is 30s; callers like rememberData need much longer).
  fetch: (input: RequestInfo | URL, init?: RequestInit & { timeoutMs?: number }) => Promise<Response>;
  // Multipart POST with upload-progress events. Separate from `fetch` because
  // fetch() cannot report request-body progress at all — the browser only
  // exposes it through XMLHttpRequest's upload events. Optional so test doubles
  // and non-pod instances don't have to implement it; callers fall back to
  // `fetch` and report progress at batch granularity instead (see rememberData).
  //
  // Takes a FormData *factory*, not an instance: a body is consumed by send(),
  // so a retried attempt (429/Retry-After) needs a freshly built one.
  upload?: (path: string, makeBody: () => FormData, opts?: UploadRequestOpts) => Promise<Response>;
}
