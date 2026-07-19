import { toHttpError, normalizeError, HttpError } from "./errors";

// ─── Types ────────────────────────────────────────────────────────────────────

export type HttpRequestOpts = Omit<RequestInit, "body"> & {
  timeoutMs?: number;
  retries?: number;
  retryOn?: number[];
  backoffMs?: number;
};

export type RequestContext = {
  url: string;
  method: string;
  headers: Record<string, string>;
  body?: BodyInit;
};

export type RequestInterceptor  = (ctx: RequestContext) => RequestContext | Promise<RequestContext>;
export type ResponseInterceptor = (res: Response, ctx: RequestContext) => Response | Promise<Response>;
// Return a Response to recover the request (e.g. token-refresh → retry original).
// Throw to re-reject with a (possibly transformed) error.
// Return void / undefined to pass through to the next interceptor or normalizeError.
export type ErrorInterceptor = (err: unknown, ctx: RequestContext) => Response | void | Promise<Response | void>;

export type HttpLogEvent = {
  url: string;
  method: string;
  status?: number;
  durationMs: number;
  attempt: number;
  error?: unknown;
};

export type HttpLogger = (event: HttpLogEvent) => void;

export type HttpClient = ReturnType<typeof createHttpClient>;

// ─── Shared helpers (stateless, safe to share across client instances) ────────

const DEFAULT_RETRY_ON = [429, 502, 503];
const SAFE_RETRY_ON    = [429]; // non-idempotent methods — only rate-limit retries
const IDEMPOTENT_METHODS = new Set(["GET", "HEAD", "OPTIONS", "PUT", "DELETE"]);

const DEFAULT_TIMEOUTS: Record<string, number> = {
  GET:    10_000,
  POST:   30_000,
  PUT:    30_000,
  PATCH:  30_000,
  DELETE: 15_000,
};

// Native browser types carry their own Content-Type (multipart boundary for
// FormData, octet-stream for Blob/ArrayBuffer, etc.) — let the browser set it.
// Strings are already valid BodyInit and must not be re-serialized.
function needsJsonSerialization(body: unknown): boolean {
  return (
    body !== undefined &&
    typeof body !== "string" &&
    !(body instanceof FormData) &&
    !(body instanceof Blob) &&
    !(body instanceof ArrayBuffer) &&
    !(body instanceof URLSearchParams) &&
    !(typeof ReadableStream !== "undefined" && body instanceof ReadableStream)
  );
}

function retryDelayMs(attempt: number, backoffMs: number, retryAfterHeader: string | null): number {
  if (retryAfterHeader) {
    const seconds = parseFloat(retryAfterHeader);
    if (!Number.isNaN(seconds)) return seconds * 1000;
    const date = Date.parse(retryAfterHeader);
    if (!Number.isNaN(date)) return Math.max(0, date - Date.now());
  }
  // Full-jitter: spread retries across [50%, 100%] of the exponential window
  // to avoid thundering herd when many clients receive the same 503 simultaneously.
  return backoffMs * Math.pow(2, attempt) * (0.5 + Math.random() * 0.5);
}

// Edge Runtime (Cloudflare Workers) does not expose `performance` — fall back to Date.now().
const now = (): number =>
  typeof performance !== "undefined" ? performance.now() : Date.now();

function headersToRecord(headers: HeadersInit | undefined): Record<string, string> {
  if (!headers) return {};
  if (headers instanceof Headers) {
    const out: Record<string, string> = {};
    headers.forEach((v, k) => { out[k] = v; });
    return out;
  }
  if (Array.isArray(headers)) return Object.fromEntries(headers);
  return headers as Record<string, string>;
}

export function makeRegistry<T>() {
  const map = new Map<number, T>();
  let seq = 0;
  return {
    // Returns an ID — call eject(id) in cleanup to avoid leaks (e.g. in React useEffect).
    use:    (fn: T): number        => { const id = seq++; map.set(id, fn); return id; },
    eject:  (id: number): void     => { map.delete(id); },
    clear:  (): void               => { map.clear(); },
    values: (): IterableIterator<T> => map.values(),
  };
}

// Without a validator this is a trust-the-API cast — T is not verified at runtime.
// Pass a validator (e.g. Zod schema's `.parse`) to get actual runtime safety.
async function parseJson<T>(res: Response, validate?: (data: unknown) => T): Promise<T> {
  const data: unknown = await res.json();
  return validate ? validate(data) : (data as T);
}

// ─── Factory ──────────────────────────────────────────────────────────────────

export function createHttpClient() {
  const reqRegistry = makeRegistry<RequestInterceptor>();
  const resRegistry = makeRegistry<ResponseInterceptor>();
  const errRegistry = makeRegistry<ErrorInterceptor>();

  let activeLogger: HttpLogger | null = null;

  function emitLog(event: HttpLogEvent): void {
    try { activeLogger?.(event); } catch { /* logger must never crash the request */ }
  }

  async function httpRequest(
    url: string,
    method: string,
    body?: unknown,
    opts: HttpRequestOpts = {},
  ): Promise<Response> {
    const normalizedMethod = method.toUpperCase();
    const {
      timeoutMs = DEFAULT_TIMEOUTS[normalizedMethod] ?? 30_000,
      retries = 0,
      retryOn = IDEMPOTENT_METHODS.has(normalizedMethod) ? DEFAULT_RETRY_ON : SAFE_RETRY_ON,
      backoffMs = 300,
      signal: callerSignal,
      ...fetchOpts
    } = opts;

    const serializedBody: BodyInit | undefined = body !== undefined
      ? (needsJsonSerialization(body) ? JSON.stringify(body) : (body as BodyInit))
      : undefined;

    const contentTypeHeader: Record<string, string> = needsJsonSerialization(body)
      ? { "Content-Type": "application/json" }
      : {};

    let ctx: RequestContext = {
      url,
      method: normalizedMethod,
      headers: { ...contentTypeHeader, ...headersToRecord(fetchOpts.headers) },
      ...(serializedBody !== undefined ? { body: serializedBody } : {}),
    };

    for (const interceptor of reqRegistry.values()) {
      ctx = await interceptor(ctx);
    }

    let attempt = 0;
    const startTime = now();

    while (attempt <= retries) {
      // Caller cancelled — not a system error. Throw directly, bypassing catch/emitLog/interceptors.
      // DOMException("Aborted", "AbortError") matches what fetch itself throws, so React Query
      // and other libraries correctly identify it as a cancellation and suppress error UI.
      if (callerSignal?.aborted) throw new DOMException("Aborted", "AbortError");

      const timeoutController = new AbortController();
      const timer = setTimeout(
        () => timeoutController.abort(new DOMException("Timed out", "TimeoutError")),
        timeoutMs,
      );

      const signal =
        callerSignal && typeof AbortSignal.any === "function"
          ? AbortSignal.any([timeoutController.signal, callerSignal])
          : callerSignal ?? timeoutController.signal;

      try {
        let res = await fetch(ctx.url, {
          cache:       fetchOpts.cache,
          credentials: fetchOpts.credentials,
          integrity:   fetchOpts.integrity,
          keepalive:   fetchOpts.keepalive,
          mode:        fetchOpts.mode,
          redirect:    fetchOpts.redirect,
          referrer:    fetchOpts.referrer,
          referrerPolicy: fetchOpts.referrerPolicy,
          method:      ctx.method,
          signal,
          headers:     ctx.headers,
          ...(ctx.body !== undefined ? { body: ctx.body } : {}),
        });

        if (!res.ok) {
          if (attempt < retries && retryOn.includes(res.status)) {
            const delay = retryDelayMs(attempt, backoffMs, res.headers.get("Retry-After"));
            attempt++;
            await new Promise((resolve) => setTimeout(resolve, delay));
            continue;
          }
          await toHttpError(res);
        }

        for (const interceptor of resRegistry.values()) {
          res = await interceptor(res, ctx);
        }

        emitLog({ url: ctx.url, method: ctx.method, status: res.status, durationMs: now() - startTime, attempt });
        return res;
      } catch (e) {
        const isNetworkDrop = e instanceof TypeError;
        if (attempt < retries && isNetworkDrop) {
          const delay = retryDelayMs(attempt, backoffMs, null);
          attempt++;
          await new Promise((resolve) => setTimeout(resolve, delay));
          continue;
        }

        emitLog({ url: ctx.url, method: ctx.method, durationMs: now() - startTime, attempt, error: e });
        for (const interceptor of errRegistry.values()) {
          const result = await interceptor(e, ctx);
          if (result instanceof Response) {
            emitLog({ url: ctx.url, method: ctx.method, status: result.status, durationMs: now() - startTime, attempt });
            return result;
          }
        }

        throw normalizeError(e);
      } finally {
        clearTimeout(timer);
      }
    }

    // The loop always exits via return or throw inside the try/catch.
    // This throw exists only to satisfy TypeScript's control-flow analysis.
    /* istanbul ignore next */
    throw new HttpError(0, "Exhausted", `Exhausted ${retries} retries for ${normalizedMethod} ${url}`);
  }

  return {
    get:    (url: string, opts?: HttpRequestOpts) =>
      httpRequest(url, "GET", undefined, opts),
    post:   (url: string, body?: unknown, opts?: HttpRequestOpts) =>
      httpRequest(url, "POST", body, opts),
    put:    (url: string, body?: unknown, opts?: HttpRequestOpts) =>
      httpRequest(url, "PUT", body, opts),
    patch:  (url: string, body?: unknown, opts?: HttpRequestOpts) =>
      httpRequest(url, "PATCH", body, opts),
    // body on DELETE is intentionally omitted — AWS ALB, Cloudflare, and many
    // reverse proxies silently strip it. Use a path param or POST+action instead.
    delete: (url: string, opts?: HttpRequestOpts) =>
      httpRequest(url, "DELETE", undefined, opts),

    getJson:   <T>(url: string, opts?: HttpRequestOpts & { validate?: (d: unknown) => T }) => {
      const { validate, ...reqOpts } = opts ?? {};
      return httpRequest(url, "GET", undefined, reqOpts).then((r) => parseJson<T>(r, validate));
    },
    postJson:  <T>(url: string, body?: unknown, opts?: HttpRequestOpts & { validate?: (d: unknown) => T }) => {
      const { validate, ...reqOpts } = opts ?? {};
      return httpRequest(url, "POST", body, reqOpts).then((r) => parseJson<T>(r, validate));
    },
    putJson:   <T>(url: string, body?: unknown, opts?: HttpRequestOpts & { validate?: (d: unknown) => T }) => {
      const { validate, ...reqOpts } = opts ?? {};
      return httpRequest(url, "PUT", body, reqOpts).then((r) => parseJson<T>(r, validate));
    },
    patchJson: <T>(url: string, body?: unknown, opts?: HttpRequestOpts & { validate?: (d: unknown) => T }) => {
      const { validate, ...reqOpts } = opts ?? {};
      return httpRequest(url, "PATCH", body, reqOpts).then((r) => parseJson<T>(r, validate));
    },

    request: (url: string, init: RequestInit & { timeoutMs?: number } = {}): Promise<Response> => {
      const { body, method = "GET", timeoutMs, ...rest } = init;
      return httpRequest(url, method, body ?? undefined, { ...rest, timeoutMs });
    },

    parseJson,

    interceptors: {
      request:  { use: reqRegistry.use,  eject: reqRegistry.eject,  clear: reqRegistry.clear },
      response: { use: resRegistry.use,  eject: resRegistry.eject,  clear: resRegistry.clear },
      error:    { use: errRegistry.use,  eject: errRegistry.eject,  clear: errRegistry.clear },
    },

    setLogger: (logger: HttpLogger) => { activeLogger = logger; },
  };
}

// ─── Default singleton (for generic use and backwards compat) ─────────────────

export const http = createHttpClient();

export { HttpError };
