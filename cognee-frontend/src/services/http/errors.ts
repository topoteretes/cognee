export type ApiErrorBody = {
  detail?: unknown;
  error?: unknown;
  message?: unknown;
  [key: string]: unknown;
};

export function isApiErrorBody(v: unknown): v is ApiErrorBody {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

// cognee's error handler nests the real message as `detail.message` (e.g.
// `{"detail": {"message": "... [EntityAlreadyExistsError]"}}`) instead of
// putting it directly on `detail`, and the same shape can turn up under
// `error` or `message` from other backends. `String(anObject)` silently
// produces the literal text "[object Object]" with no diagnostic value at
// all, which is exactly what shipped in production here: a UI finally grew
// somewhere to actually display this message, and it read "[object Object]"
// instead of the real failure. JSON.stringify keeps the raw shape visible
// when there is nothing to unwrap.
function extractMessage(value: unknown): string | undefined {
  if (typeof value === "string") return value;
  if (value && typeof value === "object" && !Array.isArray(value)) {
    const candidate = (value as Record<string, unknown>).message;
    if (typeof candidate === "string") return candidate;
    try {
      return JSON.stringify(value);
    } catch {
      return undefined;
    }
  }
  return undefined;
}

export class HttpError extends Error {
  constructor(
    public readonly status: number,
    public readonly statusText: string,
    message: string,
    public readonly body?: ApiErrorBody | string,
    // Raw Retry-After header, carried on the error because the Response is out
    // of reach by the time a caller handles the throw. A 429 handler that only
    // reads the body silently ignores the standard header the server actually
    // sends, and falls back to a fixed delay forever.
    public readonly retryAfter?: string | null,
  ) {
    super(message);
    this.name = "HttpError";
  }
}

export async function toHttpError(res: Response): Promise<never> {
  // Read body once as text on a clone — original stream stays intact so callers
  // that hold a reference to `res` can still read it (e.g. handleServerErrors).
  const text = await res.clone().text().catch(() => "");

  let body: ApiErrorBody | string;
  let message: string;
  try {
    const parsed: unknown = JSON.parse(text);
    if (isApiErrorBody(parsed)) {
      body = parsed;
      message =
        extractMessage(body.detail) ??
        extractMessage(body.error) ??
        extractMessage(body.message) ??
        res.statusText;
    } else {
      body = text;
      message = text || res.statusText;
    }
  } catch {
    body = text;
    message = text || res.statusText;
  }
  throw new HttpError(res.status, res.statusText, message, body, res.headers.get("Retry-After"));
}

// Next.js throws an error with this message when redirect() is called inside
// a Server Action or Route Handler. It must propagate untouched.
const NEXT_REDIRECT_MESSAGE = "NEXT_REDIRECT";

export function normalizeError(e: unknown): never {
  if (e instanceof HttpError) throw e;
  if (e instanceof Error && e.message === NEXT_REDIRECT_MESSAGE) throw e;
  // TimeoutError = our own timeout controller fired (set via abort(new DOMException("TimeoutError"))).
  if (e instanceof Error && e.name === "TimeoutError") throw new Error("Request timed out.");
  // AbortError at this point = callerSignal fired mid-fetch (before our loop guard caught it).
  // Re-throw as-is so React Query / SWR recognize it as a cancellation and suppress error UI.
  if (e instanceof Error && e.name === "AbortError") throw e;
  if (e instanceof TypeError) throw new Error("No connection to the server.");
  if (e instanceof Error) throw e;
  // Exhaustive fallback — wraps any non-Error value (e.g. thrown strings or plain objects).
  throw new Error(String(e));
}
