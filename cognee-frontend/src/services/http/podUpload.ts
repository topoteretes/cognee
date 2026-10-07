import { HttpError, type ApiErrorBody } from "./errors";
import type { UploadRequestOpts } from "@/modules/instances/types";

// Retry budget for a batch rejected because other work is mid-flight. The pod
// answers 429 + Retry-After for that case (CLO-555): the balance is fine and
// the block clears on its own, so retrying is correct where a 402 would not be.
const MAX_RETRIES_429 = 3;
const DEFAULT_RETRY_AFTER_MS = 5_000;

function parseBody(text: string): ApiErrorBody | string {
  try {
    const parsed: unknown = JSON.parse(text);
    if (parsed && typeof parsed === "object") return parsed as ApiErrorBody;
  } catch {
    /* not JSON — fall through to the raw text */
  }
  return text;
}

function errorMessage(body: ApiErrorBody | string, statusText: string): string {
  if (typeof body === "string") return body || statusText;
  return String(body.detail ?? body.error ?? body.message ?? statusText);
}

function retryAfterMs(header: string | null): number {
  if (!header) return DEFAULT_RETRY_AFTER_MS;
  // Clamp: a negative Retry-After would become a negative setTimeout delay,
  // which browsers treat as 0 — turning "wait" into an immediate retry loop
  // against a server that just asked us to back off. The HTTP-date branch
  // below already clamps; this one did not.
  const seconds = parseFloat(header);
  if (!Number.isNaN(seconds)) return Math.max(0, seconds * 1000);
  const date = Date.parse(header);
  if (!Number.isNaN(date)) return Math.max(0, date - Date.now());
  return DEFAULT_RETRY_AFTER_MS;
}

/**
 * One multipart POST over XMLHttpRequest, so the caller gets upload-progress
 * events — the single reason this exists instead of using the shared fetch
 * client. Mirrors that client's error contract (HttpError on non-2xx, the
 * "Request timed out." message on timeout) so callers and the 402 credits
 * bridge cannot tell the two paths apart.
 */
export default function podUpload(
  url: string,
  apiKey: string,
  body: FormData,
  { timeoutMs, signal, onProgress }: UploadRequestOpts = {},
): Promise<Response> {
  return new Promise<Response>((resolve, reject) => {
    if (signal?.aborted) {
      reject(new DOMException("Aborted", "AbortError"));
      return;
    }

    const xhr = new XMLHttpRequest();
    xhr.open("POST", url, true);
    xhr.setRequestHeader("X-Api-Key", apiKey);
    xhr.responseType = "text";
    if (timeoutMs) xhr.timeout = timeoutMs;

    const onAbort = (): void => xhr.abort();
    signal?.addEventListener("abort", onAbort, { once: true });
    const cleanup = (): void => signal?.removeEventListener("abort", onAbort);

    if (onProgress) {
      // Only `lengthComputable` events carry a real total; FormData with File
      // parts always is, but guard anyway so a browser that can't measure
      // doesn't report a nonsense denominator.
      xhr.upload.onprogress = (event: ProgressEvent): void => {
        if (event.lengthComputable) onProgress(event.loaded, event.total);
      };
    }

    xhr.onload = (): void => {
      cleanup();
      const text = xhr.responseText ?? "";
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(new Response(text, { status: xhr.status, statusText: xhr.statusText }));
        return;
      }
      const parsed = parseBody(text);
      reject(
        new HttpError(
          xhr.status,
          xhr.statusText,
          errorMessage(parsed, xhr.statusText),
          parsed,
          xhr.getResponseHeader("Retry-After"),
        ),
      );
    };

    // Network-level failure: no status, no body. Matches what the fetch client
    // surfaces for the same condition.
    xhr.onerror = (): void => {
      cleanup();
      reject(new Error("Network request failed."));
    };

    xhr.ontimeout = (): void => {
      cleanup();
      reject(new Error("Request timed out."));
    };

    xhr.onabort = (): void => {
      cleanup();
      reject(new DOMException("Aborted", "AbortError"));
    };

    xhr.send(body);
  });
}

/** podUpload plus the 429/Retry-After wait the shared client applies to POSTs. */
export async function podUploadWithRetry(
  url: string,
  apiKey: string,
  makeBody: () => FormData,
  opts: UploadRequestOpts = {},
): Promise<Response> {
  let attempt = 0;
  for (;;) {
    try {
      // A FormData instance is consumed by send(); rebuild it per attempt.
      return await podUpload(url, apiKey, makeBody(), opts);
    } catch (error) {
      const isThrottled = error instanceof HttpError && error.status === 429;
      if (!isThrottled || attempt >= MAX_RETRIES_429) throw error;
      // Retry-After (the standard header, and what the shared client reads) wins
      // over the body field: the pod sends both, and honouring only the body
      // meant every wait was the 5s default no matter what the server asked for.
      const fromBody =
        typeof error.body === "object" && error.body !== null && "retry_after_seconds" in error.body
          ? String((error.body as Record<string, unknown>).retry_after_seconds)
          : null;
      await new Promise((r) => setTimeout(r, retryAfterMs(error.retryAfter ?? fromBody)));
      attempt += 1;
    }
  }
}
