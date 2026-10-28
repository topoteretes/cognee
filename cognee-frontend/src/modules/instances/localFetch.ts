import handleServerErrors from "@/utils/handleServerErrors";
import { HttpError, toHttpError } from "@/services/http/errors";

const localApiUrl = process.env.NEXT_PUBLIC_LOCAL_API_URL || "http://localhost:8000";

let apiKey: string | null = process.env.NEXT_PUBLIC_COGWIT_API_KEY || null;

export default async function localFetch(url: URL | RequestInfo, options: RequestInit = {}): Promise<Response> {
  const authHeaders: Record<string, string> = {};
  if (apiKey) {
    authHeaders["X-Api-Key"] = apiKey;
  }

  // The local backend mounts API routes under /api/v1.
  // Most component paths arrive as "/v1/datasets/" etc., but some (like
  // "/configuration/...") omit the version prefix. Default those to /v1.
  let urlStr = typeof url === "string" ? url : url.toString();
  if (!urlStr.startsWith("/")) {
    urlStr = `/${urlStr}`;
  }
  const fullUrl = urlStr === "/health" || urlStr.startsWith("/health?")
    ? localApiUrl + urlStr
    : localApiUrl + "/api" + (/^\/v\d+(\/|\?|$)/.test(urlStr) ? urlStr : "/v1" + urlStr);
  const method = options.method || "GET";

  console.log(`[LOCAL-API] → ${method} ${fullUrl}`);

  return global.fetch(
    fullUrl,
    {
      ...options,
      headers: {
        ...options.headers,
        ...authHeaders,
      } as HeadersInit,
      credentials: "include",
    },
  )
    .then((response) => {
      console.log(`[LOCAL-API] ← ${method} ${fullUrl} — ${response.status} ${response.statusText}`);
      // Keep the HTTP status so callers can handle conflicts, rate limits, etc.
      // Authentication failures still follow the local sign-in redirect below.
      if (!response.ok && response.status !== 401 && response.status !== 403) {
        return toHttpError(response);
      }
      return handleServerErrors(response, null, false);
    })
    .catch((error) => {
      // Navigation/query cancellation must retain its identity for callers.
      if (options.signal?.aborted || error?.name === "AbortError") throw error;
      // HTTP failures belong to the caller, which may recover (e.g. a stale cursor).
      if (error instanceof HttpError) throw error;
      console.error(`[LOCAL-API] ✗ ${method} ${fullUrl} — ERROR:`, error);

      // Network errors (no response from server) are TypeErrors from fetch
      if (error instanceof TypeError) {
        return Promise.reject(
          new Error("No connection to the server.")
        );
      }

      // Server errors — pass through the actual message
      const message = error.detail || error.error || error.message || "Something went wrong";
      return Promise.reject(
        new Error(typeof message === "string" ? message : JSON.stringify(message))
      );
    });
}

export const setApiKey = (newApiKey: string) => {
  apiKey = newApiKey;
};
