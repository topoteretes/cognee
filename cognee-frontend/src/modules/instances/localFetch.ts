import handleServerErrors from "@/utils/handleServerErrors";

const localApiUrl = process.env.NEXT_PUBLIC_LOCAL_API_URL || "http://localhost:8000";

let apiKey: string | null = process.env.NEXT_PUBLIC_COGWIT_API_KEY || null;

export default async function localFetch(url: URL | RequestInfo, options: RequestInit = {}): Promise<Response> {
  const authHeaders: Record<string, string> = {};
  if (apiKey) {
    authHeaders["X-Api-Key"] = apiKey;
  }

  // The local backend mounts all routes at /api/v1/...
  // Most component paths arrive as "/v1/datasets/" etc., but some (like
  // "/configuration/...") omit the /v1 prefix. Normalize them all to /v1.
  let urlStr = typeof url === "string" ? url : url.toString();
  if (!urlStr.startsWith("/v1/") && !urlStr.startsWith("/v1?")) {
    urlStr = "/v1" + urlStr;
  }
  const fullUrl = localApiUrl + "/api" + urlStr;
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
      return handleServerErrors(response, null, false);
    })
    .catch((error) => {
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
