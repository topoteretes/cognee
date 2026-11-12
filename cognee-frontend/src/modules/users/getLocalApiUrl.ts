import { readRuntimeConfig, stripTrailingSlash } from "@/modules/config/runtimeConfig";

const DEFAULT_LOCAL_API_PORT = "8000";

export function getLocalApiUrl(): string {
  // This Next server forwards /api/v1 to the backend (COGNEE_DEV_API_PROXY in
  // next.config.ts), so the browser calls its own origin: through a tunnel that is the
  // only host it can reach, and the OAuth install-nonce cookie has to be set on the host
  // the provider's callback arrives on.
  if (process.env.NEXT_PUBLIC_API_SAME_ORIGIN === "true" && typeof window !== "undefined") {
    return window.location.origin;
  }

  // Runtime config wins: it is the only source a published Docker image can
  // change, because NEXT_PUBLIC_* is frozen into the bundle at build time.
  const runtimeUrl = readRuntimeConfig().backendUrl;
  if (runtimeUrl) return stripTrailingSlash(runtimeUrl);

  const configuredUrl = process.env.NEXT_PUBLIC_LOCAL_API_URL;
  if (configuredUrl) return stripTrailingSlash(configuredUrl);

  if (typeof window !== "undefined") {
    return `${window.location.protocol}//${window.location.hostname}:${DEFAULT_LOCAL_API_PORT}`;
  }

  return `http://localhost:${DEFAULT_LOCAL_API_PORT}`;
}
