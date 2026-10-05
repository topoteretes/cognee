import { readRuntimeConfig, stripTrailingSlash } from "@/modules/config/runtimeConfig";

const DEFAULT_LOCAL_API_PORT = "8000";

export function getLocalApiUrl(): string {
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
