const DEFAULT_LOCAL_API_PORT = "8000";

export function getLocalApiUrl(): string {
  const configuredUrl = process.env.NEXT_PUBLIC_LOCAL_API_URL;
  if (configuredUrl) return configuredUrl;

  if (typeof window !== "undefined") {
    return `${window.location.protocol}//${window.location.hostname}:${DEFAULT_LOCAL_API_PORT}`;
  }

  return `http://localhost:${DEFAULT_LOCAL_API_PORT}`;
}
