// Server-side only. Without this guard, a client component importing this
// module would get process.env.COGNEE_BACKEND_URL replaced with undefined at
// build time and silently fall back to localhost. next/jest maps this to an
// empty mock, so tests are unaffected.
import "server-only";

import { normalizeBackendUrl, type RuntimeConfig } from "./runtimeConfig";

/**
 * Read at request time, which is what lets one published image serve any
 * backend. Deliberately not NEXT_PUBLIC_-prefixed: that prefix means "inline
 * this at build time", the exact behaviour this variable exists to avoid.
 */
const BACKEND_URL_ENV = "COGNEE_BACKEND_URL";

/** Still honoured so builds that baked in a URL keep working. */
const BUILD_TIME_ENV = "NEXT_PUBLIC_LOCAL_API_URL";

const DEFAULT_BACKEND_URL = "http://localhost:8000";

/** The backend this server process should call from route handlers. */
export function getServerBackendUrl(): string {
  return (
    normalizeBackendUrl(process.env.COGNEE_BACKEND_URL, BACKEND_URL_ENV) ??
    normalizeBackendUrl(process.env.NEXT_PUBLIC_LOCAL_API_URL, BUILD_TIME_ENV) ??
    DEFAULT_BACKEND_URL
  );
}

/**
 * The config advertised to the browser.
 *
 * Only COGNEE_BACKEND_URL is published: NEXT_PUBLIC_LOCAL_API_URL is already
 * inlined in the client bundle, and the localhost default is deliberately left
 * out so the browser keeps its own smarter fallback: deriving the backend
 * host from window.location, which also works when the UI is reached from
 * another machine.
 */
export function collectRuntimeConfig(): RuntimeConfig {
  return {
    backendUrl: normalizeBackendUrl(process.env.COGNEE_BACKEND_URL, BACKEND_URL_ENV),
  };
}
