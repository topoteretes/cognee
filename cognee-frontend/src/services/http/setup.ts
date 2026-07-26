import { http } from "./client";

// Call once at app boot — idempotent (checks prevent double-registration).
// Server: call from src/instrumentation.ts
// Client: call from AppProvider or a "use client" root component
let initialized = false;

export function setupHttpDefaults(): void {
  if (initialized) return;
  initialized = true;

  // Attach a unique correlation ID to every outgoing request.
  // Backends and log aggregators can join on this ID across services.
  http.interceptors.request.use((ctx) => ({
    ...ctx,
    headers: {
      "X-Request-Id": crypto.randomUUID(),
      // Allow callers to override with their own trace ID (e.g. OpenTelemetry).
      ...ctx.headers,
    },
  }));
}
