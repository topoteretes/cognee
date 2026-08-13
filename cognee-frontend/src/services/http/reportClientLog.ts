import { http } from "./client";

export type ClientLogLevel = "log" | "warn" | "error";

// Client-side code runs entirely in the browser, so its console output never
// reaches Vercel's server logs. This forwards a structured line to the
// existing /api/log route (which just re-emits it via console on the
// server) — pulled out here so every call site (pod requests, workspace
// creation, ...) shares one implementation instead of duplicating the POST.
export function reportClientLog(
  tag: string,
  level: ClientLogLevel,
  message: string,
  extra?: { url?: string; method?: string },
): void {
  http
    .post("/api/log", { level, tag, message, ...extra })
    .catch(() => { /* best-effort — logging must never break the caller's flow */ });
}
