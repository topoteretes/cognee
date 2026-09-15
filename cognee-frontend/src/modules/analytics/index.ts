/**
 * No-op analytics stubs for the open-source build.
 * The SaaS version uses Segment; this module satisfies all imports
 * without requiring any analytics dependency.
 */

/* eslint-disable @typescript-eslint/no-unused-vars */

// Param shape mirrors the SaaS trackPageEvent input — synced callers derive
// their own types from `Parameters<typeof trackEvent>[0]` and spread it, so
// it must be an object type (not unknown[]). The index signature absorbs
// SaaS-only fields without tracking them individually here.
export interface TrackEventParams {
  pageName?: string;
  eventName?: string;
  additionalProperties?: Record<string, unknown>;
  [key: string]: unknown;
}

export function trackPageView(..._args: unknown[]) {}
export function trackPageEvent(_params: TrackEventParams) {}
export function trackEvent(_params: TrackEventParams) {}
export function identifyUser(..._args: unknown[]) {}
export function getSessionId() { return ""; }
export function getSessionOrigin() { return ""; }
export function refreshSessionActivity() {}
export function getAnonymousId() { return ""; }
export function setAnonymousId(_id: string) {}

// React components used in layouts
export function TrackPageView(_props: {
  page?: string;
  searchProperties?: unknown;
  additionalProperties?: Record<string, unknown>;
}) { return null; }
export function TrackPageEvent(_props: Record<string, unknown>) { return null; }
export function IdentifyUser() { return null; }
