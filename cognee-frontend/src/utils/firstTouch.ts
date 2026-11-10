// The campaign parameters captured once, on a visitor's first landing, and
// carried in the cognee_first_touch cookie.
//
// Shared rather than duplicated because the two ends have to agree exactly:
// middleware.ts is the only writer, trackSignup.ts is the only reader, and the
// reader treats this as an allowlist. A key added to one copy and not the other
// silently drops that attribution with a green build.
export const FIRST_TOUCH_KEYS = [
  "utm_source",
  "utm_medium",
  "utm_campaign",
  "utm_term",
  "utm_content",
  "gclid",
] as const;

export type FirstTouchKey = (typeof FIRST_TOUCH_KEYS)[number];
