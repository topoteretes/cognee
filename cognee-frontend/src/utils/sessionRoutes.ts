// Route lists shared by UserProvider (query gating) and TenantProvider
// (init skip + render gate). Kept in one place so the two providers can never
// disagree about which pages have no session.

// Auth pages have no session yet by definition — /api/me and the tenant
// queries never run there.
export const AUTH_ROUTE_PATHS = [
  "/sign-in",
  "/sign-up",
  "/forgot-password",
  "/local-login",
  "/verify-email",
  "/email-verified",
];

// Public, no-session-required pages — /api/me legitimately errors for a
// visitor with no Auth0 session at all, so user state never resolves there.
export const NO_SESSION_PATHS = ["/waitlist"];

// Boundary-aware prefix match, so "/sign-in" does not also match a
// hypothetical "/sign-in-legacy" (same guard middleware.ts uses).
function matchesPathPrefix(pathname: string, prefix: string): boolean {
  return pathname === prefix || pathname.startsWith(`${prefix}/`);
}

export function isAuthRoutePath(pathname: string): boolean {
  return AUTH_ROUTE_PATHS.some((p) => matchesPathPrefix(pathname, p));
}

export function isSessionlessPath(pathname: string): boolean {
  return (
    isAuthRoutePath(pathname) ||
    NO_SESSION_PATHS.some((p) => matchesPathPrefix(pathname, p))
  );
}
