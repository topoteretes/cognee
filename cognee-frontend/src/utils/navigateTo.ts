/**
 * A full page navigation, as its own module so it can be substituted in tests.
 *
 * Assigning to window.location cannot be spied on: jsdom defines it as
 * read-only and throws on any attempt to replace `assign`, which is how the
 * previous approach here ended up asserting nothing. Routing it through one
 * exported function keeps the production call trivial and lets a test see the
 * URL that would have been opened.
 *
 * Deliberately not next/navigation's router: these targets are handled by
 * middleware (the Auth0 SDK mounts /auth/* there), so they need a real
 * navigation rather than a client-side route change.
 */
export function navigateTo(url: string): void {
  window.location.assign(url);
}
