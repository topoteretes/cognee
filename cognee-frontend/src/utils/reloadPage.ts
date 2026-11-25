/**
 * Reloads the current page.
 *
 * A one-line seam around window.location.reload, because jsdom refuses to let
 * a test redefine window.location, and a reload is a real behaviour worth
 * asserting: withdrawing session-recording consent depends on it, since the
 * Contentsquare tag cannot be unloaded any other way.
 */
export default function reloadPage(): void {
  window.location.reload();
}
