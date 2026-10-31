/**
 * Configuration the Next server hands to the browser on every request.
 *
 * `NEXT_PUBLIC_*` values are inlined into the client bundle when the app is
 * built, so a published Docker image cannot be pointed at a different backend
 * with `docker run -e ...`: every reader would still see whatever URL was
 * baked in at build time. The server shell therefore renders the live
 * environment into the document, and client readers prefer it over the
 * build-time constant.
 *
 * It travels as `<script type="application/json">` in <head> rather than as an
 * executable script, which removes the question of ordering. An executing
 * snippet has to win a race against Next's async framework chunks, and
 * next/script's "beforeInteractive" does not even try: it queues the URL for
 * Next's own runtime to fetch after the bundle. Inert JSON needs only to be
 * parsed, and <head> is finished before the body root element exists, so the
 * data is in the DOM before React can render the component that reads it.
 */

export const RUNTIME_CONFIG_ELEMENT_ID = "cognee-runtime-config";

export interface RuntimeConfig {
  /** Absolute origin of the cognee backend, without a trailing slash. */
  backendUrl: string | null;
}

/**
 * Callers concatenate paths onto the backend URL ("/api/v1/..."), so a
 * trailing slash would produce "//api" and miss on the backend.
 */
export function stripTrailingSlash(url: string): string {
  return url.replace(/\/+$/, "");
}

/**
 * Normalise a configured backend URL, or return null when nothing is set.
 *
 * Throws when the value is set but unusable. A container started with a broken
 * URL should say so in its logs, rather than quietly falling back to localhost
 * and failing every request for a reason the operator cannot see.
 */
export function normalizeBackendUrl(
  raw: string | undefined | null,
  source: string,
): string | null {
  const value = raw?.trim();
  if (!value) return null;

  // Checked before parsing, because "cognee:8000" (the most likely mistake)
  // is a *valid* URL with the scheme "cognee:", and reporting it as a protocol
  // problem tells the operator nothing about what to type instead.
  if (!/^https?:\/\//i.test(value)) {
    throw new Error(
      `${source} must be an absolute http(s) URL, got "${value}". Expected something like "http://localhost:8000".`,
    );
  }

  try {
    new URL(value);
  } catch {
    throw new Error(`${source} is not a valid URL: "${value}".`);
  }

  return stripTrailingSlash(value);
}

/**
 * Client-side read of the config the server rendered into the document.
 *
 * Never throws: a missing or malformed element just means "not configured",
 * and the caller's own fallback is a better outcome than a blank page.
 */
export function readRuntimeConfig(): Partial<RuntimeConfig> {
  if (typeof document === "undefined") return {};

  const element = document.getElementById(RUNTIME_CONFIG_ELEMENT_ID);
  if (!element?.textContent) return {};

  let parsed: Partial<RuntimeConfig>;
  try {
    parsed = JSON.parse(element.textContent) as Partial<RuntimeConfig>;
  } catch {
    return {};
  }

  // Re-check the URL on the way out of the DOM, with the same rule the server
  // applied on the way in. The client should not trust document content it did
  // not verify itself: this value reaches href attributes, so an unvalidated
  // "javascript:" here would be an XSS sink.
  //
  // The scheme is not carried over from the input, it is picked from the two
  // literals below and the rest is rebuilt from the parsed parts. That leaves
  // no way for the document to choose the scheme, and it is why static
  // analysis can see the result is safe. Query and fragment are dropped; a
  // backend base URL has no use for them. Anything unusable degrades to the
  // caller's own fallback rather than throwing.
  const raw = parsed.backendUrl?.trim();
  if (!raw) return {};

  let url: URL;
  try {
    url = new URL(raw);
  } catch {
    return {};
  }

  if (url.protocol === "https:") {
    return { backendUrl: stripTrailingSlash(`https://${url.host}${url.pathname}`) };
  }
  if (url.protocol === "http:") {
    return { backendUrl: stripTrailingSlash(`http://${url.host}${url.pathname}`) };
  }
  return {};
}

/**
 * Serialise for embedding in the document. Escaped even though the payload is
 * inert JSON: without it a value containing "</script>" would close the tag
 * early and spill the rest into the page as markup.
 */
export function serializeRuntimeConfig(config: RuntimeConfig): string {
  return JSON.stringify(config)
    .replace(/</g, "\\u003c")
    .replace(/>/g, "\\u003e")
    .replace(/\u2028/g, "\\u2028")
    .replace(/\u2029/g, "\\u2029");
}
