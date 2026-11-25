/**
 * Whether a feature still in rollout is on for this host: on everywhere except
 * the released hosts listed, so dev, previews and local dev get it first and
 * releasing it is deleting a host from the list. Anything that is not a plain
 * host (null, a list, user info, a path) counts as a released one, so a
 * feature never shows anywhere it was not meant to. The port, the case and a
 * trailing dot are ignored: the server reads the host from a header that
 * carries the port, the browser's hostname does not.
 */
export function isRolloutEnabled(releasedHosts: ReadonlySet<string>, host: string | null): boolean {
  const hostname = hostnameOf(host);
  return hostname !== null && !releasedHosts.has(hostname);
}

const NOT_A_PLAIN_HOST = /[,@/\s]/;

function hostnameOf(host: string | null): string | null {
  if (!host || NOT_A_PLAIN_HOST.test(host)) return null;
  try {
    return new URL(`http://${host}`).hostname.replace(/\.$/, "");
  } catch {
    return null;
  }
}
