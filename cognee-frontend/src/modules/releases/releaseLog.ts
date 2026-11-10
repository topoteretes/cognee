/**
 * The Cloud release log as the backend serves it (CLO-1080): the versions and
 * customer-facing notes of the releases this environment runs, newest first.
 * The backend reads them from the GitHub Releases the release workflow writes
 * (CLO-1081); this module only shapes them for the help menu's "What's new"
 * and the /releases page.
 */

export interface Release {
  version: string;
  releasedAt: string | null;
  summary: string;
  highlights: string[];
  fixes: string[];
  hasNotes: boolean;
}

export interface ReleaseLog {
  /** The image tag this environment runs, e.g. "1.6.2-b45dc59"; null when unknown. */
  runningVersion: string | null;
  /** False when the backend has no GitHub token: the list is then always empty. */
  available: boolean;
  /** True when GitHub could not be reached and this is the last good copy. */
  stale: boolean;
  releases: Release[];
}

const VERSION = /^v?(\d+(?:\.\d+)*)(?:-(\d+(?:\.\d+)*))?/;

/**
 * "1.6.2-b45dc59" → "v1.6.2", "v1.5.4-0.0.5" → "v1.5.4-0.0.5", "prod" → null.
 * An image tag carries the build's commit hash; a release line keeps its
 * numeric suffix. Anything that does not start with a number is not a version.
 */
export function displayVersion(raw: string | null | undefined): string | null {
  const match = VERSION.exec((raw ?? "").trim());
  if (!match) return null;
  return `v${match[1]}${match[2] ? `-${match[2]}` : ""}`;
}

/** "2026-10-01T13:52:19Z" → "Oct 1, 2026". A release is a day, not a minute. */
export function formatReleaseDate(iso: string | null | undefined): string {
  if (!iso) return "—";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "—";
  return date.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric", timeZone: "UTC" });
}

/** The newest release that actually has notes; the menu shows nothing older. */
export function latestWithNotes(releases: Release[]): Release | null {
  return releases.find((release) => release.hasNotes) ?? null;
}

/** What the menu shows for a release: highlights first, fixes to fill up, at most `max`. */
export function whatsNewLines(release: Release, max = 3): string[] {
  return [...release.highlights, ...release.fixes].slice(0, max);
}

function stringList(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((v): v is string => typeof v === "string" && v.length > 0) : [];
}

/** Validates the proxy's answer; throws so the query errors instead of rendering junk. */
export function parseReleaseLog(data: unknown): ReleaseLog {
  if (typeof data !== "object" || data === null || !Array.isArray((data as { releases?: unknown }).releases)) {
    throw new Error("release log: unexpected shape");
  }
  const raw = data as Record<string, unknown>;
  const releases = (raw.releases as unknown[]).flatMap((item): Release[] => {
    if (typeof item !== "object" || item === null) return [];
    const r = item as Record<string, unknown>;
    if (typeof r.version !== "string" || !r.version) return [];
    return [{
      version: r.version,
      releasedAt: typeof r.releasedAt === "string" ? r.releasedAt : null,
      summary: typeof r.summary === "string" ? r.summary : "",
      highlights: stringList(r.highlights),
      fixes: stringList(r.fixes),
      hasNotes: r.hasNotes === true,
    }];
  });
  return {
    runningVersion: typeof raw.runningVersion === "string" ? raw.runningVersion : null,
    available: raw.available === true,
    stale: raw.stale === true,
    releases,
  };
}
