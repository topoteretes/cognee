import { displayVersion, formatReleaseDate, latestWithNotes, parseReleaseLog, whatsNewLines } from "../releaseLog";
import type { Release } from "../releaseLog";

function release(version: string, extra: Partial<Release> = {}): Release {
  return { version, releasedAt: "2026-10-01T13:52:19Z", summary: "", highlights: [], fixes: [], hasNotes: true, ...extra };
}

describe("displayVersion", () => {
  it("drops an image tag's commit hash and keeps a release line's numeric suffix", () => {
    expect(displayVersion("1.6.2-b45dc59")).toBe("v1.6.2");
    expect(displayVersion("v1.6.2")).toBe("v1.6.2");
    expect(displayVersion("v1.5.4-0.0.5")).toBe("v1.5.4-0.0.5");
    expect(displayVersion("1.5.4-0.0.5-fe.1")).toBe("v1.5.4-0.0.5");
  });

  it("is null for anything that is not a version", () => {
    expect(displayVersion("prod")).toBeNull();
    expect(displayVersion("")).toBeNull();
    expect(displayVersion(null)).toBeNull();
    expect(displayVersion(undefined)).toBeNull();
  });
});

describe("formatReleaseDate", () => {
  it("shows the day, not the minute", () => {
    expect(formatReleaseDate("2026-10-01T13:52:19Z")).toBe("Oct 1, 2026");
    expect(formatReleaseDate("2026-10-01T23:59:00+02:00")).toBe("Oct 1, 2026");
  });

  it("is a dash when the date is missing or broken", () => {
    expect(formatReleaseDate(null)).toBe("—");
    expect(formatReleaseDate("yesterday")).toBe("—");
  });
});

describe("latestWithNotes and whatsNewLines", () => {
  it("skips newer releases that have no notes", () => {
    const withNotes = release("v1.6.2");
    expect(latestWithNotes([release("v1.6.3", { hasNotes: false }), withNotes, release("v1.6.1")])).toBe(withNotes);
    expect(latestWithNotes([release("v1.6.3", { hasNotes: false })])).toBeNull();
    expect(latestWithNotes([])).toBeNull();
  });

  it("lists highlights first, then fixes, three at most", () => {
    const r = release("v1.6.2", { highlights: ["H1", "H2"], fixes: ["F1", "F2"] });
    expect(whatsNewLines(r)).toEqual(["H1", "H2", "F1"]);
    expect(whatsNewLines(r, 10)).toEqual(["H1", "H2", "F1", "F2"]);
    expect(whatsNewLines(release("v1.6.2"))).toEqual([]);
  });
});

describe("parseReleaseLog", () => {
  it("normalises the backend's answer and drops junk entries", () => {
    const parsed = parseReleaseLog({
      runningVersion: "1.6.2-b45dc59",
      available: true,
      stale: false,
      releases: [
        { version: "v1.6.2", releasedAt: "2026-10-01T13:52:19Z", summary: "S", highlights: ["H", 3, ""], fixes: ["F"], hasNotes: true },
        { version: "v1.6.1", releasedAt: null, hasNotes: false },
        { version: "", hasNotes: true },
        "nonsense",
        null,
      ],
    });
    expect(parsed).toEqual({
      runningVersion: "1.6.2-b45dc59",
      available: true,
      stale: false,
      releases: [
        { version: "v1.6.2", releasedAt: "2026-10-01T13:52:19Z", summary: "S", highlights: ["H"], fixes: ["F"], hasNotes: true },
        { version: "v1.6.1", releasedAt: null, summary: "", highlights: [], fixes: [], hasNotes: false },
      ],
    });
  });

  it("defaults the flags and tolerates a missing running version", () => {
    expect(parseReleaseLog({ releases: [] })).toEqual({ runningVersion: null, available: false, stale: false, releases: [] });
  });

  it("throws on anything that is not a log, so the query fails instead of rendering junk", () => {
    expect(() => parseReleaseLog(null)).toThrow();
    expect(() => parseReleaseLog({ error: "management_api_error" })).toThrow();
    expect(() => parseReleaseLog("[]")).toThrow();
  });
});
