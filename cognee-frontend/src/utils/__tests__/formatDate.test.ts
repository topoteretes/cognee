import { formatClockTime } from "../formatDate";

// TZ is pinned to Europe/Berlin in jest.config.ts (CEST, UTC+2 in September).
describe("formatClockTime", () => {
  const NOW = new Date("2026-09-30T11:26:00.000Z"); // 1:26 PM local

  it("shows time only for a timestamp from today", () => {
    expect(formatClockTime("2026-09-30T11:27:00.000Z", NOW)).toBe("01:27 PM");
  });

  it("shows month, day and time for a timestamp from an earlier day", () => {
    expect(formatClockTime("2026-09-28T13:27:00.000Z", NOW)).toBe("Sep 28, 03:27 PM");
  });

  it("uses the same hour style (12-hour, zero-padded AM/PM) for both the today and earlier-day case", () => {
    // This is the bug this formatter replaces: "First connected" and a
    // session's "Started" used to disagree — one 24-hour, one 12-hour — for
    // what could be the same instant.
    expect(formatClockTime("2026-09-30T13:00:00.000Z", NOW)).toBe("03:00 PM");
    expect(formatClockTime("2026-09-28T13:00:00.000Z", NOW)).toContain("03:00 PM");
  });

  it("returns null for a null input", () => {
    expect(formatClockTime(null, NOW)).toBeNull();
  });

  it("returns null for an unparseable input, rather than rendering Invalid Date", () => {
    expect(formatClockTime("not-a-date", NOW)).toBeNull();
  });
});
