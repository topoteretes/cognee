import { acquireCrossTabSurveyLock } from "../surveyCrossTabLock";

describe("acquireCrossTabSurveyLock", () => {
  let nowSpy: jest.SpyInstance;
  let now = 0;

  beforeEach(() => {
    localStorage.clear();
    now = 1_000_000;
    nowSpy = jest.spyOn(Date, "now").mockImplementation(() => now);
  });

  afterEach(() => {
    nowSpy.mockRestore();
  });

  it("grants the lock when no other tab holds it", () => {
    expect(acquireCrossTabSurveyLock("nps_quarterly")).toBe(true);
  });

  it("denies a second acquisition within the TTL window", () => {
    acquireCrossTabSurveyLock("nps_quarterly");
    now += 1000;

    expect(acquireCrossTabSurveyLock("nps_quarterly")).toBe(false);
  });

  it("grants the lock again once the TTL window has elapsed", () => {
    acquireCrossTabSurveyLock("nps_quarterly");
    now += 6000;

    expect(acquireCrossTabSurveyLock("nps_quarterly")).toBe(true);
  });

  it("scopes the lock per survey key, not globally", () => {
    acquireCrossTabSurveyLock("nps_quarterly");

    expect(acquireCrossTabSurveyLock("csat_after_upload")).toBe(true);
  });
});
