import { isSessionEvent, filterSessionEvents, reportDroppedSessionEvents, resetDropReportCooldown } from "../sessionEvent";

const mockAddBreadcrumb = jest.fn();
jest.mock("@/utils/monitoring", () => ({
  addBreadcrumb: (...args: unknown[]) => mockAddBreadcrumb(...args),
}));

beforeEach(() => {
  mockAddBreadcrumb.mockClear();
  resetDropReportCooldown();
});

describe("isSessionEvent", () => {
  // The identity requirement is not pedantry: useBusinessLiveUpdates keys its
  // dedupe on kind/qa_id/time/question, so two events carrying neither qa_id
  // nor time collapse onto one key and the second is silently dropped as a
  // repeat. Accepting them would look like validation while still losing data.
  it.each([
    ["a full search event", { kind: "search", qa_id: "q1", time: "2026-01-01T00:00:00Z", question: "why" }],
    ["qa_id alone", { qa_id: "q1" }],
    ["time alone", { time: "2026-01-01T00:00:00Z" }],
    ["node_ids of strings", { qa_id: "q1", node_ids: ["n1", "n2"] }],
    ["unknown extra fields", { qa_id: "q1", something_new: 42 }],
  ])("accepts %s", (_label, value) => {
    expect(isSessionEvent(value)).toBe(true);
  });

  it.each([
    ["no identity field at all", { kind: "search", question: "orphan" }],
    ["an epoch-number time", { qa_id: "q1", time: 1735689600000 }],
    ["a non-string kind", { qa_id: "q1", kind: 7 }],
    ["a non-string question", { qa_id: "q1", question: {} }],
    ["node_ids null", { qa_id: "q1", node_ids: null }],
    ["node_ids of objects", { qa_id: "q1", node_ids: [{ id: "n1" }] }],
    ["an empty object", {}],
    ["null", null],
    ["a string", "search"],
  ])("rejects %s", (_label, value) => {
    expect(isSessionEvent(value)).toBe(false);
  });
});

describe("filterSessionEvents", () => {
  it("keeps the usable events and reports the count it dropped", () => {
    const good = { qa_id: "q1", time: "2026-01-01T00:00:00Z" };
    const onDropped = jest.fn();
    expect(filterSessionEvents([good, {}, "junk"], onDropped)).toEqual([good]);
    expect(onDropped).toHaveBeenCalledWith(2, 3);
  });

  it("says nothing when every event is usable", () => {
    const onDropped = jest.fn();
    filterSessionEvents([{ qa_id: "q1" }], onDropped);
    expect(onDropped).not.toHaveBeenCalled();
  });

  it("survives an absent callback, since the guard matters more than the signal", () => {
    expect(filterSessionEvents([{}, { qa_id: "q1" }])).toEqual([{ qa_id: "q1" }]);
  });
});

describe("reportDroppedSessionEvents", () => {
  it("leaves one breadcrumb carrying the transport and the counts", () => {
    reportDroppedSessionEvents("socket", "ds-1")(2, 5);
    expect(mockAddBreadcrumb).toHaveBeenCalledWith({
      category: "mindmap.live-events",
      level: "warning",
      message: "dropped 2/5 unusable live events over socket",
      data: { datasetId: "ds-1", transport: "socket", dropped: 2, total: 5 },
    });
  });

  it("rate-limits, so a 1.5s poll cannot evict the rest of the Sentry trail", () => {
    const report = reportDroppedSessionEvents("poll", "ds-1");
    for (let i = 0; i < 50; i++) report(1, 1);
    // Sentry keeps 100 breadcrumbs; a per-tick warning would fill that buffer
    // with one repeated message and push out pod.ts's HTTP trail.
    expect(mockAddBreadcrumb).toHaveBeenCalledTimes(1);
  });

  it("reports each transport and dataset separately", () => {
    reportDroppedSessionEvents("poll", "ds-1")(1, 1);
    reportDroppedSessionEvents("socket", "ds-1")(1, 1);
    reportDroppedSessionEvents("poll", "ds-2")(1, 1);
    // A drift on one dataset must not silence the signal for another, nor
    // hide that both transports are seeing it.
    expect(mockAddBreadcrumb).toHaveBeenCalledTimes(3);
  });
});
