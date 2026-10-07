import { sourceLabel, sourceTooltipLabel } from "../computeBrainState";

const SESSION_SET = "session_learnings:researcher:9f3c1a2b-77d0-4e51-9a10-badc0ffee123";

afterEach(() => jest.restoreAllMocks());

describe("sourceLabel", () => {
  it("passes an ordinary source name through untouched", () => {
    expect(sourceLabel("customer-support-tickets")).toBe("customer-support-tickets");
  });

  it("shortens a session set to base, agent and six characters of id", () => {
    expect(sourceLabel(SESSION_SET)).toBe("session learnings · researcher · 9f3c1a");
  });

  it("drops the id segment when the set carries none", () => {
    expect(sourceLabel("agent_trace_feedbacks:researcher")).toBe("agent feedback · researcher");
  });
});

describe("sourceTooltipLabel", () => {
  // The rail displays sourceLabel and, before this, labelled its tooltip with
  // sourceLabel too — so hovering repeated what was already on screen and the
  // full identifier was unreachable anywhere in the UI.
  it("keeps the full raw name alongside the shortened label", () => {
    const tooltip = sourceTooltipLabel(SESSION_SET);
    expect(tooltip).toContain(sourceLabel(SESSION_SET));
    expect(tooltip).toContain(SESSION_SET);
  });

  it("does not repeat a name that was never shortened", () => {
    expect(sourceTooltipLabel("customer-support-tickets")).toBe("customer-support-tickets");
  });
});
