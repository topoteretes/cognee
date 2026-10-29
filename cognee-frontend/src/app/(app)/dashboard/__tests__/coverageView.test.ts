import { deriveCoverageView } from "@/app/(app)/dashboard/coverageView";
import type { Brain, CoverageResult } from "@/app/(app)/memory-gap-analysis/types";

function completeResult(datasetId: string, overallScore: number): CoverageResult {
  return {
    run: {
      run_id: `run-${datasetId}`,
      dataset_id: datasetId,
      status: "complete",
      created_at: "2026-09-01T00:00:00Z",
      recall_count: 1,
      deduped_question_count: 1,
      topic_count: 0,
    },
    overall_score: overallScore,
    topics: [],
    questions: [],
    suggested_topics: [],
  };
}

function brain(id: string, overrides: Partial<Brain> = {}): Brain {
  return { id, name: id, result: null, state: "loaded", isRunning: false, ...overrides };
}

describe("deriveCoverageView", () => {
  it("returns an empty, non-unavailable view for no brains", () => {
    expect(deriveCoverageView([])).toEqual({ topics: [], recallPct: null, unavailable: false });
  });

  it("keeps a row's order and reads its own null score as unmeasured, not zero", () => {
    const brains = [brain("d1", { result: completeResult("d1", 4) }), brain("d2")];

    const view = deriveCoverageView(brains);

    expect(view.topics).toEqual([
      { id: "d1", name: "d1", pct: 80 },
      { id: "d2", name: "d2", pct: null },
    ]);
  });

  it("excludes an unscored brain from the headline instead of averaging its null as a zero", () => {
    const brains = [brain("d1", { result: completeResult("d1", 4) }), brain("d2")];

    expect(deriveCoverageView(brains).recallPct).toBe(80);
  });

  it("averages the raw percentages before rounding, not the already-rounded rows", () => {
    // 3.325/5 = 66.5% and 1.675/5 = 33.5%. Rounded per row that's 67 and 34,
    // whose mean is 50.5 -> 51. The raw mean is 50 -> 50. If this ever starts
    // asserting 51, the derivation started rounding twice.
    const brains = [brain("d1", { result: completeResult("d1", 3.325) }), brain("d2", { result: completeResult("d2", 1.675) })];

    const view = deriveCoverageView(brains);

    expect(view.topics.map((t) => t.pct)).toEqual([67, 34]);
    expect(view.recallPct).toBe(50);
  });

  it("returns null recall when nothing in the set has been scored", () => {
    const brains = [brain("d1"), brain("d2")];

    expect(deriveCoverageView(brains).recallPct).toBeNull();
  });

  it("reads a total outage as unavailable, not as an unscored workspace", () => {
    const brains = [brain("d1", { state: "error" }), brain("d2", { state: "error" })];

    expect(deriveCoverageView(brains).unavailable).toBe(true);
  });

  it("does not call it an outage while some brains are still genuinely unscored", () => {
    // One failed fetch and one brain that answered with no score is not the
    // same as every answer being missing: only the failed one is an outage.
    const brains = [brain("d1", { state: "error" }), brain("d2")];

    expect(deriveCoverageView(brains).unavailable).toBe(false);
  });

  it("calls a failure alongside a real score unavailable, and still reports the score", () => {
    // The load-bearing half of the rule: every brain that is not an error has
    // a score, so the only missing numbers are the ones that failed. Without
    // the `pct !== null` arm the formula reduces to "every brain errored" and
    // this state reads as healthy, which is how a partial outage hid (CLO-726).
    const brains = [brain("d1", { state: "error" }), brain("d2", { result: completeResult("d2", 5) })];

    expect(deriveCoverageView(brains)).toEqual({
      topics: [
        { id: "d1", name: "d1", pct: null },
        { id: "d2", name: "d2", pct: 100 },
      ],
      recallPct: 100,
      unavailable: true,
    });
  });

  it("does not call a workspace with no failures unavailable", () => {
    const brains = [brain("d1", { result: completeResult("d1", 5) }), brain("d2")];

    expect(deriveCoverageView(brains).unavailable).toBe(false);
  });
});
