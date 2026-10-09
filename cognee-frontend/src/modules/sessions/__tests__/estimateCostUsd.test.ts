import { estimateCostUsd, LLM_COST_PER_1M_TOKENS } from "../getSessions";

describe("estimateCostUsd", () => {
  it("defaults to the $2.50 / 1M gateway rate", () => {
    expect(LLM_COST_PER_1M_TOKENS).toBe(2.5);
  });

  it("bills input and output tokens at the same flat rate", () => {
    // 1M in + 1M out = 2M tokens * $2.50/1M = $5.00
    expect(estimateCostUsd(1_000_000, 1_000_000)).toBeCloseTo(5.0, 6);
  });

  it("scales linearly with token usage", () => {
    expect(estimateCostUsd(400_000, 100_000)).toBeCloseTo(1.25, 6);
  });

  it("is zero when no tokens were used", () => {
    expect(estimateCostUsd(0, 0)).toBe(0);
  });
});
