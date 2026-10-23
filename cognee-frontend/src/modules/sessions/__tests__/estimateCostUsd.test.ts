import { estimateCostUsd, LLM_COST_PER_1M_TOKENS } from "../getSessions";

describe("estimateCostUsd", () => {
  it("falls back to the $1.00 / 1M gateway rate when the env override is unset", async () => {
    // Re-import with the override removed so this asserts the literal fallback,
    // not whatever NEXT_PUBLIC_LLM_COST_PER_1M_TOKENS happens to be in this run.
    const saved = process.env.NEXT_PUBLIC_LLM_COST_PER_1M_TOKENS;
    delete process.env.NEXT_PUBLIC_LLM_COST_PER_1M_TOKENS;
    try {
      await jest.isolateModulesAsync(async () => {
        const fresh = await import("../getSessions");
        expect(fresh.LLM_COST_PER_1M_TOKENS).toBe(1.0);
      });
    } finally {
      if (saved !== undefined) process.env.NEXT_PUBLIC_LLM_COST_PER_1M_TOKENS = saved;
    }
  });

  it("bills input and output tokens at the same flat rate", () => {
    expect(estimateCostUsd(1_000_000, 1_000_000)).toBeCloseTo(2 * LLM_COST_PER_1M_TOKENS, 6);
  });

  it("scales linearly with token usage", () => {
    expect(estimateCostUsd(400_000, 100_000)).toBeCloseTo(0.5 * LLM_COST_PER_1M_TOKENS, 6);
  });

  it("is zero when no tokens were used", () => {
    expect(estimateCostUsd(0, 0)).toBe(0);
  });
});
