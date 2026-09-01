import { summarizeHourlyCosts, sumMeasuredTokens, pickTicks, niceYAxis } from "../CostPanel";
import type { TenantHourlyCosts } from "@/modules/billing/getTenantHourlyCosts";
import type { PipelineRun } from "@/ui/elements/AgentActivityTerminal";
import {
  estimateNoCogneeCostUsd,
  noCogneeCostFromSpendUsd,
  tokensAvoided,
  tokensFromSpendUsd,
  BASELINE_COST_PER_1M_TOKENS,
  NO_COGNEE_TOKEN_MULTIPLIER,
  LLM_COST_PER_1M_TOKENS,
} from "@/modules/sessions/getSessions";

function run(overrides: Partial<PipelineRun>): PipelineRun {
  return {
    id: "id-1",
    pipeline_name: null,
    status: "COMPLETED",
    dataset_id: null,
    dataset_name: null,
    owner_id: null,
    owner_email: null,
    created_at: null,
    pipeline_run_id: null,
    kind: "operation",
    operation_name: "recall",
    origin: null,
    outcome: null,
    tokens_in: null,
    tokens_out: null,
    started_at: null,
    ended_at: null,
    user_id: null,
    session_id: null,
    parent_operation_id: null,
    background: null,
    ...overrides,
  };
}

function costs(points: TenantHourlyCosts["points"]): TenantHourlyCosts {
  return {
    tenantId: "tenant-1",
    start: "2026-08-05T00:00:00.000Z",
    end: "2026-08-05T12:00:00.000Z",
    currency: "USD",
    points,
  };
}

describe("summarizeHourlyCosts", () => {
  it("uses hourly endpoint points for the 24h cost chart", () => {
    const now = new Date("2026-08-05T12:30:00.000Z");
    const result = summarizeHourlyCosts("24h", costs([
      { hour: new Date(now.getTime() - 2 * 60 * 60 * 1000).toISOString(), spendUsd: 1.25 },
      { hour: new Date(now.getTime() - 1 * 60 * 60 * 1000).toISOString(), spendUsd: 2.75 },
    ]), now);

    expect(result.series).toHaveLength(24);
    expect(result.total).toBe(4);
    expect(result.series.reduce((sum, n) => sum + n, 0)).toBe(4);
  });

  it("aggregates hourly points into daily buckets for the 7d chart", () => {
    const now = new Date("2026-08-05T12:30:00.000Z");
    const result = summarizeHourlyCosts("7d", costs([
      { hour: new Date(now.getTime() - 3 * 60 * 60 * 1000).toISOString(), spendUsd: 1 },
      { hour: new Date(now.getTime() - 2 * 60 * 60 * 1000).toISOString(), spendUsd: 2 },
    ]), now);

    expect(result.series).toHaveLength(7);
    expect(result.total).toBe(3);
    expect(result.series.reduce((sum, n) => sum + n, 0)).toBe(3);
  });

  it("aggregates hourly points into daily buckets for the 30d chart", () => {
    const now = new Date("2026-08-05T12:30:00.000Z");
    const result = summarizeHourlyCosts("30d", costs([
      { hour: new Date(now.getTime() - 3 * 60 * 60 * 1000).toISOString(), spendUsd: 1 },
      { hour: new Date(now.getTime() - 27 * 24 * 60 * 60 * 1000).toISOString(), spendUsd: 2 },
    ]), now);

    expect(result.series).toHaveLength(30);
    expect(result.total).toBe(3);
    expect(result.series.reduce((sum, n) => sum + n, 0)).toBe(3);
  });

  it("ignores points that fall outside the plotted slots so the headline matches the chart", () => {
    const now = new Date("2026-08-05T12:30:00.000Z");
    const result = summarizeHourlyCosts("24h", costs([
      { hour: new Date(now.getTime() - 1 * 60 * 60 * 1000).toISOString(), spendUsd: 2 },
      // Inside the endpoint's fetched window, outside this 24-slot grid: it has
      // no slot to land in, so counting it would make the headline unverifiable.
      { hour: new Date(now.getTime() - 40 * 60 * 60 * 1000).toISOString(), spendUsd: 500 },
    ]), now);

    expect(result.total).toBe(2);
    expect(result.series.reduce((sum, n) => sum + n, 0)).toBe(2);
  });

  it("sums the baseline total over the same slots as the spend total", () => {
    const now = new Date("2026-08-05T12:30:00.000Z");
    const result = summarizeHourlyCosts("24h", costs([
      { hour: new Date(now.getTime() - 1 * 60 * 60 * 1000).toISOString(), spendUsd: 2 },
      { hour: new Date(now.getTime() - 40 * 60 * 60 * 1000).toISOString(), spendUsd: 500 },
    ]), now);

    expect(result.baselineTotal).toBeCloseTo(result.baseline.reduce((sum, n) => sum + n, 0), 6);
    expect(result.baselineTotal).toBeCloseTo(noCogneeCostFromSpendUsd(2), 6);
  });

  it("returns one point label per series value", () => {
    const now = new Date("2026-08-05T12:30:00.000Z");
    const result = summarizeHourlyCosts("30d", costs([
      { hour: new Date(now.getTime() - 3 * 60 * 60 * 1000).toISOString(), spendUsd: 1 },
    ]), now);

    expect(result.pointLabels).toHaveLength(result.series.length);
  });
});

describe("no-cognee baseline", () => {
  it("prices the baseline at the token multiplier and comparison-model rate", () => {
    expect(estimateNoCogneeCostUsd(1_000_000)).toBeCloseTo(
      NO_COGNEE_TOKEN_MULTIPLIER * BASELINE_COST_PER_1M_TOKENS,
      6,
    );
  });

  it("reaches the same baseline from a spend figure as from the tokens it was priced from", () => {
    const tokens = 4_000_000;
    const spendUsd = (tokens / 1_000_000) * LLM_COST_PER_1M_TOKENS;

    expect(noCogneeCostFromSpendUsd(spendUsd)).toBeCloseTo(estimateNoCogneeCostUsd(tokens), 6);
  });

  it("costs nothing without cognee when nothing was spent with it", () => {
    expect(noCogneeCostFromSpendUsd(0)).toBe(0);
    expect(estimateNoCogneeCostUsd(0)).toBe(0);
  });

  it("plots a baseline point per spend point, each above the spend it is compared to", () => {
    const now = new Date("2026-08-05T12:30:00.000Z");
    const result = summarizeHourlyCosts("24h", costs([
      { hour: new Date(now.getTime() - 2 * 60 * 60 * 1000).toISOString(), spendUsd: 1.25 },
      { hour: new Date(now.getTime() - 1 * 60 * 60 * 1000).toISOString(), spendUsd: 2.75 },
    ]), now);

    expect(result.baseline).toHaveLength(result.series.length);
    expect(result.baselineTotal).toBeGreaterThan(result.total);
    result.series.forEach((spend, i) => {
      expect(result.baseline[i]).toBeGreaterThanOrEqual(spend);
    });
  });

  it("leaves the baseline flat at zero for a range with no spend", () => {
    const now = new Date("2026-08-05T12:30:00.000Z");
    const result = summarizeHourlyCosts("7d", costs([]), now);

    expect(result.baselineTotal).toBe(0);
    expect(result.baseline.every((v) => v === 0)).toBe(true);
  });
});

describe("tokens saved", () => {
  it("counts only the tokens the work would have burned on top of what cognee routed", () => {
    expect(tokensAvoided(1_000_000)).toBeCloseTo(1_000_000 * (NO_COGNEE_TOKEN_MULTIPLIER - 1), 6);
  });

  it("saves no tokens when cognee routed none", () => {
    expect(tokensAvoided(0)).toBe(0);
  });

  it("recovers the token count a spend figure was priced from (still used internally by the no-cognee baseline)", () => {
    expect(tokensFromSpendUsd(LLM_COST_PER_1M_TOKENS)).toBeCloseTo(1_000_000, 6);
  });
});

describe("sumMeasuredTokens", () => {
  const now = new Date("2026-08-05T12:30:00.000Z");

  it("sums real tokens_in/tokens_out off the activity feed, not off a spend figure", () => {
    const runs = [
      run({ started_at: new Date(now.getTime() - 1 * 60 * 60 * 1000).toISOString(), tokens_in: 100, tokens_out: 50 }),
      run({ started_at: new Date(now.getTime() - 2 * 60 * 60 * 1000).toISOString(), tokens_in: 200, tokens_out: 0 }),
    ];

    expect(sumMeasuredTokens("24h", runs, now)).toBe(350);
  });

  it("reports a genuine zero when every measured row routed no tokens", () => {
    const runs = [
      run({ started_at: new Date(now.getTime() - 1 * 60 * 60 * 1000).toISOString(), tokens_in: 0, tokens_out: 0 }),
    ];

    expect(sumMeasuredTokens("24h", runs, now)).toBe(0);
  });

  it("returns null — not a fabricated zero — when nothing in range was ever measured", () => {
    const runs = [
      run({ started_at: new Date(now.getTime() - 1 * 60 * 60 * 1000).toISOString(), tokens_in: null, tokens_out: null }),
    ];

    expect(sumMeasuredTokens("24h", runs, now)).toBeNull();
  });

  it("counts a row with only one side measured, without treating the unmeasured side as zero-and-therefore-suspect", () => {
    const runs = [
      run({ started_at: new Date(now.getTime() - 1 * 60 * 60 * 1000).toISOString(), tokens_in: 40, tokens_out: null }),
    ];

    expect(sumMeasuredTokens("24h", runs, now)).toBe(40);
  });

  it("sums only the rows that were actually measured, skipping unmeasured ones entirely", () => {
    const runs = [
      run({ started_at: new Date(now.getTime() - 1 * 60 * 60 * 1000).toISOString(), tokens_in: null, tokens_out: null }),
      run({ started_at: new Date(now.getTime() - 2 * 60 * 60 * 1000).toISOString(), tokens_in: 10, tokens_out: 10 }),
    ];

    expect(sumMeasuredTokens("24h", runs, now)).toBe(20);
  });

  it("excludes rows outside the plotted range, matching how the dollar totals are scoped", () => {
    const runs = [
      run({ started_at: new Date(now.getTime() - 1 * 60 * 60 * 1000).toISOString(), tokens_in: 10, tokens_out: 0 }),
      // 40h back falls outside the 24h slot grid entirely.
      run({ started_at: new Date(now.getTime() - 40 * 60 * 60 * 1000).toISOString(), tokens_in: 999, tokens_out: 999 }),
    ];

    expect(sumMeasuredTokens("24h", runs, now)).toBe(10);
  });

  it("falls back to created_at when started_at is missing", () => {
    const runs = [
      run({ started_at: null, created_at: new Date(now.getTime() - 1 * 60 * 60 * 1000).toISOString(), tokens_in: 5, tokens_out: 5 }),
    ];

    expect(sumMeasuredTokens("24h", runs, now)).toBe(10);
  });

  it("returns null for an empty activity feed", () => {
    expect(sumMeasuredTokens("24h", [], now)).toBeNull();
  });
});

describe("niceYAxis", () => {
  it("rounds a $438.51 maximum to $100 steps instead of thirds of the data", () => {
    const axis = niceYAxis(438.51);

    expect(axis.step).toBe(100);
    expect(axis.ticks).toEqual([0, 100, 200, 300, 400, 500]);
  });

  it("always covers the data maximum", () => {
    for (const dataMax of [0.37, 4.2, 18.37, 96, 438.51, 1234, 87_654]) {
      expect(niceYAxis(dataMax).max).toBeGreaterThanOrEqual(dataMax);
    }
  });

  it("steps only in 1, 2, 2.5 or 5 times a power of ten", () => {
    for (const dataMax of [0.37, 4.2, 18.37, 96, 438.51, 1234, 87_654]) {
      const { step } = niceYAxis(dataMax);
      const mantissa = step / 10 ** Math.floor(Math.log10(step));

      expect([1, 2, 2.5, 5]).toContain(Number(mantissa.toFixed(6)));
    }
  });

  it("keeps the gridline count within the cap", () => {
    for (const dataMax of [0.37, 4.2, 18.37, 96, 438.51, 1234, 87_654]) {
      expect(niceYAxis(dataMax).ticks.length).toBeLessThanOrEqual(6);
    }
  });

  it("puts the last tick exactly on the axis maximum, so labels sit on gridlines", () => {
    const axis = niceYAxis(18.37);

    expect(axis.ticks[axis.ticks.length - 1]).toBe(axis.max);
    expect(axis.ticks[0]).toBe(0);
  });

  it("falls back to a unit axis when there is no spend to scale to", () => {
    expect(niceYAxis(0)).toEqual({ max: 1, step: 1, ticks: [0, 1] });
  });
});

describe("pickTicks", () => {
  it("always includes the true last point as a tick, not a stale stride-sampled one", () => {
    // A fixed stride of 6 over 24 points (floor(24/4)) lands on 0,6,12,18 —
    // skipping 23, the actual newest hour. This is the bug: the axis's
    // rightmost label must be the real last point, not 5 hours stale.
    const labels = Array.from({ length: 24 }, (_, i) => `${i}:00`);
    const ticks = pickTicks(labels);

    expect(ticks[ticks.length - 1].index).toBe(23);
    expect(ticks[ticks.length - 1].label).toBe("23:00");
  });

  it("always includes the first point as a tick", () => {
    const labels = Array.from({ length: 24 }, (_, i) => `${i}:00`);
    const ticks = pickTicks(labels);

    expect(ticks[0].index).toBe(0);
  });

  it("returns every point untouched when there are already few enough", () => {
    const labels = ["8/1", "8/2", "8/3"];
    const ticks = pickTicks(labels);

    expect(ticks).toEqual([
      { index: 0, label: "8/1" },
      { index: 1, label: "8/2" },
      { index: 2, label: "8/3" },
    ]);
  });
});
