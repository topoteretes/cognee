// A module, not a global script. Without a top-level import or export TS treats
// this file as a script, so `KEYS` and `original` below land in the global scope
// and collide with any other test file declaring the same names — which showed
// up as `TS2451: Cannot redeclare block-scoped variable` pointing at two
// unrelated files. `AnalyticsPage.test.tsx` opens with the same line.
export {};

/**
 * The per-deploy rate overrides are read at module load, so each case has to
 * re-import the module with the environment already set.
 */
async function loadConstants(env: Record<string, string | undefined>): Promise<{
  LLM_COST_PER_1M_TOKENS: number;
  NO_COGNEE_TOKEN_MULTIPLIER: number;
  BASELINE_COST_PER_1M_TOKENS: number;
  tokensAvoided: (tokens: number) => number;
  estimateNoCogneeCostUsd: (tokens: number) => number;
}> {
  jest.resetModules();
  for (const [key, value] of Object.entries(env)) {
    if (value === undefined) delete process.env[key];
    else process.env[key] = value;
  }
  return import("@/modules/sessions/getSessions");
}

const KEYS = [
  "NEXT_PUBLIC_LLM_COST_PER_1M_TOKENS",
  "NEXT_PUBLIC_NO_COGNEE_TOKEN_MULTIPLIER",
  "NEXT_PUBLIC_BASELINE_COST_PER_1M_TOKENS",
] as const;

const original = new Map(KEYS.map((k) => [k, process.env[k]]));
let warn: jest.SpyInstance;

beforeEach(() => { warn = jest.spyOn(console, "warn").mockImplementation(() => {}); });
afterEach(() => {
  warn.mockRestore();
  for (const [k, v] of original) {
    if (v === undefined) delete process.env[k];
    else process.env[k] = v;
  }
  jest.resetModules();
});

describe("rate overrides", () => {
  it("uses the documented defaults when nothing is set", async () => {
    const c = await loadConstants(Object.fromEntries(KEYS.map((k) => [k, undefined])));
    expect(c.LLM_COST_PER_1M_TOKENS).toBe(1.0);
    expect(c.NO_COGNEE_TOKEN_MULTIPLIER).toBe(7);
    expect(c.BASELINE_COST_PER_1M_TOKENS).toBe(15);
  });

  it("takes a valid override", async () => {
    const c = await loadConstants({ NEXT_PUBLIC_NO_COGNEE_TOKEN_MULTIPLIER: "4.5" });
    expect(c.NO_COGNEE_TOKEN_MULTIPLIER).toBe(4.5);
  });

  it("falls back on a variable that is present but empty", async () => {
    // `X=` in an env file. `Number("")` is 0, which made the multiplier zero and
    // the Cost Savings panel report "0 tokens saved" as a positive claim, with
    // its baseline line collapsed onto actual spend and no error anywhere.
    const c = await loadConstants({ NEXT_PUBLIC_NO_COGNEE_TOKEN_MULTIPLIER: "" });
    expect(c.NO_COGNEE_TOKEN_MULTIPLIER).toBe(7);
    expect(c.tokensAvoided(1000)).toBe(6000);
    // And quietly: `X=` is a variable nobody configured, not one somebody got
    // wrong, so it must not spend the warning that a real typo needs.
    expect(warn).not.toHaveBeenCalled();
  });

  it("falls back on a value that is not a number, rather than propagating NaN", async () => {
    // The panel rendered "NaN tokens saved".
    const c = await loadConstants({ NEXT_PUBLIC_NO_COGNEE_TOKEN_MULTIPLIER: "seven" });
    expect(c.NO_COGNEE_TOKEN_MULTIPLIER).toBe(7);
    expect(Number.isFinite(c.tokensAvoided(1000))).toBe(true);
    expect(warn).toHaveBeenCalledWith(expect.stringContaining("NEXT_PUBLIC_NO_COGNEE_TOKEN_MULTIPLIER"));
  });

  it("refuses a multiplier below 1, which would put the baseline under actual spend", async () => {
    const c = await loadConstants({ NEXT_PUBLIC_NO_COGNEE_TOKEN_MULTIPLIER: "0.5" });
    expect(c.NO_COGNEE_TOKEN_MULTIPLIER).toBe(7);
  });

  it("keeps a multiplier of exactly 1, which honestly claims no saving", async () => {
    const c = await loadConstants({ NEXT_PUBLIC_NO_COGNEE_TOKEN_MULTIPLIER: "1" });
    expect(c.NO_COGNEE_TOKEN_MULTIPLIER).toBe(1);
    expect(c.tokensAvoided(1000)).toBe(0);
  });

  it("refuses a zero gateway rate, which is divided by", async () => {
    const c = await loadConstants({ NEXT_PUBLIC_LLM_COST_PER_1M_TOKENS: "0" });
    expect(c.LLM_COST_PER_1M_TOKENS).toBe(1.0);
  });

  it("refuses a negative comparison rate", async () => {
    const c = await loadConstants({ NEXT_PUBLIC_BASELINE_COST_PER_1M_TOKENS: "-15" });
    expect(c.BASELINE_COST_PER_1M_TOKENS).toBe(15);
  });

  it("keeps the token and dollar baselines built on the same multiplier", async () => {
    // The two used to be guarded differently: `tokensAvoided` clamped a bad
    // multiplier and `estimateNoCogneeCostUsd` did not, so one reported no
    // saving while the other drew the chart's band inverted.
    for (const raw of ["", "seven", "0.5", "-3"]) {
      const c = await loadConstants({ NEXT_PUBLIC_NO_COGNEE_TOKEN_MULTIPLIER: raw });
      expect(c.tokensAvoided(1_000_000)).toBeGreaterThan(0);
      expect(c.estimateNoCogneeCostUsd(1_000_000)).toBeGreaterThan(0);
    }
  });
});
