import { computeLinkDensityFade, computeInstanceAlpha } from "../businessDraw";

afterEach(() => jest.restoreAllMocks());

describe("computeLinkDensityFade", () => {
  it("returns full opacity at or below the density threshold", () => {
    expect(computeLinkDensityFade(400)).toBe(1);
    expect(computeLinkDensityFade(50)).toBe(1);
  });

  it("fades a moderately dense graph well below full opacity", () => {
    const fade = computeLinkDensityFade(800);
    expect(fade).toBeLessThan(0.3);
    expect(fade).toBeGreaterThan(0);
  });

  it("never fades below the configured floor even for an extremely dense graph", () => {
    expect(computeLinkDensityFade(1_000_000)).toBe(0.06);
  });
});

describe("computeInstanceAlpha", () => {
  const K_MAX = 1;

  it("keeps the entity layer fully present when zoomed in past the crossfade", () => {
    expect(computeInstanceAlpha(2, K_MAX, true, false)).toBe(1);
  });

  it("fades the entity layer out entirely once zoomed well below the crossfade", () => {
    expect(computeInstanceAlpha(0.2, K_MAX, true, false)).toBe(0);
  });

  it("stays fully present when the graph has no schema layer to cross into", () => {
    expect(computeInstanceAlpha(0.2, K_MAX, false, false)).toBe(1);
  });

  it("stays fully present while a spotlight suppresses the schema layer", () => {
    expect(computeInstanceAlpha(0.2, K_MAX, true, true)).toBe(1);
  });

  it("reports a partial value mid-crossfade", () => {
    const mid = computeInstanceAlpha(0.92, K_MAX, true, false);
    expect(mid).toBeGreaterThan(0);
    expect(mid).toBeLessThan(1);
  });
});
