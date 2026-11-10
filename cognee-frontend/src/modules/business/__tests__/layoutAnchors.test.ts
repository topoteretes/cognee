import { buildLayoutAnchors, type LayoutEntity } from "../layoutAnchors";

const noSession = (): boolean => false;
const isSession = (name: string): boolean => name.startsWith("session_");
const ORDER = ["CodeModule", "CodeSymbol", "ExternalDependency", "CodeInsight", "ApiEndpoint"];
// 0.55 * 12 * sqrt(count): the group ring's radius inside a territory.
const groupRing = (count: number): number => 0.55 * 12 * Math.sqrt(count);

function expectPoint(actual: { x: number; y: number }, x: number, y: number): void {
  expect(actual.x).toBeCloseTo(x);
  expect(actual.y).toBeCloseTo(y);
}

function ent(id: string, sets: string[], group: string | null): LayoutEntity {
  return { id, sets, group };
}

function many(prefix: string, count: number, sets: string[], group: string | null): LayoutEntity[] {
  return Array.from({ length: count }, (_, i) => ent(`${prefix}${i}`, sets, group));
}

describe("buildLayoutAnchors", () => {
  it("puts a single source at the origin", () => {
    const { anchors } = buildLayoutAnchors([ent("a", ["s"], null)], ORDER, ["s"], [], noSession);
    expectPoint(anchors.s, 0, 0);
  });

  it("spreads several sources around a ring", () => {
    const { anchors } = buildLayoutAnchors([], ORDER, ["a", "b"], [], noSession);
    expect(anchors.a).not.toEqual(anchors.b);
  });

  it("gives an entity without a group the mean anchor of its sets", () => {
    const { anchors, entityAnchors } = buildLayoutAnchors(
      [ent("a", ["x", "y"], null)], ORDER, ["x", "y"], [], noSession,
    );
    expectPoint(entityAnchors.a, (anchors.x.x + anchors.y.x) / 2, (anchors.x.y + anchors.y.y) / 2);
  });

  it("pulls entities of different groups toward different sectors of their territory", () => {
    const entities = [ent("m", ["repo"], "CodeModule"), ent("s", ["repo"], "CodeSymbol"), ent("d", ["repo"], "ExternalDependency")];
    const { entityAnchors } = buildLayoutAnchors(entities, ORDER, ["repo"], [], noSession);
    const points = new Set([entityAnchors.m, entityAnchors.s, entityAnchors.d].map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`));
    expect(points.size).toBe(3);
  });

  it("puts the first group at the top of the ring, at the group ring radius", () => {
    const { entityAnchors } = buildLayoutAnchors(many("m", 4, ["repo"], "CodeModule"), ORDER, ["repo"], [], noSession);
    expectPoint(entityAnchors.m0, 0, -groupRing(4));
  });

  it("does not move an existing entity's anchor when another group appears", () => {
    const base = many("m", 4, ["repo"], "CodeModule");
    const before = buildLayoutAnchors(base, ORDER, ["repo"], [], noSession);
    const after = buildLayoutAnchors(
      [...base, ...many("s", 1, ["repo"], "CodeSymbol")], ORDER, ["repo"], [], noSession,
    );
    const radiusRatio = groupRing(5) / groupRing(4);
    expectPoint(after.entityAnchors.m0, before.entityAnchors.m0.x * radiusRatio, before.entityAnchors.m0.y * radiusRatio);
  });

  it("measures the group offset in the first real source, not a session set", () => {
    const entities = [ent("a", ["session_x", "repo"], "CodeModule"), ent("b", ["repo"], "CodeSymbol")];
    const { anchors, entityAnchors } = buildLayoutAnchors(entities, ORDER, ["repo"], ["session_x"], isSession);
    const meanX = (anchors.session_x.x + anchors.repo.x) / 2;
    const meanY = (anchors.session_x.y + anchors.repo.y) / 2;
    expectPoint(entityAnchors.a, meanX, meanY - groupRing(2));
  });

  it("pushes two large territories apart by more than the sum of their radii", () => {
    const entities = [...many("a", 6000, ["repoA"], "CodeModule"), ...many("b", 6000, ["repoB"], "CodeModule")];
    const { anchors } = buildLayoutAnchors(entities, ORDER, ["repoA", "repoB"], [], noSession);
    const distance = Math.hypot(anchors.repoA.x - anchors.repoB.x, anchors.repoA.y - anchors.repoB.y);
    expect(distance).toBeGreaterThanOrEqual(2 * 12 * Math.sqrt(6000));
  });

  it("widens the ring for two large text-only sources too", () => {
    const entities = [...many("a", 500, ["docsA"], null), ...many("b", 500, ["docsB"], null)];
    const { anchors } = buildLayoutAnchors(entities, ORDER, ["docsA", "docsB"], [], noSession);
    const ringRadius = Math.abs(anchors.docsA.y) / 0.6;
    expect(ringRadius).toBeCloseTo((2 * 12 * Math.sqrt(500)) / (2 * 0.6));
  });

  it("does not widen the ring for a large session set, which sits on its own arc", () => {
    const entities = [...many("s", 3000, ["session_x"], null), ...many("a", 4, ["docsA"], null), ...many("b", 4, ["docsB"], null)];
    const { anchors } = buildLayoutAnchors(entities, ORDER, ["docsA", "docsB", "session_x"], ["session_x"], isSession);
    const chord = 2 * Math.sin(Math.PI / 3) * 0.6;
    const smallPair = (2 * 12 * Math.sqrt(4)) / chord;
    expect(Math.hypot(anchors.docsA.x / 1.25, anchors.docsA.y / 0.6)).toBeCloseTo(Math.max(300, smallPair));
  });

  it("keeps the minimum ring for small territories", () => {
    const { anchors } = buildLayoutAnchors(many("a", 4, ["repoA"], null), ORDER, ["repoA", "repoB"], [], noSession);
    expect(Math.hypot(anchors.repoA.x, anchors.repoA.y)).toBeGreaterThan(0);
    expectPoint(anchors.repoA, 0, -300 * 0.6);
  });

  it("places session sets on their own arc below the sources", () => {
    const { anchors } = buildLayoutAnchors([], ORDER, ["repo"], ["session_x"], isSession);
    expect(anchors.session_x.y).toBeGreaterThan(anchors.repo.y);
  });

  it("falls back to the origin for an entity whose set has no anchor", () => {
    const { entityAnchors } = buildLayoutAnchors([ent("a", ["ghost"], null)], ORDER, [], [], noSession);
    expectPoint(entityAnchors.a, 0, 0);
  });
});
