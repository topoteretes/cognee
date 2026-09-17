import type { BusinessEntity } from "../../sceneTypes";
import { hitTestEntity, hitTestAgentMarker, type Transform } from "../businessHitTest";

afterEach(() => jest.restoreAllMocks());

function entity(id: string, x: number, y: number, radii: { _r?: number; _rVisual?: number } = {}): BusinessEntity {
  return { id, x, y, ...radii } as BusinessEntity;
}

function transform(k: number): Transform {
  return { x: 0, y: 0, k };
}

// hitTestEntity works in world space (hitTestScene projects the cursor
// before calling it); hitTestAgentMarker takes the raw canvas offset, which
// with a zero-translation transform is simply worldX * k.
function screenXFor(worldX: number, k: number): number {
  return worldX * k;
}

describe("hitTestEntity", () => {
  // CLO-604 draws entities at visualRadius (up to 20 world units) while the
  // simulation's collision radius _r tops out at 16. Hit-testing _r left the
  // visible rim dead: at high zoom the 18px screen pad (2.25 world units at
  // k=8) no longer covers the gap, so the outermost ring of every
  // mid-importance node accepted neither hover nor click.
  it("accepts a click on the visible rim that the physics radius alone would miss", () => {
    const node = entity("A", 0, 0, { _r: 5, _rVisual: 20 });
    // Inside the drawn disc, well outside _r plus the pad (5 + 2.25 = 7.25).
    expect(hitTestEntity([node], 15, 0, 8)).toBe(node);
  });

  it("still rejects a click outside the drawn disc and its screen pad", () => {
    const node = entity("A", 0, 0, { _r: 5, _rVisual: 20 });
    expect(hitTestEntity([node], 30, 0, 8)).toBeNull();
  });

  it("falls back to the physics radius for an entity the draw pass hasn't reached", () => {
    const node = entity("A", 0, 0, { _r: 16 });
    expect(hitTestEntity([node], 10, 0, 8)).toBe(node);
    expect(hitTestEntity([node], 30, 0, 8)).toBeNull();
  });

  it("keeps the sticky hover on the entity already hovered while the cursor is still within it", () => {
    const sticky = entity("A", 0, 0, { _r: 5, _rVisual: 20 });
    const other = entity("B", 12, 0, { _r: 5, _rVisual: 20 });
    expect(hitTestEntity([sticky, other], 9, 0, 8, "A")).toBe(sticky);
  });

  it("picks the closest entity when several are within range", () => {
    const near = entity("near", 0, 0, { _rVisual: 6 });
    const far = entity("far", 14, 0, { _rVisual: 6 });
    expect(hitTestEntity([far, near], 2, 0, 1)).toBe(near);
  });
});

describe("hitTestAgentMarker", () => {
  const marker = { x: 0, y: 0 };

  it("keeps a comfortable minimum target when the drawn dot is tiny", () => {
    const k = 8;
    // 14 screen px is 1.75 world units at this zoom; the dot is smaller.
    expect(hitTestAgentMarker(marker, screenXFor(1.5, k), 0, transform(k), 0.5)).toBe(true);
  });

  // The marker pulses and can carry an emphasis ring, so past k≈4.6 its drawn
  // extent overtakes the minimum and the target has to follow it.
  it("extends to the drawn radius when that is larger than the minimum", () => {
    const k = 8;
    const drawn = 6;
    expect(hitTestAgentMarker(marker, screenXFor(4, k), 0, transform(k), drawn)).toBe(true);
    expect(hitTestAgentMarker(marker, screenXFor(4, k), 0, transform(k), 0.5)).toBe(false);
  });

  it("rejects a point outside both the minimum and the drawn radius", () => {
    const k = 8;
    expect(hitTestAgentMarker(marker, screenXFor(9, k), 0, transform(k), 6)).toBe(false);
  });
});
