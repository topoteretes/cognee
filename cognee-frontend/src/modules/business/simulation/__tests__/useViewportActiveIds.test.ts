import { renderHook, act } from "@testing-library/react";
import { useViewportActiveIds } from "../useViewportActiveIds";
import type { BusinessEntity } from "../../sceneTypes";

// Derive the type of transformRef from the hook's own signature so we never
// have to import ZoomTransform directly here — keeping the test isolated from
// the d3 type shape.
type TransformRef = Parameters<typeof useViewportActiveIds>[1];

function makeTransformRef(
  invertFn: (point: [number, number]) => [number, number],
): TransformRef {
  // The hook only calls `.invert()`; the rest of ZoomTransform is irrelevant.
  return { current: { invert: invertFn } } as unknown as TransformRef;
}

// Identity: world-space coords == screen-space coords (no zoom, no pan).
function identityRef(): TransformRef {
  return makeTransformRef((p) => p);
}

function makeEntity(id: string, x?: number, y?: number): BusinessEntity {
  return { id, x, y } as BusinessEntity;
}

// Build `count` entities all at the same position.
function buildEntities(count: number, x?: number, y?: number): BusinessEntity[] {
  return Array.from({ length: count }, (_, i) => makeEntity(`e${i}`, x, y));
}

// These mirror the private constants in the source; kept local so we assert
// the observable contract rather than implementation details.
const THRESHOLD = 500;
const WIDTH = 1000;
const HEIGHT = 800;

// With identity transform, width=1000, height=800, padding=400 (internal constant):
//   world bounds: x in [-400 .. 1400], y in [-400 .. 1200]
const IN_VIEW_X = 500;
const IN_VIEW_Y = 400;
const OUT_OF_VIEW_X = 9999;
const OUT_OF_VIEW_Y = 9999;

describe("useViewportActiveIds", () => {
  afterEach(() => {
    jest.restoreAllMocks();
    jest.useRealTimers();
  });

  describe("null pass-through — capping disabled below the threshold", () => {
    it("returns null when entity count is exactly at the threshold", () => {
      const entities = buildEntities(THRESHOLD, IN_VIEW_X, IN_VIEW_Y);
      const { result } = renderHook(() =>
        useViewportActiveIds(entities, identityRef(), WIDTH, HEIGHT),
      );
      expect(result.current).toBeNull();
    });

    it("returns null when entity count is well below the threshold", () => {
      const entities = buildEntities(100, IN_VIEW_X, IN_VIEW_Y);
      const { result } = renderHook(() =>
        useViewportActiveIds(entities, identityRef(), WIDTH, HEIGHT),
      );
      expect(result.current).toBeNull();
    });

    it("returns null when the entity list is empty", () => {
      const { result } = renderHook(() =>
        useViewportActiveIds([], identityRef(), WIDTH, HEIGHT),
      );
      expect(result.current).toBeNull();
    });

    it("returns null immediately when width is 0, even if entity count exceeds the threshold", () => {
      const entities = buildEntities(THRESHOLD + 1, IN_VIEW_X, IN_VIEW_Y);
      const { result } = renderHook(() =>
        useViewportActiveIds(entities, identityRef(), 0, HEIGHT),
      );
      expect(result.current).toBeNull();
    });

    it("returns null immediately when height is 0, even if entity count exceeds the threshold", () => {
      const entities = buildEntities(THRESHOLD + 1, IN_VIEW_X, IN_VIEW_Y);
      const { result } = renderHook(() =>
        useViewportActiveIds(entities, identityRef(), WIDTH, 0),
      );
      expect(result.current).toBeNull();
    });
  });

  describe("viewport capping — active when entity count exceeds the threshold", () => {
    beforeEach(() => {
      jest.useFakeTimers();
    });

    it("returns a Set on the first render when entity count is one above the threshold", () => {
      const entities = buildEntities(THRESHOLD + 1, IN_VIEW_X, IN_VIEW_Y);
      const { result } = renderHook(() =>
        useViewportActiveIds(entities, identityRef(), WIDTH, HEIGHT),
      );
      expect(result.current).toBeInstanceOf(Set);
    });

    it("includes an entity whose position falls inside the padded viewport bounds", () => {
      // 500 far entities push the total above threshold; only the close one
      // should appear in the active set.
      const far = buildEntities(THRESHOLD, OUT_OF_VIEW_X, OUT_OF_VIEW_Y);
      const visible = makeEntity("visible", IN_VIEW_X, IN_VIEW_Y);

      const { result } = renderHook(() =>
        useViewportActiveIds([...far, visible], identityRef(), WIDTH, HEIGHT),
      );

      expect(result.current?.has("visible")).toBe(true);
    });

    it("excludes an entity whose position is outside the padded viewport bounds", () => {
      const far = buildEntities(THRESHOLD, OUT_OF_VIEW_X, OUT_OF_VIEW_Y);
      const outside = makeEntity("outside", OUT_OF_VIEW_X, OUT_OF_VIEW_Y);

      const { result } = renderHook(() =>
        useViewportActiveIds([...far, outside], identityRef(), WIDTH, HEIGHT),
      );

      expect(result.current?.has("outside")).toBe(false);
    });

    it("always includes an entity whose x is undefined, even when its y is out of view", () => {
      const far = buildEntities(THRESHOLD, OUT_OF_VIEW_X, OUT_OF_VIEW_Y);
      const noX = makeEntity("no-x", undefined, OUT_OF_VIEW_Y);

      const { result } = renderHook(() =>
        useViewportActiveIds([...far, noX], identityRef(), WIDTH, HEIGHT),
      );

      expect(result.current?.has("no-x")).toBe(true);
    });

    it("always includes an entity whose y is undefined, even when its x is out of view", () => {
      const far = buildEntities(THRESHOLD, OUT_OF_VIEW_X, OUT_OF_VIEW_Y);
      const noY = makeEntity("no-y", OUT_OF_VIEW_X, undefined);

      const { result } = renderHook(() =>
        useViewportActiveIds([...far, noY], identityRef(), WIDTH, HEIGHT),
      );

      expect(result.current?.has("no-y")).toBe(true);
    });

    it("always includes an entity with neither x nor y (brand-new, not yet seeded)", () => {
      const far = buildEntities(THRESHOLD, OUT_OF_VIEW_X, OUT_OF_VIEW_Y);
      const unpositioned = makeEntity("unpositioned");

      const { result } = renderHook(() =>
        useViewportActiveIds([...far, unpositioned], identityRef(), WIDTH, HEIGHT),
      );

      expect(result.current?.has("unpositioned")).toBe(true);
    });

    it("picks up a newly visible entity after the recheck interval elapses", () => {
      const far = buildEntities(THRESHOLD, OUT_OF_VIEW_X, OUT_OF_VIEW_Y);
      // d3-force mutates entity objects in place — simulate that by keeping a
      // mutable reference and moving it into the viewport after the first tick.
      const mover: BusinessEntity = makeEntity("mover", OUT_OF_VIEW_X, OUT_OF_VIEW_Y);
      const entities = [...far, mover];

      const { result } = renderHook(() =>
        useViewportActiveIds(entities, identityRef(), WIDTH, HEIGHT),
      );

      expect(result.current?.has("mover")).toBe(false);

      mover.x = IN_VIEW_X;
      mover.y = IN_VIEW_Y;

      act(() => jest.advanceTimersByTime(800));

      expect(result.current?.has("mover")).toBe(true);
    });

    it("drops an entity that has moved out of the viewport after the recheck interval elapses", () => {
      const far = buildEntities(THRESHOLD, OUT_OF_VIEW_X, OUT_OF_VIEW_Y);
      const leaver: BusinessEntity = makeEntity("leaver", IN_VIEW_X, IN_VIEW_Y);
      const entities = [...far, leaver];

      const { result } = renderHook(() =>
        useViewportActiveIds(entities, identityRef(), WIDTH, HEIGHT),
      );

      expect(result.current?.has("leaver")).toBe(true);

      leaver.x = OUT_OF_VIEW_X;
      leaver.y = OUT_OF_VIEW_Y;

      act(() => jest.advanceTimersByTime(800));

      expect(result.current?.has("leaver")).toBe(false);
    });

    it("returns to null when entity count drops back to the threshold on a rerender", () => {
      let entities = buildEntities(THRESHOLD + 1, IN_VIEW_X, IN_VIEW_Y);
      const { result, rerender } = renderHook(() =>
        useViewportActiveIds(entities, identityRef(), WIDTH, HEIGHT),
      );

      expect(result.current).toBeInstanceOf(Set);

      entities = buildEntities(THRESHOLD, IN_VIEW_X, IN_VIEW_Y);
      rerender();

      expect(result.current).toBeNull();
    });

    it("accounts for a non-identity zoom transform when computing world-space bounds", () => {
      // 2x zoom with translation: invert([sx, sy]) = [(sx - tx) / k, (sy - ty) / k]
      // With k=2, tx=250, ty=200 and padding=400:
      //   invert([-400, -400]) = [(-400 - 250) / 2, (-400 - 200) / 2] = [-325, -300]
      //   invert([1400, 1200]) = [(1400 - 250) / 2, (1200 - 200) / 2]  = [575,   500]
      const k = 2;
      const tx = 250;
      const ty = 200;
      const zoomedRef = makeTransformRef(
        (p: [number, number]): [number, number] => [(p[0] - tx) / k, (p[1] - ty) / k],
      );

      const far = buildEntities(THRESHOLD, OUT_OF_VIEW_X, OUT_OF_VIEW_Y);
      // (400, 300) lies inside [-325..575, -300..500]
      const inZoom = makeEntity("in-zoom", 400, 300);
      // (600, 300): x=600 > 575, outside the zoomed world bounds
      const outOfZoom = makeEntity("out-of-zoom", 600, 300);

      const { result } = renderHook(() =>
        useViewportActiveIds([...far, inZoom, outOfZoom], zoomedRef, WIDTH, HEIGHT),
      );

      expect(result.current?.has("in-zoom")).toBe(true);
      expect(result.current?.has("out-of-zoom")).toBe(false);
    });
  });
});
