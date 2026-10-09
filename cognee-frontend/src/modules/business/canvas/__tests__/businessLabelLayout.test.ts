import { pickNonOverlappingLabels, type LabelCandidate, type Box } from "../businessLabelLayout";

function candidate(overrides: Partial<LabelCandidate> = {}): LabelCandidate {
  return { key: "a", x: 0, y: 0, width: 20, height: 10, priority: 1, ...overrides };
}

afterEach(() => jest.restoreAllMocks());

describe("pickNonOverlappingLabels", () => {
  it("accepts a single candidate with no obstacles", () => {
    const picked = pickNonOverlappingLabels([candidate()]);
    expect(picked).toEqual(new Set(["a"]));
  });

  it("keeps the higher-priority label and drops the overlapping lower one", () => {
    const picked = pickNonOverlappingLabels([
      candidate({ key: "low", priority: 1 }),
      candidate({ key: "high", priority: 2, x: 5 }),
    ]);
    expect(picked).toEqual(new Set(["high"]));
  });

  it("accepts both labels when they don't overlap", () => {
    const picked = pickNonOverlappingLabels([
      candidate({ key: "left", x: 0 }),
      candidate({ key: "right", x: 200 }),
    ]);
    expect(picked).toEqual(new Set(["left", "right"]));
  });

  it("rejects a label that overlaps an obstacle even with no competing label", () => {
    const nodeCircle: Box = { x1: -10, y1: -10, x2: 10, y2: 10 };
    const picked = pickNonOverlappingLabels([candidate({ key: "a", x: 0, y: 0 })], [nodeCircle]);
    expect(picked.has("a")).toBe(false);
  });

  it("never adds an obstacle box itself to the picked set", () => {
    const nodeCircle: Box = { x1: -10, y1: -10, x2: 10, y2: 10 };
    const picked = pickNonOverlappingLabels([candidate({ key: "a", x: 500 })], [nodeCircle]);
    expect(picked).toEqual(new Set(["a"]));
  });

  // Infinity is what the entity layer marks a selected/hovered/on-path label
  // with. Sorting alone never delivered that guarantee: obstacles are in the
  // collision set before the first candidate is weighed, so a node circle
  // silently outranked the label the user had actually asked for.
  it("places an Infinity-priority label even when a node circle sits on it", () => {
    const nodeCircle: Box = { x1: -10, y1: -10, x2: 10, y2: 10 };
    const picked = pickNonOverlappingLabels([candidate({ key: "selected", priority: Infinity })], [nodeCircle]);
    expect(picked).toEqual(new Set(["selected"]));
  });

  it("places an Infinity-priority label over a competing ordinary label", () => {
    const picked = pickNonOverlappingLabels([
      candidate({ key: "ordinary", priority: 5 }),
      candidate({ key: "selected", priority: Infinity, x: 4 }),
    ]);
    expect(picked).toEqual(new Set(["selected"]));
  });

  it("still blocks ordinary labels with the box a forced label occupies", () => {
    const picked = pickNonOverlappingLabels([
      candidate({ key: "selected", priority: Infinity }),
      candidate({ key: "ordinary", priority: 5, x: 4 }),
    ]);
    expect(picked).toEqual(new Set(["selected"]));
  });

  // The collision set is bucketed by world-space cell. Negative coordinates
  // are ordinary here (the simulation centres on the origin), and cells must
  // not alias across the sign boundary or two distant labels would collide.
  it("does not confuse labels that sit far apart across the origin", () => {
    const picked = pickNonOverlappingLabels([
      candidate({ key: "left", x: -640, y: -640 }),
      candidate({ key: "right", x: 640, y: 640 }),
      candidate({ key: "mixed", x: -640, y: 640 }),
    ]);
    expect(picked).toEqual(new Set(["left", "right", "mixed"]));
  });

  it("still rejects overlapping labels that share a bucket boundary", () => {
    const picked = pickNonOverlappingLabels([
      candidate({ key: "high", priority: 2, x: -1 }),
      candidate({ key: "low", priority: 1, x: 1 }),
    ]);
    expect(picked).toEqual(new Set(["high"]));
  });

  it("rejects a label overlapping an obstacle that straddles a bucket boundary", () => {
    const straddling: Box = { x1: -70, y1: -70, x2: 70, y2: 70 };
    const picked = pickNonOverlappingLabels([candidate({ key: "a", x: 64, y: 64 })], [straddling]);
    expect(picked.has("a")).toBe(false);
  });
});
