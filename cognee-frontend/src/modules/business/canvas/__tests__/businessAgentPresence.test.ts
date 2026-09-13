import type { BusinessEntity } from "../../sceneTypes";
import type { Spotlight } from "../businessDraw";
import {
  computeMarkerPosition,
  computeCentroid,
  computePresenceAlpha,
  computeAgentPresence,
  markerPaddingWorld,
  markerRadiusWorld,
  markerHitRadiusWorld,
  type AskingAgent,
} from "../businessAgentPresence";

afterEach(() => jest.restoreAllMocks());

function entity(id: string, x?: number, y?: number): BusinessEntity {
  return { id, x, y } as BusinessEntity;
}

function spotlight(overrides: Partial<Spotlight> = {}): Spotlight {
  return { ids: new Set(["A", "B"]), startedAt: 1_000, until: 10_000, source: "answer", ...overrides };
}

function agent(overrides: Partial<AskingAgent> = {}): AskingAgent {
  return { id: "agent-1", name: "researcher", startedAt: 1_000, until: 10_000, ...overrides };
}

describe("markerPaddingWorld", () => {
  it("matches the flat 40-unit clearance the marker has always used at rest", () => {
    expect(markerPaddingWorld(1)).toBe(40);
  });

  // The entity layer floors its render radius at MIN_NODE_SCREEN_PX/k, so
  // below k≈0.22 a node's world radius alone exceeds 40 and a fixed padding
  // drops the marker inside the cluster it points at.
  it("grows past the entity layer's own low-zoom radius floor", () => {
    const k = 0.1;
    const nodeRadiusFloorWorld = 4.5 / k;
    expect(markerPaddingWorld(k)).toBeGreaterThan(nodeRadiusFloorWorld);
    expect(markerPaddingWorld(k)).toBeGreaterThan(markerPaddingWorld(1));
  });

  it("tightens when zoomed in, where a 40-unit gap would push the marker off screen", () => {
    expect(markerPaddingWorld(8)).toBeLessThan(markerPaddingWorld(1));
  });
});

describe("markerRadiusWorld / markerHitRadiusWorld", () => {
  it("never reports a hit radius smaller than the radius it draws", () => {
    [0.1, 0.5, 1, 2, 4, 6, 8].forEach((k) => {
      const drawn = markerRadiusWorld(k, false, 400);
      expect(markerHitRadiusWorld(k, false, 400, false)).toBeGreaterThanOrEqual(drawn);
    });
  });

  // At high zoom the dot outgrows the old flat 14-screen-px threshold, so a
  // target that never pulsed disagreed with the drawing for most of every
  // pulse cycle.
  it("pulses with the dot rather than holding a flat target", () => {
    const k = 8;
    const flatThresholdWorld = 14 / k;
    expect(markerRadiusWorld(k, false, 0)).toBeGreaterThan(flatThresholdWorld);
    expect(markerHitRadiusWorld(k, false, 550, false))
      .toBeGreaterThan(markerHitRadiusWorld(k, false, 0, false));
  });

  it("holds still for reduced motion", () => {
    expect(markerRadiusWorld(2, true, 0)).toBe(markerRadiusWorld(2, true, 550));
  });

  it("covers the emphasis ring, which sits outside the dot", () => {
    expect(markerHitRadiusWorld(2, true, 0, true)).toBeGreaterThan(markerHitRadiusWorld(2, true, 0, false));
  });
});

describe("computeMarkerPosition", () => {
  it("returns null for an empty id set", () => {
    expect(computeMarkerPosition([entity("A", 0, 0)], new Set(), 1)).toBeNull();
  });

  it("returns null when none of the spotlighted ids have a settled position", () => {
    const entities = [entity("A"), entity("B", 10, 10)];
    expect(computeMarkerPosition(entities, new Set(["A"]), 1)).toBeNull();
  });

  it("places the marker outside the top-right corner of the spotlighted bounding box", () => {
    const entities = [entity("A", 0, 0), entity("B", 20, 30), entity("C", -5, 5)];
    const result = computeMarkerPosition(entities, new Set(["A", "B", "C"]), 1);
    expect(result).toEqual({ x: 20 + 40, y: 0 - 40 });
  });

  it("ignores entities outside the id set even when they'd widen the box", () => {
    const entities = [entity("A", 0, 0), entity("B", 1000, 1000)];
    const result = computeMarkerPosition(entities, new Set(["A"]), 1);
    expect(result).toEqual({ x: 40, y: -40 });
  });
});

describe("computeCentroid", () => {
  it("returns null for an empty id set", () => {
    expect(computeCentroid([entity("A", 0, 0)], new Set())).toBeNull();
  });

  it("returns null when none of the spotlighted ids have a settled position", () => {
    expect(computeCentroid([entity("A")], new Set(["A"]))).toBeNull();
  });

  it("averages the positions of the spotlighted entities", () => {
    const entities = [entity("A", 0, 0), entity("B", 10, 20)];
    expect(computeCentroid(entities, new Set(["A", "B"]))).toEqual({ x: 5, y: 10 });
  });
});

describe("computePresenceAlpha", () => {
  const START = 1_000;
  const END = 10_000;

  it("is fully transparent right at the start of the window", () => {
    expect(computePresenceAlpha(START, START, END)).toBe(0);
  });

  it("reaches full opacity once past the fade-in", () => {
    expect(computePresenceAlpha(START + 300, START, END)).toBe(1);
  });

  it("stays at full opacity in the middle of the window", () => {
    expect(computePresenceAlpha((START + END) / 2, START, END)).toBe(1);
  });

  it("fades out over the final 1500ms", () => {
    expect(computePresenceAlpha(END - 750, START, END)).toBeCloseTo(0.5, 5);
  });

  it("is zero once the window has expired", () => {
    expect(computePresenceAlpha(END, START, END)).toBe(0);
    expect(computePresenceAlpha(END + 1, START, END)).toBe(0);
  });

  // The envelope used to reconstruct elapsed time from a hardcoded 9000ms
  // width, so a longer window read as a negative elapsed and pinned alpha to
  // zero for however much longer than 9s it ran.
  it("plays the same envelope for a window wider than the old hardcoded 9s", () => {
    const wideEnd = START + 16_200;
    expect(computePresenceAlpha(START + 300, START, wideEnd)).toBe(1);
    expect(computePresenceAlpha(START + 12_000, START, wideEnd)).toBe(1);
    expect(computePresenceAlpha(wideEnd - 750, START, wideEnd)).toBeCloseTo(0.5, 5);
  });

  it("plays the same envelope for a window narrower than it", () => {
    const narrowEnd = START + 1_800;
    expect(computePresenceAlpha(START + 300, START, narrowEnd)).toBe(1);
    expect(computePresenceAlpha(narrowEnd - 750, START, narrowEnd)).toBeCloseTo(0.5, 5);
  });
});

describe("computeAgentPresence", () => {
  const entities = [entity("A", 0, 0), entity("B", 10, 10)];

  it("returns marker, centroid and alpha for the agent's own active answer", () => {
    expect(computeAgentPresence(entities, spotlight(), agent(), 5_000, 1)).toEqual({
      markerWorld: { x: 10 + 40, y: 0 - 40 },
      centroidWorld: { x: 5, y: 5 },
      alpha: 1,
    });
  });

  it("also draws during the reasoning-trace walk that precedes the answer", () => {
    const walk = spotlight({ source: "trace", startedAt: 4_000, until: 5_800 });
    expect(computeAgentPresence(entities, walk, agent(), 5_000, 1)).not.toBeNull();
  });

  // The marker names an agent. Painting it onto a spotlight nobody asked for
  // attributes a claim to someone who never made it.
  it("draws nothing on an auto-insight spotlight, even while an agent window is open", () => {
    const insight = spotlight({ source: "insight" });
    expect(computeAgentPresence(entities, insight, agent(), 5_000, 1)).toBeNull();
  });

  it("draws nothing on a what-if-removal spotlight", () => {
    const whatIf = spotlight({ source: "whatIf", until: 13_000 });
    expect(computeAgentPresence(entities, whatIf, agent(), 5_000, 1)).toBeNull();
  });

  // Finding 1: the agent's window and the answer's spotlight are set at
  // different moments, so the gate has to read the agent's own bounds.
  it("returns null once the agent's window has closed, even with the spotlight still up", () => {
    const stillLit = spotlight({ until: 20_000 });
    expect(computeAgentPresence(entities, stillLit, agent({ until: 4_000 }), 5_000, 1)).toBeNull();
  });

  it("stays visible for the whole answer spotlight when the agent's window covers it", () => {
    const answer = spotlight({ startedAt: 8_200, until: 17_200 });
    const walkedAgent = agent({ startedAt: 1_000, until: 17_200 });
    expect(computeAgentPresence(entities, answer, walkedAgent, 17_000, 1)).not.toBeNull();
  });

  it("returns null when the spotlighted ids have no positioned entities", () => {
    expect(computeAgentPresence([entity("A"), entity("B")], spotlight(), agent(), 5_000, 1)).toBeNull();
  });
});
