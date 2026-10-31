import { renderHook } from "@testing-library/react";
import type { SemanticLink } from "../sceneTypes";
import { useShortestPath } from "../useShortestPath";

// ── helpers ──────────────────────────────────────────────────────────────────

function link(sid: string, tid: string): SemanticLink {
  // source/target are required by BusinessGraphLink (the wire shape); _sid/_tid
  // are what the BFS actually reads — set them consistently.
  return { source: sid, target: tid, _sid: sid, _tid: tid };
}

// ── tests ─────────────────────────────────────────────────────────────────────

describe("useShortestPath", () => {
  afterEach(() => jest.restoreAllMocks());

  describe("guard clauses — returns empty path immediately", () => {
    it("returns empty path when semanticLinks is undefined", () => {
      const { result } = renderHook(() => useShortestPath(undefined, "A", "B"));
      expect(result.current.pathIds.size).toBe(0);
      expect(result.current.pathEdgeKeys.size).toBe(0);
    });

    it("returns empty path when fromId is null", () => {
      const { result } = renderHook(() =>
        useShortestPath([link("A", "B")], null, "B"),
      );
      expect(result.current.pathIds.size).toBe(0);
      expect(result.current.pathEdgeKeys.size).toBe(0);
    });

    it("returns empty path when toId is null", () => {
      const { result } = renderHook(() =>
        useShortestPath([link("A", "B")], "A", null),
      );
      expect(result.current.pathIds.size).toBe(0);
      expect(result.current.pathEdgeKeys.size).toBe(0);
    });

    it("returns empty path when fromId and toId are the same", () => {
      const { result } = renderHook(() =>
        useShortestPath([link("A", "B")], "A", "A"),
      );
      expect(result.current.pathIds.size).toBe(0);
      expect(result.current.pathEdgeKeys.size).toBe(0);
    });

    it("returns empty path for an empty semanticLinks array when no graph can be built", () => {
      const { result } = renderHook(() =>
        useShortestPath([], "A", "B"),
      );
      expect(result.current.pathIds.size).toBe(0);
      expect(result.current.pathEdgeKeys.size).toBe(0);
    });
  });

  describe("no path — disconnected graph", () => {
    it("returns empty path when the two nodes belong to separate components", () => {
      // A-B and C-D are two isolated components; no path from A to D
      const links = [link("A", "B"), link("C", "D")];
      const { result } = renderHook(() => useShortestPath(links, "A", "D"));
      expect(result.current.pathIds.size).toBe(0);
      expect(result.current.pathEdgeKeys.size).toBe(0);
    });

    it("returns empty path when toId is not present in the graph at all", () => {
      const links = [link("A", "B"), link("B", "C")];
      const { result } = renderHook(() =>
        useShortestPath(links, "A", "GHOST"),
      );
      expect(result.current.pathIds.size).toBe(0);
      expect(result.current.pathEdgeKeys.size).toBe(0);
    });

    it("returns empty path when fromId is not present in the graph at all", () => {
      const links = [link("A", "B"), link("B", "C")];
      const { result } = renderHook(() =>
        useShortestPath(links, "GHOST", "C"),
      );
      expect(result.current.pathIds.size).toBe(0);
      expect(result.current.pathEdgeKeys.size).toBe(0);
    });
  });

  describe("direct (single-hop) connection", () => {
    it("includes both endpoints in pathIds for a directly linked pair", () => {
      const links = [link("A", "B")];
      const { result } = renderHook(() => useShortestPath(links, "A", "B"));
      expect(result.current.pathIds).toEqual(new Set(["A", "B"]));
    });

    it("includes exactly one edge key for a single-hop path", () => {
      const links = [link("A", "B")];
      const { result } = renderHook(() => useShortestPath(links, "A", "B"));
      expect(result.current.pathEdgeKeys.size).toBe(1);
    });

    it("traverses undirected — finds a path even when fromId matches _tid, not _sid", () => {
      // Link is stored as B->A; traversal from A to B must still work
      const links = [link("B", "A")];
      const { result } = renderHook(() => useShortestPath(links, "A", "B"));
      expect(result.current.pathIds).toEqual(new Set(["A", "B"]));
    });
  });

  describe("multi-hop path", () => {
    it("returns all intermediate nodes in pathIds for a three-hop chain", () => {
      // Chain: A - B - C - D
      const links = [link("A", "B"), link("B", "C"), link("C", "D")];
      const { result } = renderHook(() =>
        useShortestPath(links, "A", "D"),
      );
      expect(result.current.pathIds).toEqual(new Set(["A", "B", "C", "D"]));
    });

    it("returns three edge keys for a three-hop chain", () => {
      const links = [link("A", "B"), link("B", "C"), link("C", "D")];
      const { result } = renderHook(() =>
        useShortestPath(links, "A", "D"),
      );
      expect(result.current.pathEdgeKeys.size).toBe(3);
    });

    it("finds the shortest path in a diamond-shaped graph (two routes of equal length)", () => {
      // Diamond: A connects to B and C; both B and C connect to D.
      // BFS from A will reach D in two hops regardless of which branch it takes.
      const links = [link("A", "B"), link("B", "D"), link("A", "C"), link("C", "D")];
      const { result } = renderHook(() =>
        useShortestPath(links, "A", "D"),
      );
      // Shortest path is 2 hops: A + one intermediate + D = 3 nodes
      expect(result.current.pathIds.size).toBe(3);
      expect(result.current.pathEdgeKeys.size).toBe(2);
    });

    it("prefers the shorter route when a longer alternative also exists", () => {
      // Short route: A - B - E (2 hops)
      // Long route:  A - C - D - E (3 hops)
      const links = [
        link("A", "B"),
        link("B", "E"),
        link("A", "C"),
        link("C", "D"),
        link("D", "E"),
      ];
      const { result } = renderHook(() =>
        useShortestPath(links, "A", "E"),
      );
      expect(result.current.pathIds).toEqual(new Set(["A", "B", "E"]));
      expect(result.current.pathEdgeKeys.size).toBe(2);
    });
  });

  describe("edgeKey ordering", () => {
    it("produces a lexicographically sorted edge key regardless of traversal direction", () => {
      // Link stored as z -> a; path reconstructed as parent=z, cur=a.
      // edgeKey('z', 'a') must return 'a|z' (not 'z|a') because 'a' < 'z'.
      const links = [link("z", "a")];
      const { result } = renderHook(() =>
        useShortestPath(links, "z", "a"),
      );
      expect(result.current.pathEdgeKeys).toEqual(new Set(["a|z"]));
    });

    it("produces the same edge key for a path traversed in either direction", () => {
      const links = [link("m", "b")];

      const { result: forward } = renderHook(() =>
        useShortestPath(links, "b", "m"),
      );
      const { result: backward } = renderHook(() =>
        useShortestPath(links, "m", "b"),
      );

      expect(forward.current.pathEdgeKeys).toEqual(backward.current.pathEdgeKeys);
    });
  });

  describe("memoisation", () => {
    it("returns the same object reference when inputs have not changed", () => {
      const links = [link("A", "B")];
      const { result, rerender } = renderHook(
        ({ l, f, t }: { l: SemanticLink[]; f: string; t: string }) =>
          useShortestPath(l, f, t),
        { initialProps: { l: links, f: "A", t: "B" } },
      );

      const first = result.current;
      rerender({ l: links, f: "A", t: "B" });
      expect(result.current).toBe(first);
    });

    it("recomputes when fromId changes", () => {
      const links = [link("A", "B"), link("B", "C")];
      const { result, rerender } = renderHook(
        ({ l, f, t }: { l: SemanticLink[]; f: string; t: string }) =>
          useShortestPath(l, f, t),
        { initialProps: { l: links, f: "A", t: "C" } },
      );

      const first = result.current;
      rerender({ l: links, f: "B", t: "C" });
      expect(result.current).not.toBe(first);
      // A->B->C has 3 nodes; B->C has 2 nodes
      expect(result.current.pathIds.size).toBe(2);
    });
  });
});
