import type { SemanticLink } from "../../sceneTypes";
import {
  computeNeighborIds,
  computeConnectedIds,
  edgeKey,
} from "../businessGraphFocus";

// ─── helpers ──────────────────────────────────────────────────────────────────

function link(sid: string, tid: string): SemanticLink {
  return { _sid: sid, _tid: tid } as SemanticLink;
}

afterEach(() => jest.restoreAllMocks());

// ─── computeNeighborIds ───────────────────────────────────────────────────────

describe("computeNeighborIds", () => {
  it("returns a set containing only the center id when the links array is empty", () => {
    const result = computeNeighborIds([], "A");
    expect(result).toEqual(new Set(["A"]));
  });

  it("returns only the center id when no link involves it", () => {
    const links = [link("B", "C"), link("C", "D")];
    const result = computeNeighborIds(links, "A");
    expect(result).toEqual(new Set(["A"]));
  });

  it("includes the target when the center is the source", () => {
    const links = [link("A", "B")];
    const result = computeNeighborIds(links, "A");
    expect(result).toEqual(new Set(["A", "B"]));
  });

  it("includes the source when the center is the target", () => {
    const links = [link("B", "A")];
    const result = computeNeighborIds(links, "A");
    expect(result).toEqual(new Set(["A", "B"]));
  });

  it("collects neighbors from both source and target directions across multiple links", () => {
    const links = [link("A", "B"), link("C", "A"), link("D", "E")];
    const result = computeNeighborIds(links, "A");
    expect(result).toEqual(new Set(["A", "B", "C"]));
  });

  it("deduplicates when the same neighbor id appears in multiple links", () => {
    const links = [link("A", "B"), link("A", "B"), link("B", "A")];
    const result = computeNeighborIds(links, "A");
    expect(result).toEqual(new Set(["A", "B"]));
  });

  it("always includes the center id even when it appears in no links", () => {
    const result = computeNeighborIds([link("X", "Y")], "Z");
    expect(result.has("Z")).toBe(true);
  });
});

// ─── computeConnectedIds ──────────────────────────────────────────────────────

describe("computeConnectedIds", () => {
  it("returns an empty set when the links array is empty", () => {
    const result = computeConnectedIds([]);
    expect(result.size).toBe(0);
  });

  it("includes both the source and target of a single link", () => {
    const result = computeConnectedIds([link("A", "B")]);
    expect(result).toEqual(new Set(["A", "B"]));
  });

  it("collects ids from every link", () => {
    const links = [link("A", "B"), link("C", "D"), link("E", "F")];
    const result = computeConnectedIds(links);
    expect(result).toEqual(new Set(["A", "B", "C", "D", "E", "F"]));
  });

  it("deduplicates ids that appear in multiple links", () => {
    const links = [link("A", "B"), link("B", "C")];
    const result = computeConnectedIds(links);
    expect(result).toEqual(new Set(["A", "B", "C"]));
  });

  it("does not include an id that appears in no links", () => {
    const links = [link("A", "B")];
    const result = computeConnectedIds(links);
    expect(result.has("Z")).toBe(false);
  });
});

// ─── edgeKey ──────────────────────────────────────────────────────────────────

describe("edgeKey", () => {
  it("produces the same key regardless of argument order", () => {
    expect(edgeKey("A", "B")).toBe(edgeKey("B", "A"));
  });

  it("produces different keys for different pairs", () => {
    expect(edgeKey("A", "B")).not.toBe(edgeKey("A", "C"));
  });

  it("produces a key in the form 'lesser|greater' when the first argument is lexicographically smaller", () => {
    expect(edgeKey("A", "B")).toBe("A|B");
  });

  it("produces a key in the form 'lesser|greater' when the second argument is lexicographically smaller", () => {
    expect(edgeKey("B", "A")).toBe("A|B");
  });

  it("handles identical ids by producing a key with the id on both sides", () => {
    expect(edgeKey("A", "A")).toBe("A|A");
  });

  it("treats ids as case-sensitive when ordering", () => {
    // Uppercase letters sort before lowercase in ASCII, so "Z" < "a"
    const key = edgeKey("a", "Z");
    expect(key).toBe("Z|a");
  });
});
