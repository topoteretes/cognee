import { act, renderHook } from "@testing-library/react";
import { computeBrainState } from "@/modules/business/computeBrainState";
import { useStructuredExplorer } from "@/modules/business/useStructuredExplorer";

const brain = computeBrainState([
  { id: "a", type: "DltRow", text: "Row Data:\n  company: Atlas Labs\n  empty: " },
  { id: "b", type: "DltRow", text: "Row Data:\n  company: Other" },
], []);

describe("useStructuredExplorer", () => {
  it("skips building the explorer index when disabled but still returns the view", () => {
    const { result } = renderHook(() => useStructuredExplorer(brain, "one", false));
    expect(result.current.index.rowIds.size).toBe(0);
    expect(result.current.view).toBe(brain);
  });

  it("filters exact values, supports empty cells and clears on a dataset switch", () => {
    const { result, rerender } = renderHook(({ dataset }) => useStructuredExplorer(brain, dataset), { initialProps: { dataset: "one" } });
    expect(result.current.rowIds).toBeNull();
    act(() => result.current.setFilter({ column: "company", value: "Atlas Labs" }));
    expect(result.current.rowIds).toEqual(new Set(["a"]));
    act(() => result.current.setFilter({ column: "empty", value: "" }));
    expect(result.current.rowIds).toEqual(new Set(["a"]));
    act(() => result.current.setFilter({ column: "company", value: "No match" }));
    expect(result.current.rowIds).toEqual(new Set());
    rerender({ dataset: "two" });
    expect(result.current.rowIds).toBeNull();
    expect(result.current.filter.column).toBeNull();
    rerender({ dataset: "one" });
    expect(result.current.rowIds).toBeNull();
  });

  it("preserves the selected value and camera key when graph data refreshes", () => {
    const { result, rerender } = renderHook(({ data }) => useStructuredExplorer(data, "one"), { initialProps: { data: brain } });
    act(() => result.current.setFilter({ column: "company", value: "Atlas Labs" }));
    const key = result.current.filterKey;
    rerender({ data: computeBrainState([], []) });
    expect(result.current.rowIds).toEqual(new Set());
    expect(result.current.filterKey).toBe(key);
  });
});
