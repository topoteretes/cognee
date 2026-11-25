import noDatasetStateLabel, {
  NO_DATASET_SELECTED,
  WORKSPACE_UNREADABLE,
  type NoDatasetState,
} from "../noDatasetStateLabel";

function state(overrides: Partial<NoDatasetState> = {}): NoDatasetState {
  return {
    isLoading: false,
    activeDatasetId: null,
    listState: "ready",
    readableDatasetCount: 0,
    governanceFailed: false,
    ...overrides,
  };
}

describe("noDatasetStateLabel", () => {
  it("tells a caller with no readable datasets to create one", () => {
    expect(noDatasetStateLabel(state())).toBe(NO_DATASET_SELECTED);
  });

  it("says nothing while a dataset is on screen", () => {
    expect(noDatasetStateLabel(state({ activeDatasetId: "ds-1", readableDatasetCount: 1 }))).toBeNull();
  });

  it("says nothing while a layer query is still in flight", () => {
    expect(noDatasetStateLabel(state({ isLoading: true }))).toBeNull();
  });

  it("says nothing while the readable list is still answering", () => {
    // Its emptiness means nothing yet, and "create a dataset" on a workspace
    // full of them is worse than a blank second.
    expect(noDatasetStateLabel(state({ listState: "loading" }))).toBeNull();
  });

  it("blames the fetch, not the workspace, when the readable list failed", () => {
    // The regression this file exists for: the gate that silenced the message
    // during loading silenced it for a FAILED list too, leaving a first visit
    // with an empty provenance graph staring at nothing at all.
    expect(noDatasetStateLabel(state({ listState: "unavailable" }))).toBe(WORKSPACE_UNREADABLE);
  });

  it("blames the fetch when the provenance graph failed", () => {
    expect(noDatasetStateLabel(state({ governanceFailed: true }))).toBe(WORKSPACE_UNREADABLE);
  });

  it("prefers the failure message when the list is empty because the fetch failed", () => {
    expect(
      noDatasetStateLabel(state({ listState: "unavailable", governanceFailed: true })),
    ).toBe(WORKSPACE_UNREADABLE);
  });

  it("says nothing when the caller has datasets but auto-focus has not run yet", () => {
    expect(noDatasetStateLabel(state({ readableDatasetCount: 3 }))).toBeNull();
  });
});
