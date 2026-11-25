import { mapProcessingStatus } from "../brainsTypes";

describe("mapProcessingStatus", () => {
  it("reports empty when there's no status and the count is zero", () => {
    expect(mapProcessingStatus(undefined, 0)).toBe("empty");
  });

  it("reports completed when there's no status and the count is positive", () => {
    expect(mapProcessingStatus(undefined, 5)).toBe("completed");
  });

  it("reports completed, not empty, when the count is unknown (null)", () => {
    // null means getDatasetDataCount's fallback got capped, i.e. the dataset
    // has far more documents than could be confirmed — never "empty".
    expect(mapProcessingStatus(undefined, null)).toBe("completed");
  });

  it("reports completed, not empty, while the count is still loading (-1)", () => {
    expect(mapProcessingStatus(undefined, -1)).toBe("completed");
  });

  it("reports completed once processing finishes, regardless of doc count", () => {
    expect(mapProcessingStatus("DATASET_PROCESSING_COMPLETED", null)).toBe("completed");
  });

  it("reports failed_insufficient_credits for that specific error reason", () => {
    expect(mapProcessingStatus("DATASET_PROCESSING_ERRORED", null, "insufficient_credits")).toBe(
      "failed_insufficient_credits",
    );
  });

  it("reports failed for any other error reason", () => {
    expect(mapProcessingStatus("DATASET_PROCESSING_ERRORED", null, "some_other_reason")).toBe("failed");
  });

  it("reports running while the pipeline is started", () => {
    expect(mapProcessingStatus("DATASET_PROCESSING_STARTED", null)).toBe("running");
  });

  it("reports pending while the pipeline is only initiated", () => {
    expect(mapProcessingStatus("DATASET_PROCESSING_INITIATED", null)).toBe("pending");
  });
});
