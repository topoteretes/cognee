import {
  normalizeDatasetStatusEntry,
  normalizeDatasetStatusResponse,
  INSUFFICIENT_CREDITS_REASON,
} from "@/modules/datasets/datasetStatusDetail";

describe("normalizeDatasetStatusEntry", () => {
  it("returns a null reason for a bare status string (pod predating CLO-306)", () => {
    expect(normalizeDatasetStatusEntry("DATASET_PROCESSING_ERRORED")).toEqual({
      status: "DATASET_PROCESSING_ERRORED",
      reason: null,
    });
  });

  it("returns the reason from a detailed status object (pod with CLO-306)", () => {
    expect(
      normalizeDatasetStatusEntry({
        status: "DATASET_PROCESSING_ERRORED",
        reason: "insufficient_credits",
        error: "Budget has been exceeded!",
      }),
    ).toEqual({ status: "DATASET_PROCESSING_ERRORED", reason: "insufficient_credits" });
  });

  it("returns a null reason for a detailed status object with no reason (e.g. a completed run)", () => {
    expect(normalizeDatasetStatusEntry({ status: "DATASET_PROCESSING_COMPLETED" })).toEqual({
      status: "DATASET_PROCESSING_COMPLETED",
      reason: null,
    });
  });
});

describe("normalizeDatasetStatusResponse", () => {
  it("normalizes every entry in a mixed response independently", () => {
    const result = normalizeDatasetStatusResponse({
      "ds-bare": "DATASET_PROCESSING_COMPLETED",
      "ds-detailed": { status: "DATASET_PROCESSING_ERRORED", reason: INSUFFICIENT_CREDITS_REASON },
    });

    expect(result).toEqual({
      "ds-bare": { status: "DATASET_PROCESSING_COMPLETED", reason: null },
      "ds-detailed": { status: "DATASET_PROCESSING_ERRORED", reason: INSUFFICIENT_CREDITS_REASON },
    });
  });

  it("returns an empty object for an empty response", () => {
    expect(normalizeDatasetStatusResponse({})).toEqual({});
  });
});
