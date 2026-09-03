import { describeProcessingError } from "../processingErrorMessage";
import { DatasetProcessingTimeoutError, DatasetProcessingError } from "@/modules/datasets/pollDatasetStatus";

describe("describeProcessingError", () => {
  describe("when the error is a DatasetProcessingTimeoutError", () => {
    it("returns isTimeout true and a 'still building' message", () => {
      const error = new DatasetProcessingTimeoutError("dataset-1", 600000);

      const result = describeProcessingError(error);

      expect(result.isTimeout).toBe(true);
      expect(result.title).toBe("Still building");
      expect(result.message).toContain("still building");
    });
  });

  describe("when the error is a genuine DatasetProcessingError", () => {
    it("returns isTimeout false and includes the error message", () => {
      const error = new DatasetProcessingError("dataset-1", "DATASET_PROCESSING_ERRORED");

      const result = describeProcessingError(error);

      expect(result.isTimeout).toBe(false);
      expect(result.title).toBe("Knowledge graph build failed");
      expect(result.message).toContain(error.message);
    });
  });

  describe("when the error is not an Error instance", () => {
    it("stringifies the value into the message", () => {
      const result = describeProcessingError("network down");

      expect(result.isTimeout).toBe(false);
      expect(result.message).toContain("network down");
    });
  });
});
