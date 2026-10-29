const mockTrackEvent = jest.fn();
jest.mock("@/modules/analytics", () => ({ trackEvent: (...args: unknown[]) => mockTrackEvent(...args) }));

import { trackUploadFailed, trackUploadStarted } from "../brainUploadAnalytics";

const DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document";
const target = { datasetId: "ds-1", fileCount: 3, totalBytes: 300 };

beforeEach(() => jest.clearAllMocks());

// One entry per file made a 1000-file upload's event larger than a keepalive
// request may be, and the event was dropped.
describe("brain upload analytics", () => {
  it("reports each file type once when an upload starts", () => {
    trackUploadStarted({ ...target, fileTypes: [DOCX, DOCX, "text/plain", DOCX] });

    expect(mockTrackEvent).toHaveBeenCalledWith(
      expect.objectContaining({ additionalProperties: expect.objectContaining({ file_types: `${DOCX},text/plain`, file_count: "3" }) }),
    );
  });

  it("reports each file type once when an upload fails", () => {
    trackUploadFailed({ ...target, fileTypes: ["text/plain", "text/plain"], durationMs: 10, errorName: "Error", errorMessage: "boom" });

    expect(mockTrackEvent).toHaveBeenCalledWith(
      expect.objectContaining({ additionalProperties: expect.objectContaining({ file_types: "text/plain" }) }),
    );
  });
});
