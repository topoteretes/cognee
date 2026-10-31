import type { ReactElement, ReactNode } from "react";
import { renderHook, act } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

// ── Collaborator mocks — declared before module import ────────────────────────

const mockAddFiles = jest.fn();
jest.mock("@/modules/ingestion/addFiles", () => ({
  __esModule: true,
  default: (...args: unknown[]) => mockAddFiles(...args),
}));

const mockCognifyDataset = jest.fn();
jest.mock("@/modules/datasets/cognifyDataset", () => ({
  __esModule: true,
  default: (...args: unknown[]) => mockCognifyDataset(...args),
}));

const mockPollDatasetStatus = jest.fn();
jest.mock("@/modules/datasets/pollDatasetStatus", () => ({
  __esModule: true,
  default: (...args: unknown[]) => mockPollDatasetStatus(...args),
}));

const MOCK_QUERY_KEY = ["dataset-statuses", "tenant-1"] as const;
const mockDatasetStatusQueryKey = jest.fn().mockReturnValue(MOCK_QUERY_KEY);
jest.mock("@/modules/datasets/useDatasetStatuses", () => ({
  datasetStatusQueryKey: (...args: unknown[]) => mockDatasetStatusQueryKey(...args),
}));

const mockInvalidateQueries = jest.fn();
jest.mock("@tanstack/react-query", () => ({
  ...jest.requireActual("@tanstack/react-query"),
  useQueryClient: () => ({ invalidateQueries: mockInvalidateQueries }),
}));

const mockUseTenant = jest.fn();
jest.mock("@/modules/tenant/TenantProvider", () => ({
  useTenant: () => mockUseTenant(),
}));

// ── Module under test — imported after all mocks are in place ─────────────────

import { useBrainUpload } from "@/modules/ingestion/useBrainUpload";
import type { BrainUploadParams } from "@/modules/ingestion/useBrainUpload";
import type { CogneeInstance } from "@/modules/instances/types";
import type { AddFilesResponse } from "@/modules/ingestion/addFiles";
import { MAX_FILES_PER_UPLOAD } from "@/modules/ingestion/uploadLimits";

// ── Fixtures ──────────────────────────────────────────────────────────────────

const TENANT_ID = "tenant-1";
const DATASET_ID = "dataset-42";

const instance: CogneeInstance = {
  name: "test-instance",
  instanceId: "test-instance",
  fetch: jest.fn(),
};

function makeFile(name = "document.pdf", size = 1024): File {
  return new File(["x".repeat(size)], name, { type: "application/pdf" });
}

function makeAddResponse(): AddFilesResponse {
  return {
    status: "ok",
    dataset_name: "My Dataset",
    dataset_id: DATASET_ID,
    pipeline_run_id: "run-123",
  };
}

function makeParams(overrides: Partial<BrainUploadParams> = {}): BrainUploadParams {
  return {
    datasetId: DATASET_ID,
    files: [makeFile()],
    ...overrides,
  };
}

// Minimal QueryClientProvider wrapper — required because some internal React
// Query bookkeeping still runs even when useQueryClient is mocked.
function Wrapper({ children }: { children: ReactNode }): ReactElement {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

// ── Setup ─────────────────────────────────────────────────────────────────────

beforeEach(() => {
  jest.clearAllMocks();
  mockUseTenant.mockReturnValue({ tenant: { tenant_id: TENANT_ID } });
  mockAddFiles.mockResolvedValue(makeAddResponse());
  mockCognifyDataset.mockResolvedValue({ pipeline_run_id: "cognify-run-1" });
  mockPollDatasetStatus.mockResolvedValue("DATASET_PROCESSING_COMPLETED");
});

afterEach(() => jest.restoreAllMocks());

// ── Tests ─────────────────────────────────────────────────────────────────────

describe("useBrainUpload", () => {
  describe("when instance is null", () => {
    it("returns early without calling rememberData", async () => {
      const { result } = renderHook(() => useBrainUpload(null), { wrapper: Wrapper });

      await act(async () => {
        await result.current.upload(makeParams());
      });

      expect(mockAddFiles).not.toHaveBeenCalled();
      expect(mockPollDatasetStatus).not.toHaveBeenCalled();
    });
  });

  describe("when files exceed MAX_FILES_PER_UPLOAD", () => {
    it("calls onLimitExceeded and returns early without calling rememberData", async () => {
      const oversizedFiles = Array.from({ length: MAX_FILES_PER_UPLOAD + 1 }, (_, i) =>
        makeFile(`file-${i}.pdf`),
      );
      const onLimitExceeded = jest.fn();
      const onUploaded = jest.fn();

      const { result } = renderHook(() => useBrainUpload(instance), { wrapper: Wrapper });

      await act(async () => {
        await result.current.upload(makeParams({ files: oversizedFiles, onLimitExceeded, onUploaded }));
      });

      expect(onLimitExceeded).toHaveBeenCalledWith(oversizedFiles);
      expect(mockAddFiles).not.toHaveBeenCalled();
      expect(onUploaded).not.toHaveBeenCalled();
    });
  });

  describe("happy path — rememberData and pollDatasetStatus both succeed", () => {
    it("calls onUploaded after rememberData resolves and onProcessed after polling completes", async () => {
      const onUploaded = jest.fn();
      const onProcessed = jest.fn();
      const onUploadError = jest.fn();
      const onProcessingError = jest.fn();

      const { result } = renderHook(() => useBrainUpload(instance), { wrapper: Wrapper });

      await act(async () => {
        await result.current.upload(makeParams({ onUploaded, onProcessed, onUploadError, onProcessingError }));
      });

      expect(onUploaded).toHaveBeenCalledTimes(1);
      expect(onProcessed).toHaveBeenCalledTimes(1);
      expect(onUploadError).not.toHaveBeenCalled();
      expect(onProcessingError).not.toHaveBeenCalled();
    });

    it("invalidates the dataset-status query with the tenant-scoped key after polling completes", async () => {
      const { result } = renderHook(() => useBrainUpload(instance), { wrapper: Wrapper });

      await act(async () => {
        await result.current.upload(makeParams());
      });

      expect(mockInvalidateQueries).toHaveBeenCalledWith({
        queryKey: MOCK_QUERY_KEY,
      });
      expect(mockDatasetStatusQueryKey).toHaveBeenCalledWith(TENANT_ID);
    });
  });

  describe("when rememberData rejects (upload phase failure — CLO-219)", () => {
    it("calls onUploadError with the error and does not call pollDatasetStatus or onProcessed", async () => {
      const uploadError = new Error("network timeout");
      mockAddFiles.mockRejectedValue(uploadError);

      const onUploadError = jest.fn();
      const onProcessed = jest.fn();
      const onProcessingError = jest.fn();

      const { result } = renderHook(() => useBrainUpload(instance), { wrapper: Wrapper });

      await act(async () => {
        await result.current.upload(makeParams({ onUploadError, onProcessed, onProcessingError }));
      });

      expect(onUploadError).toHaveBeenCalledWith(uploadError, expect.objectContaining({ datasetId: DATASET_ID }));
      expect(mockPollDatasetStatus).not.toHaveBeenCalled();
      expect(onProcessed).not.toHaveBeenCalled();
      expect(onProcessingError).not.toHaveBeenCalled();
    });

    it("sets isUploading back to false after the upload error", async () => {
      mockAddFiles.mockRejectedValue(new Error("upload failed"));

      const { result } = renderHook(() => useBrainUpload(instance), { wrapper: Wrapper });

      await act(async () => {
        await result.current.upload(makeParams());
      });

      expect(result.current.isUploading).toBe(false);
    });
  });

  describe("when rememberData resolves but pollDatasetStatus rejects (build phase failure — CLO-219)", () => {
    it("calls onUploaded then onProcessingError (NOT onUploadError)", async () => {
      const buildError = new Error("graph build failed");
      mockPollDatasetStatus.mockRejectedValue(buildError);

      const onUploaded = jest.fn();
      const onUploadError = jest.fn();
      const onProcessingError = jest.fn();

      const { result } = renderHook(() => useBrainUpload(instance), { wrapper: Wrapper });

      await act(async () => {
        await result.current.upload(makeParams({ onUploaded, onUploadError, onProcessingError }));
      });

      expect(onUploaded).toHaveBeenCalledTimes(1);
      expect(onProcessingError).toHaveBeenCalledWith(
        buildError,
        expect.objectContaining({ datasetId: DATASET_ID }),
      );
      expect(onUploadError).not.toHaveBeenCalled();
    });

    it("still invalidates the dataset-status query even when polling rejects", async () => {
      mockPollDatasetStatus.mockRejectedValue(new Error("build timed out"));

      const { result } = renderHook(() => useBrainUpload(instance), { wrapper: Wrapper });

      await act(async () => {
        await result.current.upload(makeParams());
      });

      expect(mockInvalidateQueries).toHaveBeenCalledWith({
        queryKey: MOCK_QUERY_KEY,
      });
    });

    it("sets isUploading back to false after the processing error", async () => {
      mockPollDatasetStatus.mockRejectedValue(new Error("build timed out"));

      const { result } = renderHook(() => useBrainUpload(instance), { wrapper: Wrapper });

      await act(async () => {
        await result.current.upload(makeParams());
      });

      expect(result.current.isUploading).toBe(false);
    });
  });

  describe("add + cognify split", () => {
    it("uploads with addFiles and does NOT forward cognify options to it", async () => {
      const options = { graphModel: { title: "MyModel" }, chunkSize: 512 };
      const { result } = renderHook(() => useBrainUpload(instance), { wrapper: Wrapper });
      const files = [makeFile()];

      await act(async () => {
        await result.current.upload(makeParams({ files, options }));
      });

      // addFiles is upload-only: it must not carry graphModel/chunkSize or a
      // runInBackground flag — those belong to the cognify step.
      expect(mockAddFiles).toHaveBeenCalledWith(
        expect.anything(),
        files,
        instance,
        expect.not.objectContaining({ graphModel: expect.anything(), runInBackground: expect.anything() }),
      );
    });

    it("triggers cognifyDataset exactly once, after uploading, with the dataset options", async () => {
      const options = { graphModel: { title: "MyModel" }, chunkSize: 512 };
      const { result } = renderHook(() => useBrainUpload(instance), { wrapper: Wrapper });

      await act(async () => {
        await result.current.upload(makeParams({ files: [makeFile(), makeFile("b.pdf")], options }));
      });

      expect(mockCognifyDataset).toHaveBeenCalledTimes(1);
      expect(mockCognifyDataset).toHaveBeenCalledWith(
        expect.objectContaining({ id: DATASET_ID }),
        instance,
        options,
      );
      // Order: every add resolves before the single cognify fires.
      expect(mockAddFiles.mock.invocationCallOrder[0]).toBeLessThan(
        mockCognifyDataset.mock.invocationCallOrder[0],
      );
    });

    it("does not cognify when the upload phase fails", async () => {
      mockAddFiles.mockRejectedValue(new Error("upload failed"));
      const { result } = renderHook(() => useBrainUpload(instance), { wrapper: Wrapper });

      await act(async () => {
        await result.current.upload(makeParams());
      });

      expect(mockCognifyDataset).not.toHaveBeenCalled();
      expect(mockPollDatasetStatus).not.toHaveBeenCalled();
    });

    it("surfaces a cognify-trigger failure as a processing error, not an upload error", async () => {
      mockCognifyDataset.mockRejectedValue(new Error("cognify refused"));
      const onUploaded = jest.fn();
      const onUploadError = jest.fn();
      const onProcessingError = jest.fn();
      const { result } = renderHook(() => useBrainUpload(instance), { wrapper: Wrapper });

      await act(async () => {
        await result.current.upload(makeParams({ onUploaded, onUploadError, onProcessingError }));
      });

      // The files landed, so the upload succeeded; only the build failed.
      expect(onUploaded).toHaveBeenCalledTimes(1);
      expect(onUploadError).not.toHaveBeenCalled();
      expect(onProcessingError).toHaveBeenCalledTimes(1);
      expect(mockPollDatasetStatus).not.toHaveBeenCalled();
    });

    it("passes target as { id, name } when datasetName is provided", async () => {
      const { result } = renderHook(() => useBrainUpload(instance), { wrapper: Wrapper });

      await act(async () => {
        await result.current.upload(makeParams({ datasetName: "My Dataset" }));
      });

      expect(mockAddFiles).toHaveBeenCalledWith(
        { id: DATASET_ID, name: "My Dataset" },
        expect.anything(),
        instance,
        expect.anything(),
      );
    });

    it("passes target as { id } only when datasetName is not provided", async () => {
      const { result } = renderHook(() => useBrainUpload(instance), { wrapper: Wrapper });

      await act(async () => {
        await result.current.upload(makeParams({ datasetName: undefined }));
      });

      expect(mockAddFiles).toHaveBeenCalledWith(
        { id: DATASET_ID },
        expect.anything(),
        instance,
        expect.anything(),
      );
    });
  });

  describe("isUploading state transitions", () => {
    it("starts false, becomes true during the upload, and returns to false on success", async () => {
      let resolveAdd!: (r: AddFilesResponse) => void;
      mockAddFiles.mockReturnValue(
        new Promise<AddFilesResponse>(res => {
          resolveAdd = res;
        }),
      );

      const { result } = renderHook(() => useBrainUpload(instance), { wrapper: Wrapper });

      expect(result.current.isUploading).toBe(false);

      // Use a synchronous act() so that the setIsUploading(true) call — which
      // runs before the first await inside upload() — is flushed immediately.
      // Capturing the returned Promise lets us await it later to drain the rest
      // of the async flow.
      let uploadPromise!: Promise<void>;
      act(() => {
        uploadPromise = result.current.upload(makeParams());
      });

      // State update is flushed synchronously by act().
      expect(result.current.isUploading).toBe(true);

      // Complete the upload and let all state updates settle.
      await act(async () => {
        resolveAdd(makeAddResponse());
        await uploadPromise;
      });

      expect(result.current.isUploading).toBe(false);
    });

    it("returns isUploading to false after an upload error", async () => {
      mockAddFiles.mockRejectedValue(new Error("upload error"));

      const { result } = renderHook(() => useBrainUpload(instance), { wrapper: Wrapper });

      await act(async () => {
        await result.current.upload(makeParams());
      });

      expect(result.current.isUploading).toBe(false);
    });
  });

  describe("stage transitions", () => {
    it("starts idle, moves to uploading then processing, and returns to idle on success", async () => {
      let resolveAdd!: (r: AddFilesResponse) => void;
      mockAddFiles.mockReturnValue(
        new Promise<AddFilesResponse>(res => {
          resolveAdd = res;
        }),
      );
      let resolvePoll!: (status: string) => void;
      mockPollDatasetStatus.mockReturnValue(
        new Promise<string>(res => {
          resolvePoll = res;
        }),
      );

      const { result } = renderHook(() => useBrainUpload(instance), { wrapper: Wrapper });

      expect(result.current.stage).toBe("idle");

      let uploadPromise!: Promise<void>;
      act(() => {
        uploadPromise = result.current.upload(makeParams());
      });
      expect(result.current.stage).toBe("uploading");

      await act(async () => {
        resolveAdd(makeAddResponse());
        // Let the microtask queue drain so setStage("processing") flushes
        // before the assertion below.
        await Promise.resolve();
      });
      expect(result.current.stage).toBe("processing");

      await act(async () => {
        resolvePoll("DATASET_PROCESSING_COMPLETED");
        await uploadPromise;
      });
      expect(result.current.stage).toBe("idle");
    });

    it("returns stage to idle after an upload error", async () => {
      mockAddFiles.mockRejectedValue(new Error("upload error"));

      const { result } = renderHook(() => useBrainUpload(instance), { wrapper: Wrapper });

      await act(async () => {
        await result.current.upload(makeParams());
      });

      expect(result.current.stage).toBe("idle");
    });

    it("returns stage to idle after a processing error", async () => {
      mockPollDatasetStatus.mockRejectedValue(new Error("build failed"));

      const { result } = renderHook(() => useBrainUpload(instance), { wrapper: Wrapper });

      await act(async () => {
        await result.current.upload(makeParams());
      });

      expect(result.current.stage).toBe("idle");
    });
  });
});
