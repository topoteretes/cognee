import type { ReactNode } from "react";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { PipelineRun } from "@/ui/elements/AgentActivityTerminal";

const mockUseCogniInstance = jest.fn();
const mockUseTenant = jest.fn();
jest.mock("@/modules/tenant/TenantProvider", () => ({
  useCogniInstance: () => mockUseCogniInstance(),
  useTenant: () => mockUseTenant(),
}));

const mockGetDatasetGraphSummary = jest.fn();
jest.mock("@/modules/datasets/getDatasetGraphSummary", () => ({
  __esModule: true,
  default: (...args: unknown[]) => mockGetDatasetGraphSummary(...args),
}));

jest.mock("@/utils/browserStorage", () => ({
  getCachedGraphNodes: () => null,
  setCachedGraphNodes: jest.fn(),
}));

import { useGraphSummary } from "../useGraphSummary";

function makeRun(status: string): PipelineRun {
  return {
    id: "run-1",
    pipeline_run_id: "run-1",
    pipeline_name: "cognify",
    status,
    dataset_id: "dataset-1",
    dataset_name: "Dataset 1",
    owner_email: null,
    created_at: null,
  };
}

function Wrapper({ children }: { children: ReactNode }) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

beforeEach(() => {
  jest.clearAllMocks();
  mockUseCogniInstance.mockReturnValue({ cogniInstance: { fetch: jest.fn() } });
  mockUseTenant.mockReturnValue({ tenantReady: true });
  mockGetDatasetGraphSummary.mockResolvedValue([{ numNodes: 3, numEdges: 2 }]);
});

describe("useGraphSummary — pipeline-completion refetch", () => {
  it("does not treat runs already completed on the first render as a fresh completion", async () => {
    const { result } = renderHook(
      () => useGraphSummary({ id: "dataset-1", name: "Dataset 1" }, [],[makeRun("DATASET_PROCESSING_COMPLETED")]),
      { wrapper: Wrapper },
    );

    await waitFor(() => expect(result.current.graphNodes).toBe(3));
    expect(mockGetDatasetGraphSummary).toHaveBeenCalledTimes(1);
  });

  it("refetches when a run transitions to completed", async () => {
    const { rerender } = renderHook(
      ({ runs }: { runs: PipelineRun[] }) => useGraphSummary({ id: "dataset-1", name: "Dataset 1" }, [],runs),
      { wrapper: Wrapper, initialProps: { runs: [makeRun("DATASET_PROCESSING_STARTED")] } },
    );

    await waitFor(() => expect(mockGetDatasetGraphSummary).toHaveBeenCalledTimes(1));

    rerender({ runs: [makeRun("DATASET_PROCESSING_COMPLETED")] });

    await waitFor(() => expect(mockGetDatasetGraphSummary).toHaveBeenCalledTimes(2));
  });

  it("schedules a second refetch after the run completes to catch up with delayed GraphMetrics", async () => {
    jest.useFakeTimers();
    try {
      const { rerender } = renderHook(
        ({ runs }: { runs: PipelineRun[] }) => useGraphSummary({ id: "dataset-1", name: "Dataset 1" }, [],runs),
        { wrapper: Wrapper, initialProps: { runs: [makeRun("DATASET_PROCESSING_STARTED")] } },
      );

      await waitFor(() => expect(mockGetDatasetGraphSummary).toHaveBeenCalledTimes(1));

      rerender({ runs: [makeRun("DATASET_PROCESSING_COMPLETED")] });
      await waitFor(() => expect(mockGetDatasetGraphSummary).toHaveBeenCalledTimes(2));

      await jest.advanceTimersByTimeAsync(15_000);
      await waitFor(() => expect(mockGetDatasetGraphSummary).toHaveBeenCalledTimes(3));
    } finally {
      jest.useRealTimers();
    }
  });
});
