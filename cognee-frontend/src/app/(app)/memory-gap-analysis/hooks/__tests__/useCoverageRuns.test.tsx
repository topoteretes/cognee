import { act, renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { useCoverageRuns } from "../useCoverageRuns";

const mockFetch = jest.fn();
const mockInstance = { fetch: mockFetch };
const mockDatasets = [{ id: "brain", name: "Test" }];
const mockRefresh = jest.fn();
let mockCloud = false;
jest.mock("@/utils/isCloudEnvironment", () => ({ __esModule: true, default: () => mockCloud }));
jest.mock("@/modules/analytics", () => ({ trackEvent: jest.fn() }));
jest.mock("@/modules/tenant/TenantProvider", () => ({
  useCogniInstance: () => ({ cogniInstance: mockInstance, isInitializing: false }),
  useTenant: () => ({ tenantReady: true }),
}));
jest.mock("@/ui/layout/FilterContext", () => ({
  useFilter: () => ({ datasets: mockDatasets, loading: false, datasetsUnavailable: false, refreshDatasets: mockRefresh }),
}));

function setup() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return renderHook(() => useCoverageRuns(), {
    wrapper: ({ children }: { children: ReactNode }) => <QueryClientProvider client={client}>{children}</QueryClientProvider>,
  });
}

it("never calls Cloud coverage endpoints locally, including refresh and start", async () => {
  mockCloud = false;
  const { result } = setup();
  await act(async () => {
    result.current.refresh();
    await result.current.startRun("brain");
  });
  expect(mockFetch).not.toHaveBeenCalled();
  expect(mockRefresh).not.toHaveBeenCalled();
  expect(result.current.runError?.message).toContain("Cognee Cloud");
});

it("continues loading coverage in Cloud", async () => {
  mockCloud = true;
  mockFetch.mockResolvedValue({ json: async () => null });
  const { result } = setup();
  await waitFor(() => expect(result.current.brains[0].state).toBe("loaded"));
  expect(mockFetch).toHaveBeenCalledWith("/v1/datasets/brain/coverage", expect.any(Object));
});
