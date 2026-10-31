import type { ReactNode } from "react";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

const mockUseCogniInstance = jest.fn();
const mockUseTenant = jest.fn();
jest.mock("@/modules/tenant/TenantProvider", () => ({
  useCogniInstance: () => mockUseCogniInstance(),
  useTenant: () => mockUseTenant(),
}));

import { useDatasetStatuses } from "../useDatasetStatuses";

const mockFetch = jest.fn();

function Wrapper({ children }: { children: ReactNode }) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

beforeEach(() => {
  jest.clearAllMocks();
  mockUseCogniInstance.mockReturnValue({ cogniInstance: { fetch: mockFetch } });
  mockFetch.mockResolvedValue({ ok: true, json: async () => ({}) });
});

describe("useDatasetStatuses", () => {
  it("does not poll the pod while the workspace isn't ready", async () => {
    mockUseTenant.mockReturnValue({ tenant: { tenant_id: "t1" }, tenantReady: false });

    renderHook(() => useDatasetStatuses(true), { wrapper: Wrapper });

    // Give any (incorrect) fetch a tick to fire, then assert it never did —
    // a direct URL/bookmark to a pod-dependent page must not hammer an
    // unreachable pod every 5s just because the sidebar hides its own link.
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(mockFetch).not.toHaveBeenCalled();
  });

  it("polls once the workspace is ready", async () => {
    mockUseTenant.mockReturnValue({ tenant: { tenant_id: "t1" }, tenantReady: true });

    renderHook(() => useDatasetStatuses(true), { wrapper: Wrapper });

    await waitFor(() =>
      expect(mockFetch).toHaveBeenCalledWith("/v1/datasets/status?include_error_detail=true"),
    );
  });

  it("normalizes a bare-status response (pod predating CLO-306) into statusDetails with a null reason", async () => {
    mockUseTenant.mockReturnValue({ tenant: { tenant_id: "t1" }, tenantReady: true });
    mockFetch.mockResolvedValue({ ok: true, json: async () => ({ "ds-1": "DATASET_PROCESSING_ERRORED" }) });

    const { result } = renderHook(() => useDatasetStatuses(true), { wrapper: Wrapper });

    await waitFor(() => expect(result.current.statuses["ds-1"]).toBe("DATASET_PROCESSING_ERRORED"));
    expect(result.current.statusDetails["ds-1"]).toEqual({ status: "DATASET_PROCESSING_ERRORED", reason: null });
  });

  it("surfaces the reason from a detailed response (pod with CLO-306) in statusDetails", async () => {
    mockUseTenant.mockReturnValue({ tenant: { tenant_id: "t1" }, tenantReady: true });
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({
        "ds-1": { status: "DATASET_PROCESSING_ERRORED", reason: "insufficient_credits", error: "Budget has been exceeded!" },
      }),
    });

    const { result } = renderHook(() => useDatasetStatuses(true), { wrapper: Wrapper });

    await waitFor(() => expect(result.current.statusDetails["ds-1"]?.reason).toBe("insufficient_credits"));
    expect(result.current.statuses["ds-1"]).toBe("DATASET_PROCESSING_ERRORED");
  });
});
