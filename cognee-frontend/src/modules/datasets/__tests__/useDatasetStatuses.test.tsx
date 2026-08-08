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

    await waitFor(() => expect(mockFetch).toHaveBeenCalledWith("/v1/datasets/status"));
  });
});
