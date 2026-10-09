import type { ReactNode } from "react";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

const mockUseCogniInstance = jest.fn();
const mockUseTenant = jest.fn();
jest.mock("@/modules/tenant/TenantProvider", () => ({
  useCogniInstance: () => mockUseCogniInstance(),
  useTenant: () => mockUseTenant(),
}));

const mockGetDatasets = jest.fn();
jest.mock("@/modules/datasets/getDatasets", () => ({
  __esModule: true,
  default: (...args: unknown[]) => mockGetDatasets(...args),
}));

import { FilterProvider, useFilter } from "../FilterContext";

function Wrapper({ children }: { children: ReactNode }) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return (
    <QueryClientProvider client={client}>
      <FilterProvider>{children}</FilterProvider>
    </QueryClientProvider>
  );
}

beforeEach(() => {
  jest.clearAllMocks();
  localStorage.clear();
  mockUseCogniInstance.mockReturnValue({ cogniInstance: null, isInitializing: true });
  mockGetDatasets.mockResolvedValue([]);
});

describe("FilterContext", () => {
  it("shows the persisted workspace selection instead of the hardcoded personal default while tenant is still resolving", async () => {
    // Regression test: tenant init is async (can take seconds), and the
    // workspace state used to start at a hardcoded "Personal workspace"
    // default until it resolved — flashing the wrong workspace in the topbar
    // for every user who was actually on a different one.
    localStorage.setItem("cognee_selected_tenant", "tenant-2");
    localStorage.setItem("cognee_selected_tenant_name", "Workspace 2");
    mockUseTenant.mockReturnValue({
      tenant: null, // not resolved yet
      tenantReady: false,
      availableTenants: [
        { id: "tenant-1", name: "Personal Workspace", isOwner: true, ownerHasSubscription: true },
        { id: "tenant-2", name: "Workspace 2", isOwner: true, ownerHasSubscription: true },
      ],
      switchTenant: jest.fn(),
    });

    const { result } = renderHook(() => useFilter(), { wrapper: Wrapper });

    await waitFor(() => expect(result.current.workspace.name).toBe("Workspace 2"));
  });

  it("uses the resolved tenant once it loads, even if it differs from the persisted selection", async () => {
    localStorage.setItem("cognee_selected_tenant", "tenant-2");
    localStorage.setItem("cognee_selected_tenant_name", "Workspace 2");
    mockUseTenant.mockReturnValue({
      tenant: { tenant_id: "tenant-1", tenant_name: "Personal Workspace" },
      tenantReady: true,
      availableTenants: [
        { id: "tenant-1", name: "Personal Workspace", isOwner: true, ownerHasSubscription: true },
        { id: "tenant-2", name: "Workspace 2", isOwner: true, ownerHasSubscription: true },
      ],
      switchTenant: jest.fn(),
    });

    const { result } = renderHook(() => useFilter(), { wrapper: Wrapper });

    await waitFor(() => expect(result.current.workspace.name).toBe("Personal Workspace"));
  });
});
