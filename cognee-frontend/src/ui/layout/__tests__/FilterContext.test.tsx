import type { ReactNode } from "react";
import { act, renderHook, waitFor } from "@testing-library/react";
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

const mockGetAgents = jest.fn();
jest.mock("@/modules/agents/getAgents", () => ({
  __esModule: true,
  default: (...args: unknown[]) => mockGetAgents(...args),
}));

import { FilterProvider, useFilter } from "../FilterContext";

// One client per test, created OUTSIDE the render body. Building it inline
// swapped the cache on every re-render, so nothing that depends on cache
// identity (an invalidation, a refetch) could be asserted reliably.
let client: QueryClient;
beforeEach(() => {
  // staleTime Infinity, so the ONLY thing that can trigger a refetch is an
  // explicit invalidation. With the default 0 any re-render refetches, which
  // made the refreshAgents test pass even against a mismatched query key.
  client = new QueryClient({
    defaultOptions: { queries: { retry: false, staleTime: Infinity } },
  });
});

function Wrapper({ children }: { children: ReactNode }) {
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
  mockGetAgents.mockResolvedValue([]);
});

const READY_TENANT = {
  tenant: { tenant_id: "tenant-1", tenant_name: "Personal Workspace" },
  tenantReady: true,
  availableTenants: [{ id: "tenant-1", name: "Personal Workspace", isOwner: true, ownerHasSubscription: true }],
  switchTenant: jest.fn(),
};

const AGENT = {
  id: "agent-1",
  email: "claude-code+3333@cognee.agent",
  agent_type: "claude code",
  agent_short_id: "3333",
  is_agent: true,
  is_default: false,
  status: "LIVE",
};

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

  it("does not show a persisted selection that belongs to another account once this account's workspaces are known", async () => {
    // A login without a sign-out keeps localStorage, so the previous
    // account's workspace name would otherwise sit in the topbar until
    // tenant init overwrote it.
    localStorage.setItem("cognee_selected_tenant", "other-account-tenant");
    localStorage.setItem("cognee_selected_tenant_name", "Someone Else's Workspace");
    mockUseTenant.mockReturnValue({
      tenant: null,
      tenantReady: false,
      availableTenants: [{ id: "tenant-1", name: "Personal Workspace", isOwner: true, ownerHasSubscription: true }],
      switchTenant: jest.fn(),
    });

    const { result } = renderHook(() => useFilter(), { wrapper: Wrapper });

    await waitFor(() => expect(result.current.workspace.name).toBe("Personal Workspace"));
  });
});

describe("FilterContext agents roster", () => {
  it("fetches the roster once the tenant is ready", async () => {
    // Was dead state — useState<Agent[]>([]) with the setter discarded — so
    // every consumer read an empty roster forever.
    mockUseCogniInstance.mockReturnValue({ cogniInstance: { fetch: jest.fn() }, isInitializing: false });
    mockUseTenant.mockReturnValue(READY_TENANT);
    mockGetAgents.mockResolvedValue([AGENT]);

    const { result } = renderHook(() => useFilter(), { wrapper: Wrapper });

    await waitFor(() => expect(result.current.agents).toHaveLength(1));
    expect(result.current.agents[0].id).toBe("agent-1");
    expect(result.current.agentsLoading).toBe(false);
    expect(result.current.agentsError).toBe(false);
  });

  it("does not fetch while the tenant is still resolving, and says so", async () => {
    // agentsLoading has to stay true here: isLoading is already false for a
    // disabled query, which would let a consumer call an empty roster final.
    mockUseCogniInstance.mockReturnValue({ cogniInstance: null, isInitializing: true });
    mockUseTenant.mockReturnValue({ ...READY_TENANT, tenant: null, tenantReady: false });

    const { result } = renderHook(() => useFilter(), { wrapper: Wrapper });

    await waitFor(() => expect(result.current.workspace).toBeDefined());
    expect(mockGetAgents).not.toHaveBeenCalled();
    expect(result.current.agentsLoading).toBe(true);
    expect(result.current.agents).toEqual([]);
  });

  it("refreshAgents invalidates the key the roster query actually uses", async () => {
    // The pair has to match: the queryKey here and the invalidate key in
    // refreshAgents. Nothing asserted either, so mutating one silently made
    // the Members page refresh a no-op.
    mockUseCogniInstance.mockReturnValue({ cogniInstance: { fetch: jest.fn() }, isInitializing: false });
    mockUseTenant.mockReturnValue(READY_TENANT);
    mockGetAgents.mockResolvedValue([AGENT]);

    const { result } = renderHook(() => useFilter(), { wrapper: Wrapper });
    await waitFor(() => expect(result.current.agents).toHaveLength(1));
    expect(mockGetAgents).toHaveBeenCalledTimes(1);

    mockGetAgents.mockResolvedValue([AGENT, { ...AGENT, id: "agent-2" }]);
    act(() => { result.current.refreshAgents(); });

    await waitFor(() => expect(result.current.agents).toHaveLength(2));
    expect(mockGetAgents).toHaveBeenCalledTimes(2);
  });

  it("reports a failed fetch instead of an empty roster", async () => {
    // The distinction ShareDatasetModal needs: "nobody to share with" versus
    // "we could not find out", which used to render identically.
    mockUseCogniInstance.mockReturnValue({ cogniInstance: { fetch: jest.fn() }, isInitializing: false });
    mockUseTenant.mockReturnValue(READY_TENANT);
    mockGetAgents.mockRejectedValue(Object.assign(new Error("not found"), { status: 404 }));

    const { result } = renderHook(() => useFilter(), { wrapper: Wrapper });

    await waitFor(() => expect(result.current.agentsError).toBe(true));
    expect(result.current.agents).toEqual([]);
    expect(result.current.agentsLoading).toBe(false);
  });
});
