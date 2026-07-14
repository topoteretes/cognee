export {};

const cookieJar = new Map<string, { name: string; value: string }>();

jest.mock("next/headers", () => ({
  cookies: async () => ({
    get: (name: string) => cookieJar.get(name),
  }),
}));

const mockManagementFetch = jest.fn();
jest.mock("../managementFetch", () => ({
  __esModule: true,
  default: (...args: unknown[]) => mockManagementFetch(...args),
}));

const mockGetMyTenants = jest.fn();
jest.mock("@/modules/tenant/getMyTenants", () => ({
  __esModule: true,
  default: (...args: unknown[]) => mockGetMyTenants(...args),
}));

const jsonResponse = (body: unknown) => ({ json: async () => body }) as Response;

function mockManagementResponses(): void {
  mockManagementFetch.mockImplementation((path: string) => {
    if (path === "/tenants/current/service-url") {
      return Promise.resolve(jsonResponse({ service_url: "http://tenant-primary.pod.example.com" }));
    }
    if (path === "/api-keys") {
      return Promise.resolve(jsonResponse([{ key: "user-owned-api-key" }]));
    }
    if (path === "/tenants/current") {
      return Promise.resolve(jsonResponse({ tenant_id: "primary-tenant" }));
    }
    throw new Error(`unexpected managementFetch call: ${path}`);
  });
}

async function loadGetPodContext() {
  const mod = await import("../getPodContext");
  return mod.default;
}

beforeEach(() => {
  jest.resetModules();
  cookieJar.clear();
  jest.clearAllMocks();
  mockManagementResponses();
});

describe("getPodContext", () => {
  it("ignores a selected-tenant cookie pointing at a tenant the user does not belong to", async () => {
    cookieJar.set("cognee_selected_tenant", { name: "cognee_selected_tenant", value: "other-users-tenant" });
    mockGetMyTenants.mockResolvedValue([{ id: "primary-tenant", name: "Personal", isOwner: true, ownerHasSubscription: true }]);
    const getPodContext = await loadGetPodContext();

    const result = await getPodContext();

    expect(result.tenant_id).toBe("primary-tenant");
    expect(result.base).toContain("tenant-primary");
    expect(result.apiKey).toBe("user-owned-api-key");
  });

  it("honors a selected-tenant cookie when the user is a member of that tenant", async () => {
    cookieJar.set("cognee_selected_tenant", { name: "cognee_selected_tenant", value: "shared-tenant" });
    mockGetMyTenants.mockResolvedValue([
      { id: "primary-tenant", name: "Personal", isOwner: true, ownerHasSubscription: true },
      { id: "shared-tenant", name: "Shared", isOwner: false, ownerHasSubscription: true },
    ]);
    const getPodContext = await loadGetPodContext();

    const result = await getPodContext();

    expect(result.tenant_id).toBe("shared-tenant");
    expect(result.base).toContain("tenant-shared-tenant");
  });

  it("does not call getMyTenants when no tenant switch is requested", async () => {
    const getPodContext = await loadGetPodContext();

    const result = await getPodContext();

    expect(result.tenant_id).toBe("primary-tenant");
    expect(mockGetMyTenants).not.toHaveBeenCalled();
  });

  it("resolves fresh on every call instead of reusing a previous caller's tenant/apiKey", async () => {
    // Regression test for a cross-request cache that used to skip
    // managementFetch (the actual per-request authentication) whenever the
    // caller had no cognee_selected_tenant cookie, returning whichever
    // tenant/apiKey a *previous* caller had resolved.
    const getPodContext = await loadGetPodContext();

    await getPodContext();
    cookieJar.clear();
    mockManagementFetch.mockClear();
    mockManagementResponses();
    await getPodContext();

    expect(mockManagementFetch).toHaveBeenCalledWith("/tenants/current/service-url");
    expect(mockManagementFetch).toHaveBeenCalledWith("/api-keys");
    expect(mockManagementFetch).toHaveBeenCalledWith("/tenants/current");
  });
});
