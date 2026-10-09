import persistSelectedTenant from "../persistSelectedTenant";

afterEach(() => {
  localStorage.clear();
  sessionStorage.clear();
  document.cookie = "cognee_selected_tenant=;path=/;max-age=0";
});

describe("persistSelectedTenant", () => {
  it("writes the tenant id to cookie, localStorage, and sessionStorage", () => {
    persistSelectedTenant("tenant-1", "My Workspace");
    expect(document.cookie).toContain("cognee_selected_tenant=tenant-1");
    expect(localStorage.getItem("cognee_selected_tenant")).toBe("tenant-1");
    expect(localStorage.getItem("cognee_selected_tenant_name")).toBe("My Workspace");
    expect(sessionStorage.getItem("cognee_selected_tenant")).toBe("tenant-1");
  });

  it("leaves the stored name untouched when no name is passed", () => {
    localStorage.setItem("cognee_selected_tenant_name", "Old Name");
    persistSelectedTenant("tenant-2");
    expect(localStorage.getItem("cognee_selected_tenant")).toBe("tenant-2");
    expect(localStorage.getItem("cognee_selected_tenant_name")).toBe("Old Name");
  });
});
