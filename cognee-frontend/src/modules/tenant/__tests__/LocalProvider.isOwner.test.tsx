import { render, screen, waitFor } from "@testing-library/react";
import { LocalProvider } from "../LocalProvider";
import { useTenant } from "../TenantContext";
import { useUser } from "@/modules/users/UserContext";

// Regression guard for a sync revert that shipped twice.
//
// `useTenant()` derives isOwner by looking the active tenant up in
// UserContext's availableTenants. In cloud mode UserProvider fills that list;
// in local mode nothing does, so LocalProvider has to supply it. When the SaaS
// sync dropped that (it deleted LocalProvider's `isOwner: true`), isOwner went
// permanently false and every owner-gated control rendered inert — most
// visibly the Connect buttons on the Data Sources cards, which
// DataSourceCard's `ctaFor` returns null for when !isOwner.
//
// The failure was invisible: it type-checks, it builds, and the only frontend
// CI job renders /local-login, which reads none of this.

function Probe() {
  const { isOwner, tenant } = useTenant();
  const { availableTenants } = useUser();
  return (
    <>
      <span data-testid="is-owner">{String(isOwner)}</span>
      <span data-testid="tenant-id">{tenant?.tenant_id ?? "none"}</span>
      <span data-testid="tenant-count">{availableTenants.length}</span>
      <span data-testid="owned-count">
        {availableTenants.filter((t) => t.isOwner).length}
      </span>
    </>
  );
}

describe("LocalProvider ownership", () => {
  beforeEach(() => {
    // LocalProvider skips its auth probe on the login page, which would leave
    // tenant null; pretend we are anywhere else.
    window.history.replaceState({}, "", "/dashboard");
    fetchMock.resetMocks();
  });

  it("treats the single local user as owner of the local workspace", async () => {
    // The /api/v1/users/me probe LocalProvider runs before it sets the tenant.
    fetchMock.mockResponseOnce(JSON.stringify({ id: "local" }));

    render(
      <LocalProvider>
        <Probe />
      </LocalProvider>,
    );

    await waitFor(() => {
      expect(screen.getByTestId("tenant-id")).toHaveTextContent("local");
    });
    expect(screen.getByTestId("is-owner")).toHaveTextContent("true");
  });

  it("exposes the local workspace as an owned tenant, so owner-gated lists are not empty", async () => {
    fetchMock.mockResponseOnce(JSON.stringify({ id: "local" }));

    render(
      <LocalProvider>
        <Probe />
      </LocalProvider>,
    );

    // DataSourceSection filters availableTenants by isOwner to decide which
    // workspaces a connector can be routed to; an empty list disables the UI.
    await waitFor(() => {
      expect(screen.getByTestId("owned-count")).toHaveTextContent("1");
    });
    expect(screen.getByTestId("tenant-count")).toHaveTextContent("1");
  });

  it("does not claim ownership before the tenant is resolved", () => {
    // Probe never resolves, so tenant stays null. isOwner must stay false
    // rather than defaulting to true, matching the cloud behaviour.
    fetchMock.mockResponseOnce(() => new Promise(() => {}));

    render(
      <LocalProvider>
        <Probe />
      </LocalProvider>,
    );

    expect(screen.getByTestId("tenant-id")).toHaveTextContent("none");
    expect(screen.getByTestId("is-owner")).toHaveTextContent("false");
  });
});
