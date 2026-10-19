import { render, screen, waitFor } from "@testing-library/react";
import { LocalProvider } from "../LocalProvider";
import { useTenant } from "../TenantContext";
import { useUser } from "@/modules/users/UserContext";

// Local ownership follows the authenticated server account, including invited members.

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

  it.each([true, false])("uses the server ownership flag %s", async (isOwner) => {
    fetchMock.mockResponses(
      JSON.stringify({ id: "person" }),
      JSON.stringify({ user: { id: "person", tenant_id: "team" }, teams: [{ id: "team", name: "Engineering", is_owner: isOwner }] }),
    );

    render(
      <LocalProvider>
        <Probe />
      </LocalProvider>,
    );

    await waitFor(() => {
      expect(screen.getByTestId("tenant-id")).toHaveTextContent("team");
    });
    expect(screen.getByTestId("is-owner")).toHaveTextContent(String(isOwner));
  });

  it("exposes the local workspace as an owned tenant, so owner-gated lists are not empty", async () => {
    fetchMock.mockResponses(
      JSON.stringify({ id: "person" }),
      JSON.stringify({ user: { id: "person", tenant_id: "team" }, teams: [{ id: "team", name: "Engineering", is_owner: true }] }),
    );

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
