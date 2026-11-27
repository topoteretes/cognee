import { render, screen, waitFor } from "@testing-library/react";
import { LocalProvider } from "../LocalProvider";
import { useTenant } from "../TenantContext";

// The backend's access token lasts an hour, so a tab opened later than that
// starts with a 401 from the auth probe. The refresh cookie renews the login;
// only when that fails does the user have to sign in again.

function Probe() {
  const { tenant } = useTenant();
  return <span data-testid="tenant-id">{tenant?.tenant_id ?? "none"}</span>;
}

const requested = () => fetchMock.mock.calls.map(([url]) => String(url).replace(/^.*\/api\/v1/, ""));

describe("LocalProvider login renewal", () => {
  beforeEach(() => {
    // LocalProvider skips its auth probe on the login page.
    window.history.replaceState({}, "", "/dashboard");
    fetchMock.resetMocks();
    jest.spyOn(console, "error").mockImplementation(() => {});
  });
  afterEach(() => jest.restoreAllMocks());

  it("renews an expired login instead of sending the user to sign in", async () => {
    fetchMock.mockResponses(
      ["", { status: 401 }],
      ["", { status: 200 }],
      [JSON.stringify({ id: "local" }), { status: 200 }],
    );

    render(<LocalProvider><Probe /></LocalProvider>);

    await waitFor(() => {
      expect(screen.getByTestId("tenant-id")).toHaveTextContent("local");
    });
    expect(requested()).toEqual(["/users/me", "/auth/refresh", "/users/me"]);
  });

  it("does not set up the workspace when the login cannot be renewed", async () => {
    fetchMock.mockResponses(["", { status: 401 }], ["", { status: 401 }]);

    render(<LocalProvider><Probe /></LocalProvider>);

    await waitFor(() => {
      expect(requested()).toEqual(["/users/me", "/auth/refresh"]);
    });
    expect(screen.getByTestId("tenant-id")).toHaveTextContent("none");
  });
});
