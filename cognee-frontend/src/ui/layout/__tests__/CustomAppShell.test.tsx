import { render, screen } from "@testing-library/react";

const mockUseTenant = jest.fn();
const mockUseCogniInstance = jest.fn();
const mockUsePathname = jest.fn();

jest.mock("next/navigation", () => ({ usePathname: () => mockUsePathname() }));
jest.mock("@/modules/tenant/TenantProvider", () => ({
  useTenant: () => mockUseTenant(),
  useCogniInstance: () => mockUseCogniInstance(),
}));
jest.mock("../TopBar", () => ({ __esModule: true, default: () => <div>top bar</div> }));
jest.mock("../Navbar/CustomAppShellNavbar", () => ({ __esModule: true, default: () => <div>navbar</div> }));
jest.mock("@/ui/elements/PageLoading", () => ({ __esModule: true, default: () => <div>page loading</div> }));
jest.mock("../WorkspaceWaking", () => ({
  __esModule: true,
  default: ({ workspaceName }: { workspaceName?: string }) => <div>waking screen for {workspaceName}</div>,
}));
jest.mock("../WorkspaceProvisioning", () => ({ __esModule: true, default: () => <div>provisioning screen</div> }));
jest.mock("../ProvisioningBanner", () => ({
  __esModule: true,
  default: ({ waking }: { waking?: boolean }) => <div>banner {waking ? "waking" : "provisioning"}</div>,
}));
jest.mock("@/ui/elements/PodUnreachableCard", () => ({
  __esModule: true,
  default: ({ workspaceName, wakeFailed }: { workspaceName?: string; wakeFailed?: boolean }) => (
    <div>{wakeFailed ? "wake failed" : "unreachable"} for {workspaceName ?? "(unnamed)"}</div>
  ),
}));

import CustomAppShell, { POD_DEPENDENT_PATHS } from "../CustomAppShell";

const POD_ROUTE = "/datasets";
const PLAIN_ROUTE = "/settings";

function setup({
  pathname = POD_ROUTE,
  wakeState = "idle",
  tenantReady = false,
  podUnreachable = false,
  isInitializing = false,
}: {
  pathname?: string;
  wakeState?: "idle" | "waking" | "failed";
  tenantReady?: boolean;
  podUnreachable?: boolean;
  isInitializing?: boolean;
} = {}) {
  mockUsePathname.mockReturnValue(pathname);
  mockUseCogniInstance.mockReturnValue({ cogniInstance: { name: "t" }, statusMessage: null, isInitializing });
  mockUseTenant.mockReturnValue({
    tenantReady,
    podUnreachable,
    wakeState,
    wakeStartedAt: null,
    tenant: { tenant_id: "t1", tenant_name: "stale name" },
    currentName: "Research",
  });
  render(<CustomAppShell><div>page content</div></CustomAppShell>);
}

// The branch selection is the whole point of this file and the only part of it
// no leaf-component test can see.
describe("CustomAppShell", () => {
  afterEach(() => {
    jest.clearAllMocks();
  });

  it("shows the waking screen, named, while a pod-dependent route waits on a wake", () => {
    setup({ wakeState: "waking" });
    expect(screen.getByText("waking screen for Research")).toBeInTheDocument();
    expect(screen.queryByText("provisioning screen")).not.toBeInTheDocument();
  });

  it("falls back to the provisioning screen when the pod is pending for any other reason", () => {
    setup({ wakeState: "idle" });
    expect(screen.getByText("provisioning screen")).toBeInTheDocument();
    expect(screen.queryByText(/waking screen/)).not.toBeInTheDocument();
  });

  it("drops the banner where the waking screen already says it", () => {
    setup({ wakeState: "waking" });
    expect(screen.queryByText(/^banner/)).not.toBeInTheDocument();
  });

  it("keeps the banner, in wake wording, on a route with no waking screen", () => {
    setup({ pathname: PLAIN_ROUTE, wakeState: "waking" });
    // These routes have nothing to replace the banner with, so suppressing it
    // there left the user waiting out a wake with no indication at all.
    expect(screen.getByText("banner waking")).toBeInTheDocument();
    expect(screen.getByText("page content")).toBeInTheDocument();
  });

  // The new onboarding window renders OverviewPage, which brings its own
  // skeleton and waking screen, so the shell must treat it like /dashboard.
  it("lets the new onboarding route show its own skeleton while the tenant initializes", () => {
    setup({ pathname: "/onboarding/window", isInitializing: true });
    expect(screen.getByText("page content")).toBeInTheDocument();
    expect(screen.queryByText("page loading")).not.toBeInTheDocument();
  });

  it("still shows the page loading screen on other routes while the tenant initializes", () => {
    setup({ pathname: PLAIN_ROUTE, isInitializing: true });
    expect(screen.getByText("page loading")).toBeInTheDocument();
  });

  it("drops the banner on the new onboarding route, where the waking screen already says it", () => {
    setup({ pathname: "/onboarding/window", wakeState: "waking" });
    expect(screen.queryByText(/^banner/)).not.toBeInTheDocument();
  });

  it("names the workspace on the failure card, not only on the dashboard", () => {
    setup({ wakeState: "failed", podUnreachable: true });
    expect(screen.getByText("wake failed for Research")).toBeInTheDocument();
  });

  it("prefers the tenant list's name over the locally cached one", () => {
    setup({ wakeState: "waking" });
    expect(screen.queryByText(/stale name/)).not.toBeInTheDocument();
  });

  // The bug: `useDashboardTelemetry`'s query is gated on `tenantReady`, so on a
  // suspended or unreachable workspace it never runs and `/analytics` rendered
  // an eternal skeleton with no error and no way out — its own page had no
  // guard, and the shell didn't gate the route either. Pinning both: the path
  // is listed, and the shell actually withholds the page for it.
  it("lists /analytics as pod-dependent", () => {
    expect(POD_DEPENDENT_PATHS).toContain("/analytics");
  });

  it("shows the unreachable card on /analytics instead of an eternal skeleton", () => {
    setup({ pathname: "/analytics", wakeState: "failed", podUnreachable: true });
    expect(screen.getByText("wake failed for Research")).toBeInTheDocument();
    expect(screen.queryByText("page content")).not.toBeInTheDocument();
  });
});

describe("POD_DEPENDENT_PATHS membership", () => {
  it("gates every route whose whole content comes from the pod", () => {
    // Asserted by name because the test's own POD_ROUTE is a single example, so
    // a route could be dropped from the list and nothing would fail. /activity
    // was missing, and a disabled react-query stays pending forever, so a dead
    // pod left that route on a permanent skeleton with no error and no card.
    for (const route of ["/search", "/datasets", "/skills", "/knowledge-graph", "/sessions", "/graph-models", "/activity"]) {
      expect(POD_DEPENDENT_PATHS).toContain(route);
    }
  });

  it("does not gate the routes documented as exceptions", () => {
    // Dashboard has its own skeleton; API Keys shows the tenant URL early.
    expect(POD_DEPENDENT_PATHS).not.toContain("/dashboard");
    expect(POD_DEPENDENT_PATHS).not.toContain("/api-keys");
    expect(POD_DEPENDENT_PATHS).not.toContain("/settings");
  });
});
