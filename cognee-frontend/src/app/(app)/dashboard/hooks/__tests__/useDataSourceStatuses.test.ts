const mockGetConnectionStatus = jest.fn();
const mockGetConnectorCatalog = jest.fn();
jest.mock("@/modules/integrations/getConnectionStatus", () => ({
  __esModule: true,
  default: (...args: unknown[]) => mockGetConnectionStatus(...args),
}));
jest.mock("@/modules/integrations/getConnectorCatalog", () => ({
  __esModule: true,
  default: () => mockGetConnectorCatalog(),
}));

import { renderHook, waitFor } from "@testing-library/react";
import { useDataSourceStatuses } from "../useDataSourceStatuses";

const TENANT = "tenant-1";
const PROVIDERS = ["slack", "github", "google_drive"];
const GITHUB = { provider: "github", displayName: "GitHub", configured: true, supportsResourceSelection: true, degraded: false };

function askedProviders(): string[] {
  return mockGetConnectionStatus.mock.calls.map((call) => String(call[0])).sort();
}

describe("useDataSourceStatuses", () => {
  beforeEach(() => {
    mockGetConnectorCatalog.mockReset();
    mockGetConnectionStatus.mockReset().mockResolvedValue({ connected: false, failed: false });
  });

  it("leaves out a connector this deployment does not ship instead of drawing it disconnected forever (review-reported)", async () => {
    mockGetConnectorCatalog.mockResolvedValue([GITHUB]);
    const warn = jest.spyOn(console, "warn").mockImplementation(() => {});

    const { result } = renderHook(() => useDataSourceStatuses(PROVIDERS, TENANT));

    await waitFor(() => expect(result.current.google_drive).toBe("absent"));
    expect(result.current).toEqual({ slack: "disconnected", github: "disconnected", google_drive: "absent" });
    expect(askedProviders()).toEqual(["github", "slack"]);
    expect(warn).not.toHaveBeenCalled();
    warn.mockRestore();
  });

  it("asks for every provider when the catalog cannot be read, as before", async () => {
    mockGetConnectorCatalog.mockResolvedValue(null);
    mockGetConnectionStatus.mockResolvedValue({ connected: true, failed: false });

    const { result } = renderHook(() => useDataSourceStatuses(PROVIDERS, TENANT));

    await waitFor(() => expect(result.current.google_drive).toBe("connected"));
    expect(askedProviders()).toEqual(["github", "google_drive", "slack"]);
  });

  it("counts channels routed in from another workspace as connected", async () => {
    mockGetConnectorCatalog.mockResolvedValue([GITHUB]);
    mockGetConnectionStatus.mockResolvedValue({ connected: false, failed: false, viaRouting: true });

    const { result } = renderHook(() => useDataSourceStatuses(["slack"], TENANT));

    await waitFor(() => expect(result.current.slack).toBe("connected"));
  });

  it("reads nothing without a tenant", () => {
    const { result } = renderHook(() => useDataSourceStatuses(PROVIDERS, null));

    expect(result.current).toEqual({});
    expect(mockGetConnectorCatalog).not.toHaveBeenCalled();
  });
});
