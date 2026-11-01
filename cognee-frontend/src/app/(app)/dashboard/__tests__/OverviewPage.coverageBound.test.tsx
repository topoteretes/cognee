import React from "react";
import { render } from "@testing-library/react";

/**
 * Pins the one thing that bounds the dashboard's coverage fetch: the argument
 * at the call site (CLO-759).
 *
 * The hook's own suite proves it honours `limit`, and `coverageView` proves the
 * derivation. Neither notices if the page stops passing the option, and dropping
 * it costs three regressions at once: the request-per-dataset fan-out returns,
 * `PerformancePanel` renders a row per dataset in a panel sized for four
 * (nothing slices `topics` any more), and the "Showing N of M" disclosure goes
 * silent because `datasetTotal > topics.length` stops being true. The full suite
 * stayed green through all three.
 *
 * The page renders behind `podUnreachable`, which returns before any panel but
 * after every hook, so this needs the hooks above that return and nothing below
 * it. Hooks are mocked by module, and `useCoverageRuns` forwards its arguments
 * rather than swallowing them.
 */

const mockUseCoverageRuns = jest.fn();
jest.mock("@/app/(app)/memory-gap-analysis/hooks/useCoverageRuns", () => ({
  useCoverageRuns: (...args: unknown[]) => mockUseCoverageRuns(...args),
}));

jest.mock("next/navigation", () => ({ useRouter: () => ({ push: jest.fn() }) }));
jest.mock("@mantine/notifications", () => ({ notifications: { show: jest.fn() } }));
jest.mock("@/modules/analytics", () => ({ trackEvent: jest.fn() }));
jest.mock("@/ui/elements/PodUnreachableCard", () => ({
  __esModule: true,
  default: () => <div data-testid="pod-unreachable" />,
}));
jest.mock("@/modules/tenant/TenantProvider", () => ({
  useCogniInstance: () => ({ cogniInstance: null, isInitializing: false, serviceUrl: "", apiKey: "" }),
  // The terminal state, so the page returns before rendering any panel.
  useTenant: () => ({
    tenantReady: true,
    podUnreachable: true,
    wakeState: "ready",
    wakeStartedAt: null,
    tenant: null,
    isOwner: false,
    currentName: "acme",
  }),
}));
jest.mock("@/ui/layout/FilterContext", () => ({
  useFilter: () => ({
    agents: [],
    datasets: [],
    selectedAgent: null,
    loading: false,
    agentsLoading: false,
    agentsError: null,
  }),
  useRefreshDatasetsOnMount: () => undefined,
}));
jest.mock("@/modules/users/useCurrentUser", () => ({ useCurrentUser: () => ({ data: null }) }));
jest.mock("@/modules/billing/useTenantHourlyCosts", () => ({ useTenantHourlyCosts: () => ({ data: null }) }));
jest.mock("@/app/(app)/dashboard/hooks/useDashboardTelemetry", () => ({
  useDashboardTelemetry: () => ({
    runs: [],
    sessions: [],
    connections: [],
    loading: false,
    error: null,
    sessionsFailed: false,
    updatedAt: null,
    runsTruncated: false,
    sessionsTruncated: false,
  }),
}));
jest.mock("@/app/(app)/dashboard/hooks/useConnectedIntegrations", () => ({ useConnectedIntegrations: () => [] }));
jest.mock("@/app/(app)/dashboard/hooks/useDataSourceStatuses", () => ({ useDataSourceStatuses: () => ({}) }));
jest.mock("@/app/(app)/dashboard/hooks/useCreditsBanner", () => ({ useCreditsBanner: () => ({}) }));
jest.mock("@/app/(app)/dashboard/hooks/useAwaitingDataset", () => ({ useAwaitingDataset: () => false }));
jest.mock("@/app/(app)/dashboard/hooks/useDatasetUpload", () => ({
  useDatasetUpload: () => ({ uploading: false, upload: jest.fn() }),
}));
jest.mock("@/app/(app)/dashboard/hooks/useChecklistCompletionTracking", () => ({
  useChecklistCompletionTracking: () => undefined,
}));

import OverviewPage from "@/app/(app)/dashboard/OverviewPage";

describe("OverviewPage coverage fetch bound", () => {
  beforeEach(() => {
    mockUseCoverageRuns.mockReset();
    mockUseCoverageRuns.mockReturnValue({ brains: [], loading: false, refresh: jest.fn() });
  });

  it("asks for coverage on only the datasets it renders", () => {
    render(<OverviewPage />);

    expect(mockUseCoverageRuns).toHaveBeenCalledWith({ limit: 4 });
  });

  it("never asks for the whole workspace", () => {
    // The unbounded call is `useCoverageRuns()`, which is what Memory Gap
    // Analysis needs and what the dashboard paid for before CLO-759.
    render(<OverviewPage />);

    for (const call of mockUseCoverageRuns.mock.calls) {
      expect(call).toHaveLength(1);
      expect(call[0]).toEqual(expect.objectContaining({ limit: expect.any(Number) }));
    }
  });
});
