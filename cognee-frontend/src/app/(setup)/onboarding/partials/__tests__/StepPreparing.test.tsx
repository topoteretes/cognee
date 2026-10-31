interface MockTenantState {
  tenantReady: boolean;
  podUnreachable: boolean;
  wakeState: "idle" | "waking" | "failed";
  currentName?: string;
}

let mockTenantState: MockTenantState;
jest.mock("@/modules/tenant/TenantProvider", () => ({
  useTenant: () => mockTenantState,
  useCogniInstance: () => ({ cogniInstance: { name: "CloudCognee" }, apiKey: "key-123" }),
}));

jest.mock("@/modules/analytics", () => ({ trackEvent: jest.fn() }));
jest.mock("../../useOnboardingTrackEvent", () => ({ useOnboardingTrackEvent: () => jest.fn() }));
jest.mock("@/ui/elements/Auth/TetrisBackground", () => ({ __esModule: true, default: () => null }));
jest.mock("../GraphBuildingAnimation", () => ({ GraphBuildingAnimation: () => null }));

import { render, screen } from "@testing-library/react";
import { StepPreparing } from "../StepPreparing";

beforeEach(() => {
  mockTenantState = { tenantReady: false, podUnreachable: false, wakeState: "idle", currentName: "Personal Workspace" };
});

afterEach(() => {
  jest.clearAllMocks();
});

describe("StepPreparing", () => {
  it("moves on once the workspace's pod is confirmed ready", () => {
    mockTenantState.tenantReady = true;
    const onReady = jest.fn();

    render(<StepPreparing onReady={onReady} />);

    expect(onReady).toHaveBeenCalledTimes(1);
  });

  it("keeps showing the progress screen while the pod is still coming up", () => {
    const onReady = jest.fn();

    render(<StepPreparing onReady={onReady} />);

    expect(screen.getByRole("heading", { name: "Building your workspace…" })).toBeInTheDocument();
    expect(onReady).not.toHaveBeenCalled();
  });

  it("shows an error with a retry instead of the spinner once the readiness poll has given up", () => {
    mockTenantState.podUnreachable = true;
    const onReady = jest.fn();

    render(<StepPreparing onReady={onReady} />);

    expect(screen.getByRole("heading", { name: "We're having trouble reaching your workspace" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Try again" })).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Building your workspace…" })).not.toBeInTheDocument();
    expect(onReady).not.toHaveBeenCalled();
  });

  it("uses the wake-specific copy when the workspace was asleep and did not come back", () => {
    mockTenantState.podUnreachable = true;
    mockTenantState.wakeState = "failed";

    render(<StepPreparing onReady={jest.fn()} />);

    expect(screen.getByRole("heading", { name: "Personal Workspace didn't wake up" })).toBeInTheDocument();
  });
});
