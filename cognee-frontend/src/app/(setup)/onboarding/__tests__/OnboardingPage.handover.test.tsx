const mockReplace = jest.fn();
let mockSearch = "";
jest.mock("next/navigation", () => ({
  useRouter: () => ({ replace: mockReplace }),
  useSearchParams: () => new URLSearchParams(mockSearch),
}));

let mockTenantReady = false;
jest.mock("@/modules/tenant/TenantProvider", () => ({
  useCogniInstance: () => ({ cogniInstance: null, isInitializing: false, serviceUrl: "", apiKey: "" }),
  useTenant: () => ({ tenantReady: mockTenantReady, podUnreachable: false, wakeState: "idle", currentName: "Personal Workspace" }),
}));
jest.mock("@/modules/users/UserContext", () => ({
  useUser: () => ({ markWelcomeSeen: jest.fn(), markOnboardingComplete: jest.fn() }),
}));
jest.mock("@/modules/analytics", () => ({ TrackPageView: () => null, trackEvent: jest.fn() }));
jest.mock("../useOnboardingTrackEvent", () => ({ useOnboardingTrackEvent: () => jest.fn() }));
jest.mock("../ServeOnboarding", () => ({ __esModule: true, default: () => null }));
jest.mock("../partials/Step1", () => ({ Step1: () => null }));
jest.mock("../partials/Step2", () => ({ Step2: () => null }));
jest.mock("../partials/Step3", () => ({ Step3: () => null, extractFirstAnswer: jest.fn() }));
jest.mock("../partials/AgentOnboarding", () => ({ AgentOnboarding: () => null }));
jest.mock("../partials/StepSelect", () => ({ StepSelect: () => <div>path selection</div> }));
jest.mock("@/ui/elements/Auth/TetrisBackground", () => ({ __esModule: true, default: () => null }));
jest.mock("../partials/GraphBuildingAnimation", () => ({ GraphBuildingAnimation: () => null }));

const mockIsNewOnboardingHost = jest.fn();
jest.mock("@/utils/newOnboardingRollout", () => ({
  isNewOnboardingHost: (...args: unknown[]) => mockIsNewOnboardingHost(...args),
}));
const mockIsCloudEnvironment = jest.fn();
jest.mock("@/utils/isCloudEnvironment", () => ({
  __esModule: true,
  default: () => mockIsCloudEnvironment(),
}));

import { act, fireEvent, render, screen } from "@testing-library/react";
import OnboardingPage from "../OnboardingPage";

beforeEach(() => {
  jest.clearAllMocks();
  mockSearch = "";
  mockTenantReady = false;
  mockIsCloudEnvironment.mockReturnValue(true);
  mockIsNewOnboardingHost.mockReturnValue(false);
});

// The workspace comes up: preparing reports ready from its own effect.
function workspaceComesUp(view: ReturnType<typeof render>): void {
  mockTenantReady = true;
  view.rerender(<OnboardingPage />);
}

describe("OnboardingPage hand-over after preparing", () => {
  it("goes on to path selection on a host that keeps the current onboarding", () => {
    const view = render(<OnboardingPage />);

    workspaceComesUp(view);

    expect(screen.getByText("path selection")).toBeInTheDocument();
    expect(mockReplace).not.toHaveBeenCalled();
  });

  it("reads the host the browser is on", () => {
    const view = render(<OnboardingPage />);

    workspaceComesUp(view);

    expect(mockIsNewOnboardingHost).toHaveBeenCalledWith(window.location.host);
  });

  it("hands over to the onboarding window on a host that has the new onboarding", () => {
    mockIsNewOnboardingHost.mockReturnValue(true);
    const view = render(<OnboardingPage />);

    workspaceComesUp(view);

    expect(mockReplace).toHaveBeenCalledWith("/onboarding/window");
    expect(screen.queryByText("path selection")).not.toBeInTheDocument();
  });

  it("keeps ?replay=1 on the way to the window", () => {
    mockIsNewOnboardingHost.mockReturnValue(true);
    mockSearch = "replay=1";
    const view = render(<OnboardingPage />);

    workspaceComesUp(view);

    expect(mockReplace).toHaveBeenCalledWith("/onboarding/window?replay=1");
  });

  it("hands over once when the user continues without waiting and the workspace then comes up", () => {
    jest.useFakeTimers();
    try {
      mockIsNewOnboardingHost.mockReturnValue(true);
      const view = render(<OnboardingPage />);
      act(() => {
        jest.advanceTimersByTime(31_000);
      });

      fireEvent.click(screen.getByRole("button", { name: "Continue without waiting" }));
      workspaceComesUp(view);

      expect(mockReplace).toHaveBeenCalledTimes(1);
    } finally {
      jest.useRealTimers();
    }
  });

  it("never hands over outside the cloud, whatever the host", () => {
    mockIsNewOnboardingHost.mockReturnValue(true);
    mockIsCloudEnvironment.mockReturnValue(false);
    const view = render(<OnboardingPage />);

    workspaceComesUp(view);

    expect(mockReplace).not.toHaveBeenCalled();
    expect(screen.getByText("path selection")).toBeInTheDocument();
  });
});
