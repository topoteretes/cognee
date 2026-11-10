const mockPush = jest.fn();
jest.mock("next/navigation", () => ({
  useRouter: () => ({ push: mockPush }),
}));

const mockMarkWelcomeSeen = jest.fn().mockResolvedValue(undefined);
const mockMarkOnboardingComplete = jest.fn().mockResolvedValue(undefined);
jest.mock("@/modules/users/UserContext", () => ({
  useUser: () => ({ markWelcomeSeen: mockMarkWelcomeSeen, markOnboardingComplete: mockMarkOnboardingComplete }),
}));

const mockTrack = jest.fn();
jest.mock("../../useOnboardingTrackEvent", () => ({
  useOnboardingTrackEvent: () => mockTrack,
}));

const mockAgentConnectionStatus = jest.fn((props: unknown): null => { void props; return null; });
jest.mock("@/ui/elements/AgentConnectionStatus", () => ({
  AgentConnectionStatus: (props: unknown) => mockAgentConnectionStatus(props),
}));

jest.mock("@/ui/layout/OsPreferenceContext", () => ({
  useOsPreference: () => ({ os: "mac", setOs: jest.fn() }),
}));

jest.mock("@/ui/elements/OsToggle", () => ({
  OsToggle: () => null,
}));

import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { AgentOnboarding } from "../AgentOnboarding";
import type { CogneeInstance } from "@/modules/instances/types";

const INSTANCE = { name: "CloudCognee", instanceId: "tenant-1", fetch: jest.fn() } as CogneeInstance;

function renderPage(agent: "claude-code" | "codex" = "claude-code") {
  return render(
    <AgentOnboarding
      agent={agent}
      serviceUrl="https://tenant.aws.cognee.ai"
      apiKey="key-123"
      cogniInstance={INSTANCE}
      onRestart={jest.fn()}
    />,
  );
}

beforeEach(() => {
  jest.clearAllMocks();
  mockMarkWelcomeSeen.mockResolvedValue(undefined);
  mockMarkOnboardingComplete.mockResolvedValue(undefined);
});

describe("AgentOnboarding", () => {
  it.each(["claude-code", "codex"] as const)("shows a green border when %s connects", (agent) => {
    renderPage(agent);
    const card = screen.getByRole("heading", { name: `Connect ${agent === "codex" ? "Codex" : "Claude Code"}` }).parentElement?.parentElement?.parentElement?.parentElement;
    expect(card).not.toHaveStyle({ borderColor: "#22C55E" });

    const statusProps = mockAgentConnectionStatus.mock.lastCall?.[0] as { onConnectedChange: (connected: boolean) => void };
    act(() => statusProps.onConnectedChange(true));
    expect(card).toHaveStyle({ borderColor: "#22C55E" });

    act(() => statusProps.onConnectedChange(false));
    expect(card).not.toHaveStyle({ borderColor: "#22C55E" });
  });

  it("renders the shared connection status with this page's agent, instance and tracker", () => {
    renderPage();

    expect(mockAgentConnectionStatus).toHaveBeenCalledWith(
      expect.objectContaining({ agent: "claude-code", cogniInstance: INSTANCE, pageName: "Onboarding", track: mockTrack }),
    );
  });

  it("never auto-navigates on mount, connection status is informational only", () => {
    renderPage();

    expect(mockPush).not.toHaveBeenCalled();
    expect(mockMarkOnboardingComplete).not.toHaveBeenCalled();
  });

  it("lets the user finish through Continue with no connection ever detected", async () => {
    renderPage();

    const continueButton = screen.getByRole("button", { name: /continue/i });
    expect(continueButton).not.toBeDisabled();

    fireEvent.click(continueButton);

    await waitFor(() => expect(mockPush).toHaveBeenCalledWith("/dashboard"));
    expect(mockMarkWelcomeSeen).toHaveBeenCalled();
    expect(mockMarkOnboardingComplete).toHaveBeenCalled();
  });

  it("lets the user skip through Set up later just as freely", async () => {
    renderPage();

    fireEvent.click(screen.getByRole("button", { name: /set up later/i }));

    await waitFor(() => expect(mockPush).toHaveBeenCalledWith("/dashboard"));
    expect(mockTrack).toHaveBeenCalledWith({ pageName: "Onboarding", eventName: "onboarding_skipped" });
  });
});
