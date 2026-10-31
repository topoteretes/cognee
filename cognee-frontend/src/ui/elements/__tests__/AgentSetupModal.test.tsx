jest.mock("@/modules/tenant/TenantProvider", () => ({
  useCogniInstance: () => ({ cogniInstance: null, serviceUrl: "https://t.aws.cognee.ai", apiKey: "key", isInitializing: false }),
}));
jest.mock("@/ui/layout/OsPreferenceContext", () => ({ useOsPreference: () => ({ os: "mac" }) }));

const mockUseDashboardTelemetry = jest.fn();
jest.mock("@/app/(app)/dashboard/hooks/useDashboardTelemetry", () => ({
  useDashboardTelemetry: () => mockUseDashboardTelemetry(),
}));

jest.mock("../ConnectedAgentModal", () => ({
  ConnectedAgentModal: ({ agent, initialTab }: { agent: string; initialTab: string }) => <div data-testid="connected-modal" data-initial-tab={initialTab}>{agent}</div>,
}));
jest.mock("../SetupWizardModal", () => ({
  SetupWizardModal: ({ cfg }: { cfg: { key: string } }) => <div data-testid="wizard-modal">{cfg.key}</div>,
}));

import { render, screen } from "@testing-library/react";
import { AgentSetupModal } from "../AgentSetupModal";
import type { SetupConnectorCfg } from "@/modules/integrations/types";

function cfg(overrides: Partial<SetupConnectorCfg> = {}): SetupConnectorCfg {
  return {
    key: "claude-code",
    name: "Claude Code",
    cta: "Connect via plugin",
    description: "",
    icon: <span />,
    buildSteps: () => [],
    ...overrides,
  };
}

const CARDS = [cfg(), cfg({ key: "cursor", name: "Cursor", cta: "Connect via MCP" })];

beforeEach(() => {
  mockUseDashboardTelemetry.mockReturnValue({ sessions: [], connections: [] });
});

describe("AgentSetupModal", () => {
  it("renders nothing when activeKey names no card", () => {
    const { container } = render(<AgentSetupModal cards={CARDS} connectedKeys={{}} activeKey="nope" onClose={jest.fn()} />);

    expect(container).toBeEmptyDOMElement();
  });

  it("opens the connected summary for a connected plugin agent", () => {
    render(<AgentSetupModal cards={CARDS} connectedKeys={{ "claude-code": true }} activeKey="claude-code" onClose={jest.fn()} />);

    expect(screen.getByTestId("connected-modal")).toHaveTextContent("claude-code");
    expect(screen.getByTestId("connected-modal")).toHaveAttribute("data-initial-tab", "status");
    expect(screen.queryByTestId("wizard-modal")).not.toBeInTheDocument();
  });

  it("opens the install wizard for a not-yet-connected plugin agent", () => {
    render(<AgentSetupModal cards={CARDS} connectedKeys={{ "claude-code": false }} activeKey="claude-code" onClose={jest.fn()} />);

    expect(screen.getByTestId("wizard-modal")).toBeInTheDocument();
    expect(screen.queryByTestId("connected-modal")).not.toBeInTheDocument();
  });

  it("keeps setup selected when the agent connects while the modal is open", () => {
    const { rerender } = render(<AgentSetupModal cards={CARDS} connectedKeys={{ "claude-code": false }} activeKey="claude-code" onClose={jest.fn()} />);
    rerender(<AgentSetupModal cards={CARDS} connectedKeys={{ "claude-code": true }} activeKey="claude-code" onClose={jest.fn()} />);

    expect(screen.getByTestId("connected-modal")).toHaveAttribute("data-initial-tab", "setup");
  });

  it("always opens the install wizard for a non-plugin (MCP) card, connected or not", () => {
    render(<AgentSetupModal cards={CARDS} connectedKeys={{ cursor: true }} activeKey="cursor" onClose={jest.fn()} />);

    expect(screen.getByTestId("wizard-modal")).toHaveTextContent("cursor");
  });
});
