import { render, screen } from "@testing-library/react";
import type { SetupConnectorCfg } from "@/modules/integrations/types";
import SetupConnectorCard from "../SetupConnectorCard";

function cfg(overrides: Partial<SetupConnectorCfg> = {}): SetupConnectorCfg {
  return {
    key: "claude-code",
    name: "Claude Code",
    cta: "Connect via plugin",
    description: "Give Claude Code persistent memory across sessions.",
    icon: <span>icon</span>,
    buildSteps: () => [],
    ...overrides,
  };
}

beforeEach(() => {
  jest.useFakeTimers().setSystemTime(new Date("2026-09-29T12:05:00.000Z"));
});

afterEach(() => {
  jest.useRealTimers();
});

describe("SetupConnectorCard", () => {
  it("shows Connected with a relative last-seen line for a connected plugin agent", () => {
    render(
      <SetupConnectorCard
        card={cfg()}
        isActive={false}
        hasSignal={true}
        isConnected={true}
        lastSeenAt="2026-09-29T12:00:00.000Z"
        onOpen={jest.fn()}
      />,
    );

    expect(screen.getByText("Connected")).toBeInTheDocument();
    expect(screen.getByText("Last seen 5m ago")).toBeInTheDocument();
  });

  it("shows Live, the same answer as the dashboard, while a session of the agent is running", () => {
    render(
      <SetupConnectorCard
        card={cfg()}
        isActive={false}
        hasSignal={true}
        isConnected={true}
        lastSeenAt="2026-09-29T12:00:00.000Z"
        running={true}
        onOpen={jest.fn()}
      />,
    );

    expect(screen.getByText("Live")).toBeInTheDocument();
  });

  it("shows Connected with no last-seen line when only the sticky registry knows about it", () => {
    render(
      <SetupConnectorCard
        card={cfg()}
        isActive={false}
        hasSignal={true}
        isConnected={true}
        onOpen={jest.fn()}
      />,
    );

    expect(screen.getByText("Connected")).toBeInTheDocument();
    expect(screen.queryByText(/Last seen/)).not.toBeInTheDocument();
  });

  it("shows Not connected yet before any registration is seen", () => {
    render(
      <SetupConnectorCard
        card={cfg()}
        isActive={false}
        hasSignal={true}
        isConnected={false}
        onOpen={jest.fn()}
      />,
    );

    expect(screen.getByText("Not connected yet")).toBeInTheDocument();
    expect(screen.queryByText("Connected")).not.toBeInTheDocument();
  });

  it("shows the same Connected wording for a non-plugin (MCP) connector, with no last-seen line", () => {
    // last_active_at is only ever populated for a plugin-agent key
    // (lastSeenByAgent), so a real caller never passes lastSeenAt for a card
    // like this one, and the card renders as if it hadn't been.
    render(
      <SetupConnectorCard
        card={cfg({ key: "cursor", name: "Cursor" })}
        isActive={false}
        hasSignal={true}
        isConnected={true}
        onOpen={jest.fn()}
      />,
    );

    expect(screen.getByText("Connected")).toBeInTheDocument();
    expect(screen.queryByText(/Last seen/)).not.toBeInTheDocument();
  });

  it("shows View connection for a connected plugin agent, not the install cta", () => {
    render(
      <SetupConnectorCard
        card={cfg()}
        isActive={false}
        hasSignal={true}
        isConnected={true}
        onOpen={jest.fn()}
      />,
    );

    expect(screen.getByText("View connection")).toBeInTheDocument();
    expect(screen.queryByText("Connect via plugin")).not.toBeInTheDocument();
  });

  it("keeps the install cta for a not-yet-connected plugin agent", () => {
    render(
      <SetupConnectorCard
        card={cfg()}
        isActive={false}
        hasSignal={true}
        isConnected={false}
        onOpen={jest.fn()}
      />,
    );

    expect(screen.getByText("Connect via plugin")).toBeInTheDocument();
  });

  it("keeps its own cta for a connected non-plugin (MCP) connector", () => {
    render(
      <SetupConnectorCard
        card={cfg({ key: "cursor", name: "Cursor", cta: "Connect via MCP" })}
        isActive={false}
        hasSignal={true}
        isConnected={true}
        onOpen={jest.fn()}
      />,
    );

    expect(screen.getByText("Connect via MCP")).toBeInTheDocument();
    expect(screen.queryByText("View connection")).not.toBeInTheDocument();
  });

  it("renders no status at all for a card with no detection signal", () => {
    render(
      <SetupConnectorCard
        card={cfg({ key: "vscode", name: "VS Code" })}
        isActive={false}
        hasSignal={false}
        isConnected={false}
        onOpen={jest.fn()}
      />,
    );

    expect(screen.queryByText("Not connected yet")).not.toBeInTheDocument();
    expect(screen.queryByText("Connected")).not.toBeInTheDocument();
  });

  it("shows Checking… instead of a confident Not connected yet while telemetry is loading", () => {
    render(
      <SetupConnectorCard
        card={cfg()}
        isActive={false}
        hasSignal={true}
        isConnected={false}
        loading
        onOpen={jest.fn()}
      />,
    );

    expect(screen.getByText("Checking…")).toBeInTheDocument();
    expect(screen.queryByText("Not connected yet")).not.toBeInTheDocument();
  });

  it("keeps a real Connected reading even while loading", () => {
    render(
      <SetupConnectorCard
        card={cfg()}
        isActive={false}
        hasSignal={true}
        isConnected={true}
        loading
        onOpen={jest.fn()}
      />,
    );

    expect(screen.getByText("Connected")).toBeInTheDocument();
    expect(screen.queryByText("Checking…")).not.toBeInTheDocument();
  });
});
