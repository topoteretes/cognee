import { render, screen } from "@testing-library/react";
import { AciCard } from "../AciCard";
import { CARDS_CFG } from "../agentConnectionSteps";
import type { AciCardConfig } from "../agentConnectionSteps";

function card(key: AciCardConfig["key"]): AciCardConfig {
  const cfg = CARDS_CFG.find((c) => c.key === key);
  if (!cfg) throw new Error(`no CARDS_CFG entry for ${key}`);
  return cfg;
}

beforeEach(() => {
  jest.useFakeTimers().setSystemTime(new Date("2026-09-29T12:05:00.000Z"));
});

afterEach(() => {
  jest.useRealTimers();
});

describe("AciCard", () => {
  it("shows Connected with a relative last-seen line for a connected plugin agent", () => {
    render(
      <AciCard
        card={card("claude-code")}
        activeKey={null}
        isUploading={false}
        hasDocuments={false}
        integrationConnected={{ "claude-code": true }}
        lastSeenByAgent={{ "claude-code": "2026-09-29T12:00:00.000Z" }}
        onCardClick={jest.fn()}
      />,
    );

    expect(screen.getByText("Connected")).toBeInTheDocument();
    expect(screen.getByText("Last seen 5m ago")).toBeInTheDocument();
  });

  it("shows Live, the same answer as the memory graph, while a session of the agent is running and fresh", () => {
    render(
      <AciCard
        card={card("claude-code")}
        activeKey={null}
        isUploading={false}
        hasDocuments={false}
        integrationConnected={{ "claude-code": true }}
        lastSeenByAgent={{ "claude-code": "2026-09-29T12:04:00.000Z" }}
        runningAgents={new Set(["claude-code"])}
        onCardClick={jest.fn()}
      />,
    );

    expect(screen.getByText("Live")).toBeInTheDocument();
    expect(screen.getByText("Active now")).toBeInTheDocument();
  });

  it("shows Live with the last-activity phrase, not Active now, once a running session has gone quiet past the threshold", () => {
    render(
      <AciCard
        card={card("claude-code")}
        activeKey={null}
        isUploading={false}
        hasDocuments={false}
        integrationConnected={{ "claude-code": true }}
        lastSeenByAgent={{ "claude-code": "2026-09-29T11:41:00.000Z" }}
        runningAgents={new Set(["claude-code"])}
        onCardClick={jest.fn()}
      />,
    );

    expect(screen.getByText("Live")).toBeInTheDocument();
    expect(screen.queryByText("Active now")).not.toBeInTheDocument();
    expect(screen.getByText("Running · last activity 24m ago")).toBeInTheDocument();
  });

  it("shows Connected with no last-seen line when only the sticky registry knows about it", () => {
    render(
      <AciCard
        card={card("codex")}
        activeKey={null}
        isUploading={false}
        hasDocuments={false}
        integrationConnected={{ codex: true }}
        onCardClick={jest.fn()}
      />,
    );

    expect(screen.getByText("Connected")).toBeInTheDocument();
    expect(screen.queryByText(/Last seen/)).not.toBeInTheDocument();
  });

  it("shows Not connected yet before any registration is seen", () => {
    render(
      <AciCard
        card={card("claude-code")}
        activeKey={null}
        isUploading={false}
        hasDocuments={false}
        integrationConnected={{}}
        onCardClick={jest.fn()}
      />,
    );

    expect(screen.getByText("Not connected yet")).toBeInTheDocument();
    expect(screen.queryByText("Connected")).not.toBeInTheDocument();
  });

  it("shows the same Connected wording for a non-plugin card (Openclaw), with no last-seen line", () => {
    // lastSeenByAgent is only ever populated for a plugin-agent key (see
    // lastSeenByAgent in getAgentConnections.ts), so a real caller never
    // passes an entry for Openclaw here.
    render(
      <AciCard
        card={card("openclaw")}
        activeKey={null}
        isUploading={false}
        hasDocuments={false}
        integrationConnected={{ openclaw: true }}
        onCardClick={jest.fn()}
      />,
    );

    expect(screen.getByText("Connected")).toBeInTheDocument();
    expect(screen.queryByText(/Last seen/)).not.toBeInTheDocument();
  });

  it("renders no status line at all for an unconnected non-plugin card", () => {
    render(
      <AciCard
        card={card("openclaw")}
        activeKey={null}
        isUploading={false}
        hasDocuments={false}
        integrationConnected={{}}
        onCardClick={jest.fn()}
      />,
    );

    expect(screen.queryByText("Not connected yet")).not.toBeInTheDocument();
    expect(screen.queryByText("Connected")).not.toBeInTheDocument();
  });

  it("shows Checking… instead of a confident Not connected yet while telemetry is loading", () => {
    render(
      <AciCard
        card={card("claude-code")}
        activeKey={null}
        isUploading={false}
        hasDocuments={false}
        integrationConnected={{}}
        loading
        onCardClick={jest.fn()}
      />,
    );

    expect(screen.getByText("Checking…")).toBeInTheDocument();
    expect(screen.queryByText("Not connected yet")).not.toBeInTheDocument();
  });

  it("keeps a real Connected reading even while loading", () => {
    render(
      <AciCard
        card={card("claude-code")}
        activeKey={null}
        isUploading={false}
        hasDocuments={false}
        integrationConnected={{ "claude-code": true }}
        loading
        onCardClick={jest.fn()}
      />,
    );

    expect(screen.getByText("Connected")).toBeInTheDocument();
    expect(screen.queryByText("Checking…")).not.toBeInTheDocument();
  });

  it("does not apply loading to the upload card, which reads hasDocuments rather than telemetry", () => {
    render(
      <AciCard
        card={card("upload")}
        activeKey={null}
        isUploading={false}
        hasDocuments={false}
        integrationConnected={{}}
        loading
        onCardClick={jest.fn()}
      />,
    );

    expect(screen.queryByText("Checking…")).not.toBeInTheDocument();
  });
});
