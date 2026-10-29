jest.mock("next/navigation", () => ({
  useRouter: () => ({ push: jest.fn() }),
}));

import type { ReactElement } from "react";
import { MantineProvider } from "@mantine/core";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { ConnectedAgentModal } from "../ConnectedAgentModal";
import { OsPreferenceProvider } from "@/ui/layout/OsPreferenceContext";
import type { SetupConnectorCfg, StepDef } from "@/modules/integrations/types";
import type { AgentConnection } from "@/modules/agents/getAgentConnections";
import type { CogneeInstance } from "@/modules/instances/types";
import type { SessionRow } from "@/modules/sessions/getSessions";

// jsdom has no ResizeObserver, which Mantine's SegmentedControl (the OS
// toggle) needs to position its FloatingIndicator.
class ResizeObserverStub {
  observe(): void {}
  unobserve(): void {}
  disconnect(): void {}
}
global.ResizeObserver = ResizeObserverStub;

function renderModal(ui: ReactElement) {
  return render(<MantineProvider><OsPreferenceProvider>{ui}</OsPreferenceProvider></MantineProvider>);
}

function cogneeInstance(fetchImpl: jest.Mock): CogneeInstance {
  return { name: "CloudCognee", instanceId: "tenant-1", fetch: fetchImpl };
}

function jsonResponse(payload: unknown): { json: () => Promise<unknown> } {
  return { json: async () => payload };
}

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

function connection(overrides: Partial<AgentConnection> = {}): AgentConnection {
  return {
    id: "conn-1",
    agent_session_name: "goran-laptop",
    type: "claude_code",
    session_id: "",
    last_active_at: "2026-09-29T12:00:00.000Z",
    status: "active",
    user_id: "u1",
    memory_mode: "hybrid",
    datasets: [],
    ...overrides,
  };
}

function session(overrides: Partial<SessionRow> = {}): SessionRow {
  return {
    session_id: "claude_abc",
    user_id: "u1",
    dataset_id: null,
    status: "completed",
    effective_status: "completed",
    started_at: "2026-09-29T11:00:00Z",
    last_activity_at: "2026-09-29T12:00:00.000Z",
    ended_at: null,
    tokens_in: null,
    tokens_out: null,
    cost_usd: null,
    error_count: 0,
    last_model: null,
    ...overrides,
  };
}

const STEPS: StepDef[] = [{ title: "Install the plugin", description: "Run this command." }];

beforeEach(() => {
  jest.useFakeTimers().setSystemTime(new Date("2026-09-29T12:05:00.000Z"));
});

afterEach(() => {
  jest.useRealTimers();
});

describe("ConnectedAgentModal", () => {
  it("shows status details immediately and keeps setup in a separate tab", () => {
    const steps: StepDef[] = [
      { title: "Open terminal", description: "Find your terminal" },
      { title: "Install plugin", description: "Run the install command" },
      { title: "Start Claude Code", description: "Connection confirmed" },
    ];
    renderModal(
      <ConnectedAgentModal
        agent="claude-code"
        cfg={cfg()}
        cogniInstance={null}
        sessions={[session({ session_id: "claude_new", last_activity_at: "2026-09-29T12:00:00Z" }), session({ session_id: "claude_old", last_activity_at: "2026-09-29T11:00:00Z" })]}
        connections={[connection()]}
        connectionsTruncated={false}
        steps={steps}
        currentStep={0}
        onClose={jest.fn()}
        onStepSelect={jest.fn()}
      />,
    );

    expect(screen.getByRole("tab", { name: "Status" })).toHaveAttribute("aria-selected", "true");
    expect(screen.queryByText("Open terminal")).not.toBeInTheDocument();
    expect(screen.getByTitle("claude_new")).toBeInTheDocument();
    expect(screen.getByTitle("claude_old")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /show details|hide details/i })).not.toBeInTheDocument();
    expect(screen.getByText("Sessions (2)")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("tab", { name: "Setup" }));
    expect(screen.queryByText("Sessions (2)")).not.toBeInTheDocument();
    expect(screen.getByText("Connection confirmed")).toBeInTheDocument();
    expect(screen.getByText("Find your terminal").closest("[aria-hidden]")).toHaveAttribute("aria-hidden", "true");
    expect(screen.getByText("Run the install command").closest("[aria-hidden]")).toHaveAttribute("aria-hidden", "true");

    fireEvent.click(screen.getByText("Open terminal"));
    expect(screen.getByText("Find your terminal").closest("[aria-hidden]")).toHaveAttribute("aria-hidden", "false");
    fireEvent.click(screen.getByRole("tab", { name: "Status" }));
    expect(screen.getByText("Sessions (2)")).toBeInTheDocument();
  });

  it("keeps the confirmation visible when an agent connects during setup", () => {
    renderModal(
      <ConnectedAgentModal
        agent="claude-code"
        cfg={cfg()}
        cogniInstance={null}
        sessions={[]}
        connections={[connection()]}
        connectionsTruncated={false}
        steps={STEPS}
        currentStep={0}
        initialTab="setup"
        onClose={jest.fn()}
        onStepSelect={jest.fn()}
      />,
    );

    expect(screen.getByRole("tab", { name: "Setup" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByRole("status")).toHaveTextContent("Connected successfully");
  });

  it("shows Active now when the running session's last activity is within the freshness threshold", () => {
    renderModal(
      <ConnectedAgentModal
        agent="claude-code"
        cfg={cfg()}
        cogniInstance={null}
        sessions={[session({ status: "running", effective_status: "running", last_activity_at: "2026-09-29T12:04:00.000Z" })]}
        connections={[connection({ last_active_at: "2026-09-29T12:04:00.000Z" })]}
        connectionsTruncated={false}
        steps={STEPS}
        currentStep={0}
        onClose={jest.fn()}
        onStepSelect={jest.fn()}
      />,
    );

    expect(screen.getByText("Active now")).toBeInTheDocument();
  });

  it("does not claim Active now for a session still reading running but quiet past the threshold", () => {
    renderModal(
      <ConnectedAgentModal
        agent="claude-code"
        cfg={cfg()}
        cogniInstance={null}
        sessions={[session({ status: "running", effective_status: "running", last_activity_at: "2026-09-29T11:41:00.000Z" })]}
        connections={[connection({ last_active_at: "2026-09-29T11:41:00.000Z" })]}
        connectionsTruncated={false}
        steps={STEPS}
        currentStep={0}
        onClose={jest.fn()}
        onStepSelect={jest.fn()}
      />,
    );

    expect(screen.queryByText("Active now")).not.toBeInTheDocument();
    expect(screen.getByText("Running · last activity 24m ago")).toBeInTheDocument();
  });

  it("shows a relative last-seen line when nothing is running", () => {
    renderModal(
      <ConnectedAgentModal
        agent="claude-code"
        cfg={cfg()}
        cogniInstance={null}
        sessions={[session({ last_activity_at: "2026-09-29T12:00:00.000Z" })]}
        connections={[connection({ last_active_at: "2026-09-29T11:00:00.000Z" })]}
        connectionsTruncated={false}
        steps={STEPS}
        currentStep={0}
        onClose={jest.fn()}
        onStepSelect={jest.fn()}
      />,
    );

    expect(screen.getByText("Last seen 5m ago")).toBeInTheDocument();
  });

  it("lists running sessions before finished ones, most recent first within each group", () => {
    renderModal(
      <ConnectedAgentModal
        agent="claude-code"
        cfg={cfg()}
        cogniInstance={null}
        sessions={[
          session({ session_id: "cc_old-run", status: "running", effective_status: "running", last_activity_at: "2026-09-29T10:00:00Z" }),
          session({ session_id: "cc_finished", status: "completed", effective_status: "completed", last_activity_at: "2026-09-29T11:30:00Z" }),
          session({ session_id: "cc_new-run", status: "running", effective_status: "running", last_activity_at: "2026-09-29T11:00:00Z" }),
        ]}
        connections={[connection()]}
        connectionsTruncated={false}
        steps={STEPS}
        currentStep={0}
        onClose={jest.fn()}
        onStepSelect={jest.fn()}
      />,
    );

    // testing-library returns matches in document order, so this doubles as
    // the ordering assertion.
    const order = screen.getAllByTitle(/^cc_(old-run|finished|new-run)$/).map((el) => el.getAttribute("title"));
    expect(order).toEqual(["cc_new-run", "cc_old-run", "cc_finished"]);
    expect(screen.getByText("Sessions (3)")).toBeInTheDocument();
  });

  it("shows only the newest 5 sessions in the list while the header keeps the true total", () => {
    const manySessions = Array.from({ length: 8 }, (_, i) =>
      session({ session_id: `cc_s${i}`, status: "completed", effective_status: "completed", last_activity_at: `2026-09-29T${String(10 + i).padStart(2, "0")}:00:00Z` }),
    );
    renderModal(
      <ConnectedAgentModal
        agent="claude-code"
        cfg={cfg()}
        cogniInstance={null}
        sessions={manySessions}
        connections={[connection()]}
        connectionsTruncated={false}
        steps={STEPS}
        currentStep={0}
        onClose={jest.fn()}
        onStepSelect={jest.fn()}
      />,
    );

    expect(screen.getByText("Sessions (8)")).toBeInTheDocument();
    expect(screen.getAllByTitle(/^cc_s\d$/)).toHaveLength(5);
    // Newest (s7, last_activity_at 17:00) is in the visible page, oldest (s0) is not.
    expect(screen.getByTitle("cc_s7")).toBeInTheDocument();
    expect(screen.queryByTitle("cc_s0")).not.toBeInTheDocument();
  });

  it("shows the full session id in the list, not a truncated one", () => {
    renderModal(
      <ConnectedAgentModal
        agent="claude-code"
        cfg={cfg()}
        cogniInstance={null}
        sessions={[session({ session_id: "claude_e7c5b154_a_very_long_id" })]}
        connections={[connection()]}
        connectionsTruncated={false}
        steps={STEPS}
        currentStep={0}
        onClose={jest.fn()}
        onStepSelect={jest.fn()}
      />,
    );

    expect(screen.getByTitle("claude_e7c5b154_a_very_long_id")).toHaveTextContent(
      "claude_e7c5b154_a_very_long_id",
    );
  });

  it("ignores a session belonging to a different agent", () => {
    renderModal(
      <ConnectedAgentModal
        agent="claude-code"
        cfg={cfg()}
        cogniInstance={null}
        sessions={[session({ session_id: "codex_other" })]}
        connections={[connection()]}
        connectionsTruncated={false}
        steps={STEPS}
        currentStep={0}
        onClose={jest.fn()}
        onStepSelect={jest.fn()}
      />,
    );

    expect(screen.getByText("Sessions (0)")).toBeInTheDocument();
    expect(screen.getByText("No sessions yet.")).toBeInTheDocument();
  });

  it("links View all sessions to the activity page filtered by this agent's channel", () => {
    renderModal(
      <ConnectedAgentModal
        agent="claude-code"
        cfg={cfg()}
        cogniInstance={null}
        sessions={[]}
        connections={[connection()]}
        connectionsTruncated={false}
        steps={STEPS}
        currentStep={0}
        onClose={jest.fn()}
        onStepSelect={jest.fn()}
      />,
    );

    expect(screen.getByText("View all sessions →")).toHaveAttribute(
      "href",
      "/activity?via=Claude%20Code",
    );
  });

  it("shows the datasets this agent writes to, deduplicated across connections", () => {
    renderModal(
      <ConnectedAgentModal
        agent="claude-code"
        cfg={cfg()}
        cogniInstance={null}
        sessions={[]}
        connections={[
          connection({
            id: "c1",
            memory_mode: "none",
            datasets: [{ name: "handbook", role: null }, { name: "contracts", role: null }],
          }),
          connection({ id: "c2", memory_mode: "none", datasets: [{ name: "contracts", role: null }] }),
        ]}
        connectionsTruncated={false}
        steps={STEPS}
        currentStep={0}
        onClose={jest.fn()}
        onStepSelect={jest.fn()}
      />,
    );

    // The detail view lists every dataset, not one name plus a count.
    expect(screen.getByText("Writes to 2 datasets")).toBeInTheDocument();
    expect(screen.getByText("contracts")).toBeInTheDocument();
    expect(screen.getByText("handbook")).toBeInTheDocument();
  });

  it("omits the datasets line entirely when no connection names one", () => {
    renderModal(
      <ConnectedAgentModal
        agent="claude-code"
        cfg={cfg()}
        cogniInstance={null}
        sessions={[]}
        connections={[connection({ datasets: [] })]}
        connectionsTruncated={false}
        steps={STEPS}
        currentStep={0}
        onClose={jest.fn()}
        onStepSelect={jest.fn()}
      />,
    );

    expect(screen.queryByText(/Writes to/)).not.toBeInTheDocument();
  });

  it("keeps setup separate from status until its tab is opened", () => {
    renderModal(
      <ConnectedAgentModal
        agent="claude-code"
        cfg={cfg()}
        cogniInstance={null}
        sessions={[]}
        connections={[connection()]}
        connectionsTruncated={false}
        steps={STEPS}
        currentStep={0}
        onClose={jest.fn()}
        onStepSelect={jest.fn()}
      />,
    );

    expect(screen.queryByText("Install the plugin")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("tab", { name: "Setup" }));

    expect(screen.getByText("Install the plugin")).toBeInTheDocument();
  });

  it("keeps the OS toggle out of the header, showing it only once the setup section is open", () => {
    renderModal(
      <ConnectedAgentModal
        agent="claude-code"
        cfg={cfg({ osAware: true })}
        cogniInstance={null}
        sessions={[]}
        connections={[connection()]}
        connectionsTruncated={false}
        steps={STEPS}
        currentStep={0}
        onClose={jest.fn()}
        onStepSelect={jest.fn()}
      />,
    );

    expect(screen.queryByLabelText("Operating system")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("tab", { name: "Setup" }));

    expect(screen.getByLabelText("Operating system")).toBeInTheDocument();
  });

  it("shows no OS toggle at all for a connector whose steps don't vary by OS", () => {
    renderModal(
      <ConnectedAgentModal
        agent="claude-code"
        cfg={cfg({ osAware: false })}
        cogniInstance={null}
        sessions={[]}
        connections={[connection()]}
        connectionsTruncated={false}
        steps={STEPS}
        currentStep={0}
        onClose={jest.fn()}
        onStepSelect={jest.fn()}
      />,
    );

    fireEvent.click(screen.getByRole("tab", { name: "Setup" }));

    expect(screen.queryByLabelText("Operating system")).not.toBeInTheDocument();
  });

  describe("recent memory", () => {
    it("shows a loading line, then the qas, once the detail fetch resolves", async () => {
      jest.useRealTimers();
      const fetchImpl = jest.fn().mockResolvedValue(
        jsonResponse({ recent_qas: [{ qa_id: "qa-1", question: "What's the release date?", time: "2026-09-29T12:00:00.000Z" }], recent_traces: [] }),
      );
      renderModal(
        <ConnectedAgentModal
          agent="claude-code"
          cfg={cfg()}
          cogniInstance={cogneeInstance(fetchImpl)}
          sessions={[]}
          connections={[connection()]}
          connectionsTruncated={false}
          steps={STEPS}
          currentStep={0}
          onClose={jest.fn()}
          onStepSelect={jest.fn()}
        />,
      );

      expect(screen.getByText("Loading recent memory...")).toBeInTheDocument();

      await waitFor(() => expect(screen.getByText("What's the release date?")).toBeInTheDocument());

      expect(screen.getByText("Most recent session only")).toBeInTheDocument();
    });

    it("hides recent memory when both qas and traces come back empty", async () => {
      jest.useRealTimers();
      const fetchImpl = jest.fn().mockResolvedValue(jsonResponse({ recent_qas: [], recent_traces: [] }));
      renderModal(
        <ConnectedAgentModal
          agent="claude-code"
          cfg={cfg()}
          cogniInstance={cogneeInstance(fetchImpl)}
          sessions={[]}
          connections={[connection()]}
          connectionsTruncated={false}
          steps={STEPS}
          currentStep={0}
          onClose={jest.fn()}
          onStepSelect={jest.fn()}
        />,
      );

      await waitFor(() => expect(screen.queryByText("Recent memory")).not.toBeInTheDocument());
    });

    it("falls back to traces when there are no qas", async () => {
      jest.useRealTimers();
      const fetchImpl = jest.fn().mockResolvedValue(
        jsonResponse({ recent_qas: [], recent_traces: [{ trace_id: "t1", status: "ok", memory_query: "search handbook" }] }),
      );
      renderModal(
        <ConnectedAgentModal
          agent="claude-code"
          cfg={cfg()}
          cogniInstance={cogneeInstance(fetchImpl)}
          sessions={[]}
          connections={[connection()]}
          connectionsTruncated={false}
          steps={STEPS}
          currentStep={0}
          onClose={jest.fn()}
          onStepSelect={jest.fn()}
        />,
      );

      await waitFor(() => expect(screen.getByText("search handbook")).toBeInTheDocument());
    });

    it("prefers qas over traces when both are present", async () => {
      jest.useRealTimers();
      const fetchImpl = jest.fn().mockResolvedValue(
        jsonResponse({
          recent_qas: [{ qa_id: "qa-1", question: "a real question", time: null }],
          recent_traces: [{ trace_id: "t1", status: "ok", memory_query: "a trace" }],
        }),
      );
      renderModal(
        <ConnectedAgentModal
          agent="claude-code"
          cfg={cfg()}
          cogniInstance={cogneeInstance(fetchImpl)}
          sessions={[]}
          connections={[connection()]}
          connectionsTruncated={false}
          steps={STEPS}
          currentStep={0}
          onClose={jest.fn()}
          onStepSelect={jest.fn()}
        />,
      );

      await waitFor(() => expect(screen.getByText("a real question")).toBeInTheDocument());
      expect(screen.queryByText("a trace")).not.toBeInTheDocument();
    });

    it("shows a quiet failure line, without a page error, when the detail fetch rejects", async () => {
      jest.useRealTimers();
      const fetchImpl = jest.fn().mockRejectedValue(new Error("pod unreachable"));
      renderModal(
        <ConnectedAgentModal
          agent="claude-code"
          cfg={cfg()}
          cogniInstance={cogneeInstance(fetchImpl)}
          sessions={[]}
          connections={[connection()]}
          connectionsTruncated={false}
          steps={STEPS}
          currentStep={0}
          onClose={jest.fn()}
          onStepSelect={jest.fn()}
        />,
      );

      await waitFor(() => expect(screen.getByText("Could not load recent memory")).toBeInTheDocument());
    });

    it("shows no recent memory section at all when no connection has a usable user_id", () => {
      renderModal(
        <ConnectedAgentModal
          agent="claude-code"
          cfg={cfg()}
          cogniInstance={cogneeInstance(jest.fn())}
          sessions={[]}
          connections={[connection({ user_id: "" })]}
          connectionsTruncated={false}
          steps={STEPS}
          currentStep={0}
          onClose={jest.fn()}
          onStepSelect={jest.fn()}
        />,
      );

      expect(screen.queryByText("Recent memory")).not.toBeInTheDocument();
    });
  });

  describe("first connected", () => {
    it("shows the earliest connection date when the connections page is complete", () => {
      renderModal(
        <ConnectedAgentModal
          agent="claude-code"
          cfg={cfg()}
          cogniInstance={null}
          sessions={[]}
          connections={[
            connection({ id: "c1", last_active_at: "2026-01-01T00:00:00.000Z" }),
            connection({ id: "c2", last_active_at: "2026-06-01T00:00:00.000Z" }),
          ]}
          connectionsTruncated={false}
          steps={STEPS}
          currentStep={0}
          onClose={jest.fn()}
          onStepSelect={jest.fn()}
        />,
      );

      expect(screen.getByText(/First connected/)).toBeInTheDocument();
    });

    it("omits First connected when the connections page is truncated", () => {
      renderModal(
        <ConnectedAgentModal
          agent="claude-code"
          cfg={cfg()}
          cogniInstance={null}
          sessions={[]}
          connections={[connection({ last_active_at: "2026-01-01T00:00:00.000Z" })]}
          connectionsTruncated
          steps={STEPS}
          currentStep={0}
          onClose={jest.fn()}
          onStepSelect={jest.fn()}
        />,
      );

      expect(screen.queryByText(/First connected/)).not.toBeInTheDocument();
    });
  });

  describe("memory mode line", () => {
    it("describes a hybrid connection as session + knowledge graph", () => {
      renderModal(
        <ConnectedAgentModal
          agent="claude-code"
          cfg={cfg()}
          cogniInstance={null}
          sessions={[]}
          connections={[connection({ memory_mode: "hybrid", datasets: [] })]}
          connectionsTruncated={false}
          steps={STEPS}
          currentStep={0}
          onClose={jest.fn()}
          onStepSelect={jest.fn()}
        />,
      );

      expect(screen.getByText("Memory: session + knowledge graph")).toBeInTheDocument();
    });

    it("names the shared role once when every dataset agrees on it", () => {
      renderModal(
        <ConnectedAgentModal
          agent="claude-code"
          cfg={cfg()}
          cogniInstance={null}
          sessions={[]}
          connections={[connection({ memory_mode: "hybrid", datasets: [{ name: "agent_sessions", role: "read_write" }] })]}
          connectionsTruncated={false}
          steps={STEPS}
          currentStep={0}
          onClose={jest.fn()}
          onStepSelect={jest.fn()}
        />,
      );

      expect(screen.getByText("Memory: session + knowledge graph · reads and writes agent_sessions")).toBeInTheDocument();
    });

    it("omits the mode phrase for none/unknown, keeping the plain dataset count", () => {
      renderModal(
        <ConnectedAgentModal
          agent="claude-code"
          cfg={cfg()}
          cogniInstance={null}
          sessions={[]}
          connections={[connection({ memory_mode: "unknown", datasets: [{ name: "handbook", role: null }, { name: "contracts", role: null }] })]}
          connectionsTruncated={false}
          steps={STEPS}
          currentStep={0}
          onClose={jest.fn()}
          onStepSelect={jest.fn()}
        />,
      );

      expect(screen.getByText("Writes to 2 datasets")).toBeInTheDocument();
    });
  });

  describe("session rows", () => {
    it("shows status, started time, duration, last activity and errors, omitting what's missing", () => {
      renderModal(
        <ConnectedAgentModal
          agent="claude-code"
          cfg={cfg()}
          cogniInstance={null}
          sessions={[session({
            session_id: "claude_full",
            status: "completed",
            effective_status: "completed",
            started_at: "2026-09-29T11:00:00.000Z",
            last_activity_at: "2026-09-29T11:30:00.000Z",
            ended_at: "2026-09-29T11:30:00.000Z",
            error_count: 2,
          })]}
          connections={[connection()]}
          connectionsTruncated={false}
          steps={STEPS}
          currentStep={0}
          onClose={jest.fn()}
          onStepSelect={jest.fn()}
        />,
      );

      expect(screen.getByText(/Ended/)).toBeInTheDocument();
      // TZ is pinned to Europe/Berlin in jest.config.ts (CEST, UTC+2 in September).
      expect(screen.getByText(/Started 01:00 PM/)).toBeInTheDocument();
      expect(screen.getByText(/30m/)).toBeInTheDocument();
      expect(screen.getByText(/last activity/)).toBeInTheDocument();
      expect(screen.getByText(/2 errors/)).toBeInTheDocument();
    });

    it("colors only the status word by state: running green, everything else muted", () => {
      renderModal(
        <ConnectedAgentModal
          agent="claude-code"
          cfg={cfg()}
          cogniInstance={null}
          sessions={[session({ session_id: "claude_live", status: "running", effective_status: "running" })]}
          connections={[connection()]}
          connectionsTruncated={false}
          steps={STEPS}
          currentStep={0}
          onClose={jest.fn()}
          onStepSelect={jest.fn()}
        />,
      );

      expect(screen.getByText("Running")).toHaveStyle({ color: "#22C55E" });
    });

    it("colors a non-running status word muted, not green", () => {
      renderModal(
        <ConnectedAgentModal
          agent="claude-code"
          cfg={cfg()}
          cogniInstance={null}
          sessions={[session({ session_id: "claude_done", status: "completed", effective_status: "completed" })]}
          connections={[connection()]}
          connectionsTruncated={false}
          steps={STEPS}
          currentStep={0}
          onClose={jest.fn()}
          onStepSelect={jest.fn()}
        />,
      );

      expect(screen.getByText("Ended")).toHaveStyle({ color: "rgba(237,236,234,0.65)" });
    });

    it("renders the row's first line at the modal's body text size, not the browser default", () => {
      renderModal(
        <ConnectedAgentModal
          agent="claude-code"
          cfg={cfg()}
          cogniInstance={null}
          sessions={[session({ session_id: "claude_sized" })]}
          connections={[connection()]}
          connectionsTruncated={false}
          steps={STEPS}
          currentStep={0}
          onClose={jest.fn()}
          onStepSelect={jest.fn()}
        />,
      );

      expect(screen.getByText("Ended").closest("div")).toHaveStyle({ fontSize: "13px", fontWeight: "400" });
    });

    it("omits duration and errors when the session has neither", () => {
      renderModal(
        <ConnectedAgentModal
          agent="claude-code"
          cfg={cfg()}
          cogniInstance={null}
          sessions={[session({ session_id: "claude_bare", started_at: null, error_count: 0 })]}
          connections={[connection()]}
          connectionsTruncated={false}
          steps={STEPS}
          currentStep={0}
          onClose={jest.fn()}
          onStepSelect={jest.fn()}
        />,
      );

      expect(screen.queryByText(/error/)).not.toBeInTheDocument();
    });

    it("shows the full session id in the row, not a truncated one, including on a long id", () => {
      renderModal(
        <ConnectedAgentModal
          agent="claude-code"
          cfg={cfg()}
          cogniInstance={null}
          sessions={[session({ session_id: "claude_e7c5b154_a_very_long_session_identifier" })]}
          connections={[connection()]}
          connectionsTruncated={false}
          steps={STEPS}
          currentStep={0}
          onClose={jest.fn()}
          onStepSelect={jest.fn()}
        />,
      );

      expect(screen.getByText("claude_e7c5b154_a_very_long_session_identifier")).toBeInTheDocument();
    });

    it("copies the session id to the clipboard and shows brief feedback", async () => {
      const writeText = jest.fn().mockResolvedValue(undefined);
      Object.assign(navigator, { clipboard: { writeText } });
      renderModal(
        <ConnectedAgentModal
          agent="claude-code"
          cfg={cfg()}
          cogniInstance={null}
          sessions={[session({ session_id: "claude_copyme" })]}
          connections={[connection()]}
          connectionsTruncated={false}
          steps={STEPS}
          currentStep={0}
          onClose={jest.fn()}
          onStepSelect={jest.fn()}
        />,
      );

      fireEvent.click(screen.getByLabelText("Copy session id"));

      expect(writeText).toHaveBeenCalledWith("claude_copyme");
      await waitFor(() => expect(screen.getByLabelText("Copy session id")).toHaveTextContent("Copied"));
    });

    it("does not crash when the clipboard write rejects", () => {
      const writeText = jest.fn().mockRejectedValue(new Error("permission denied"));
      Object.assign(navigator, { clipboard: { writeText } });
      renderModal(
        <ConnectedAgentModal
          agent="claude-code"
          cfg={cfg()}
          cogniInstance={null}
          sessions={[session({ session_id: "claude_copyfail" })]}
          connections={[connection()]}
          connectionsTruncated={false}
          steps={STEPS}
          currentStep={0}
          onClose={jest.fn()}
          onStepSelect={jest.fn()}
        />,
      );

      expect(() => fireEvent.click(screen.getByLabelText("Copy session id"))).not.toThrow();
    });

    it("does nothing when the clipboard API is unavailable", () => {
      Object.assign(navigator, { clipboard: undefined });
      renderModal(
        <ConnectedAgentModal
          agent="claude-code"
          cfg={cfg()}
          cogniInstance={null}
          sessions={[session({ session_id: "claude_noclipboard" })]}
          connections={[connection()]}
          connectionsTruncated={false}
          steps={STEPS}
          currentStep={0}
          onClose={jest.fn()}
          onStepSelect={jest.fn()}
        />,
      );

      expect(() => fireEvent.click(screen.getByLabelText("Copy session id"))).not.toThrow();
    });
  });
});
