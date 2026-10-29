import { renderHook } from "@testing-library/react";
import { useChecklistCompletionTracking } from "../useChecklistCompletionTracking";
import type { Agent } from "@/modules/agents/getAgents";
import type { PipelineRun, SessionRow } from "@/modules/sessions/getSessions";

const mockTrackEvent = jest.fn();
jest.mock("@/modules/analytics", () => ({ trackEvent: (...args: unknown[]) => mockTrackEvent(...args) }));

function run(overrides: Partial<PipelineRun> = {}): PipelineRun {
  return {
    id: "r1", pipeline_name: "cognify", status: "COMPLETED", dataset_id: null, dataset_name: null,
    owner_id: null, owner_email: null, created_at: null, pipeline_run_id: "pr-1", kind: "pipeline",
    operation_name: null, origin: null, outcome: "succeeded", error_class: null, tokens_in: null, tokens_out: null,
    started_at: null, ended_at: null, user_id: null, session_ids: [], parent_operation_id: null,
    background: null, ...overrides,
  };
}

function session(overrides: Partial<SessionRow> = {}): SessionRow {
  return {
    session_id: "s1", user_id: "u1", dataset_id: null, status: "completed", effective_status: "completed",
    started_at: null, last_activity_at: null, ended_at: null, tokens_in: 0, tokens_out: 0, cost_usd: 0,
    error_count: 0, last_model: null, ...overrides,
  };
}

function agent(overrides: Partial<Agent> = {}): Agent {
  return { id: "a1", email: "a@example.com", agent_type: "claude-code", agent_short_id: "cc1", is_agent: true, is_default: false, status: "active", ...overrides };
}

function input(overrides: Partial<Parameters<typeof useChecklistCompletionTracking>[0]> = {}) {
  return {
    runs: [] as PipelineRun[],
    sessions: [] as SessionRow[],
    agents: [] as Agent[],
    telemetryNotReady: false,
    agentsNotReady: false,
    ...overrides,
  };
}

beforeEach(() => {
  mockTrackEvent.mockClear();
});

describe("upload_document and first_query, independent of the agent roster", () => {
  it("still fires on a pod that never allowlists the agents route", () => {
    // (a): agentsError used to be folded into the one shared gate, so a 404 on
    // GET /v1/activity/agents (never retried, never refetched) left the entire
    // checklist permanently dark — including these two items, which read
    // runs/sessions and have nothing to do with the roster.
    const { rerender } = renderHook(
      (props) => useChecklistCompletionTracking(props),
      { initialProps: input({ agentsNotReady: true }) },
    );

    rerender(input({ runs: [run()], sessions: [session()], agentsNotReady: true }));

    expect(mockTrackEvent).toHaveBeenCalledWith(expect.objectContaining({
      additionalProperties: { item: "upload_document" },
    }));
    expect(mockTrackEvent).toHaveBeenCalledWith(expect.objectContaining({
      additionalProperties: { item: "first_query" },
    }));
  });

  it("does not seed off an empty array while telemetry has errored", () => {
    // (b): a first-load telemetry failure used to seed prevChecklistDone off
    // empty runs/sessions (loading goes false immediately on an error), so the
    // next successful poll of a workspace that finished both items long ago
    // fired two spurious completions.
    const { rerender } = renderHook(
      (props) => useChecklistCompletionTracking(props),
      { initialProps: input({ telemetryNotReady: true }) },
    );

    // Telemetry recovers with a workspace that already has history.
    rerender(input({ runs: [run()], sessions: [session()] }));

    expect(mockTrackEvent).not.toHaveBeenCalled();
  });

  it("fires exactly once when an item newly completes after telemetry recovers", () => {
    const { rerender } = renderHook(
      (props) => useChecklistCompletionTracking(props),
      { initialProps: input({ telemetryNotReady: true }) },
    );

    rerender(input()); // telemetry answers: nothing done yet
    rerender(input({ runs: [run()] })); // upload_document newly completes
    rerender(input({ runs: [run()] })); // unchanged — must not refire

    expect(mockTrackEvent).toHaveBeenCalledTimes(1);
  });

  it("fires first_query exactly once too, not on every render it stays true", () => {
    const { rerender } = renderHook(
      (props) => useChecklistCompletionTracking(props),
      { initialProps: input({ telemetryNotReady: true }) },
    );

    rerender(input());
    rerender(input({ sessions: [session()] }));
    rerender(input({ sessions: [session()] }));

    expect(mockTrackEvent).toHaveBeenCalledTimes(1);
  });
});

describe("upload_document counts only real pipeline runs", () => {
  it("does not count an operation row (recall/search/remember) as an upload", () => {
    // Operation rows mean the agent talked to memory, not that anything was
    // uploaded. `runs.length > 0` would count them; only `kind === "pipeline"`
    // is correct here.
    const { rerender } = renderHook(
      (props) => useChecklistCompletionTracking(props),
      { initialProps: input() },
    );

    rerender(input({ runs: [run({ kind: "operation", pipeline_name: null })] }));

    expect(mockTrackEvent).not.toHaveBeenCalled();
  });
});

describe("connect_agent, independent of telemetry", () => {
  it("still fires while telemetry has not answered yet", () => {
    const { rerender } = renderHook(
      (props) => useChecklistCompletionTracking(props),
      { initialProps: input({ telemetryNotReady: true }) },
    );

    rerender(input({ telemetryNotReady: true, agents: [agent()] }));

    expect(mockTrackEvent).toHaveBeenCalledWith(expect.objectContaining({
      additionalProperties: { item: "connect_agent" },
    }));
  });

  it("does not count the workspace's own default agent as a connection", () => {
    const { rerender } = renderHook(
      (props) => useChecklistCompletionTracking(props),
      { initialProps: input() },
    );

    rerender(input({ agents: [agent({ is_default: true })] }));

    expect(mockTrackEvent).not.toHaveBeenCalled();
  });

  it("waits for the roster to actually answer before seeding", () => {
    const { rerender } = renderHook(
      (props) => useChecklistCompletionTracking(props),
      { initialProps: input({ agentsNotReady: true, agents: [agent()] }) },
    );

    // The roster resolves, still reporting the same connected agent — must not
    // read as newly-completed.
    rerender(input({ agents: [agent()] }));

    expect(mockTrackEvent).not.toHaveBeenCalled();
  });

  it("does not treat a change seen while not-ready as the newly-completed transition", () => {
    // The roster answers "connected" only once agentsNotReady actually goes
    // false. If a not-ready render were allowed to seed or compare, the
    // agent appearing while the roster was still loading would already look
    // like a change by the time readiness catches up, firing a completion for
    // a state transition nothing here was ready to observe yet.
    const { rerender } = renderHook(
      (props) => useChecklistCompletionTracking(props),
      { initialProps: input({ agentsNotReady: true, agents: [] }) },
    );

    rerender(input({ agentsNotReady: true, agents: [agent()] }));

    expect(mockTrackEvent).not.toHaveBeenCalled();
  });

  it("fires exactly once when connect_agent newly completes, not again on the next render", () => {
    const { rerender } = renderHook(
      (props) => useChecklistCompletionTracking(props),
      { initialProps: input({ agentsNotReady: true }) },
    );

    rerender(input()); // roster answers: no agent connected yet
    rerender(input({ agents: [agent()] })); // newly connects
    rerender(input({ agents: [agent()] })); // unchanged — must not refire

    expect(mockTrackEvent).toHaveBeenCalledTimes(1);
  });
});
