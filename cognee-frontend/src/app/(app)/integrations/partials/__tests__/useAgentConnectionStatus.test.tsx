const mockUseTenant = jest.fn();
jest.mock("@/modules/tenant/TenantContext", () => ({
  useTenant: () => mockUseTenant(),
}));

const mockUseDashboardTelemetry = jest.fn();
jest.mock("@/app/(app)/dashboard/hooks/useDashboardTelemetry", () => ({
  useDashboardTelemetry: () => mockUseDashboardTelemetry(),
}));

import { renderHook } from "@testing-library/react";
import { useAgentConnectionStatus } from "../useAgentConnectionStatus";
import type { AgentConnection } from "@/modules/agents/getAgentConnections";
import type { SessionRow } from "@/modules/sessions/getSessions";

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

function telemetry(overrides: { sessions?: SessionRow[]; connections?: AgentConnection[]; loading?: boolean } = {}) {
  return { sessions: overrides.sessions ?? [], connections: overrides.connections ?? [], loading: overrides.loading ?? false };
}

beforeEach(() => {
  localStorage.clear();
  mockUseTenant.mockReturnValue({ tenant: { tenant_id: "t1" } });
  mockUseDashboardTelemetry.mockReturnValue(telemetry());
});

describe("useAgentConnectionStatus", () => {
  it("reads connections and sessions from the shared telemetry poll, not a fetch of its own", () => {
    mockUseDashboardTelemetry.mockReturnValue(telemetry({ connections: [connection()] }));

    const { result } = renderHook(() => useAgentConnectionStatus());

    expect(result.current.connected["claude-code"]).toBe(true);
    expect(result.current.lastSeenAt["claude-code"]).toBe("2026-09-29T12:00:00.000Z");
  });

  it("reports no last-seen entry for an agent telemetry never mentioned", () => {
    const { result } = renderHook(() => useAgentConnectionStatus());

    expect(result.current.lastSeenAt["claude-code"]).toBeUndefined();
  });

  it("updates when the shared telemetry poll returns new data, without remounting", () => {
    const { result, rerender } = renderHook(() => useAgentConnectionStatus());
    expect(result.current.connected["claude-code"]).toBe(false);

    // Models the next 15s telemetry tick landing a registration that was not
    // there before — the same query updates in place, this hook (and every
    // card reading it) must reflect that on the very next render.
    mockUseDashboardTelemetry.mockReturnValue(telemetry({ connections: [connection()] }));
    rerender();

    expect(result.current.connected["claude-code"]).toBe(true);
  });

  it("forwards telemetry's own loading flag, not a fixed value", () => {
    mockUseDashboardTelemetry.mockReturnValue(telemetry({ loading: true }));

    const { result } = renderHook(() => useAgentConnectionStatus());

    expect(result.current.loading).toBe(true);
  });
});
