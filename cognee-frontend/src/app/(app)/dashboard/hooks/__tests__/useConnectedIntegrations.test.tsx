import { renderHook, waitFor } from "@testing-library/react";
import type { SessionRow } from "@/modules/sessions/getSessions";

const mockUseCogniInstance = jest.fn();
const mockGetConnectedIntegrations = jest.fn();
const mockPersistConnectedIntegrations = jest.fn();

jest.mock("@/modules/tenant/TenantProvider", () => ({
  useCogniInstance: () => mockUseCogniInstance(),
}));

jest.mock("@/utils/browserStorage", () => ({
  getConnectedIntegrations: (...args: unknown[]) => mockGetConnectedIntegrations(...args),
  setConnectedIntegrations: (...args: unknown[]) => mockPersistConnectedIntegrations(...args),
}));

import { useConnectedIntegrations } from "../useConnectedIntegrations";

const mockFetch = jest.fn();

function session(sessionId: string): SessionRow {
  return {
    session_id: sessionId,
    user_id: "user-1",
    dataset_id: null,
    status: "running",
    effective_status: "running",
    started_at: null,
    last_activity_at: null,
    ended_at: null,
    tokens_in: 0,
    tokens_out: 0,
    cost_usd: 0,
    error_count: 0,
    last_model: null,
  };
}

beforeEach(() => {
  mockUseCogniInstance.mockReturnValue({ cogniInstance: { fetch: mockFetch } });
  mockGetConnectedIntegrations.mockReturnValue({});
  mockFetch.mockResolvedValue({
    ok: true,
    json: async () => ({ agents: [] }),
  });
});

describe("useConnectedIntegrations", () => {
  it("marks Codex connected from an active connection before a session exists", async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({ agents: [{ session_id: "codex_live-session" }] }),
    });

    const { result } = renderHook(() => useConnectedIntegrations([], "tenant-1"));

    await waitFor(() => expect(result.current.codex).toBe(true));
    expect(mockFetch).toHaveBeenCalledWith("/v1/agents/connections?active_only=true");
    expect(mockPersistConnectedIntegrations).toHaveBeenCalledWith("tenant-1", { codex: true });
  });

  it("keeps detecting integrations from sessions when the connection endpoint fails", async () => {
    mockFetch.mockRejectedValue(new Error("endpoint unavailable"));

    const { result } = renderHook(() =>
      useConnectedIntegrations([session("cc_existing-session")], "tenant-1"),
    );

    await waitFor(() => expect(result.current["claude-code"]).toBe(true));
  });
});
