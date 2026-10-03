import { renderHook, waitFor } from "@testing-library/react";
import type { CogneeInstance } from "@/modules/instances/types";
import { integrationsFromSessionIds, useConnectedIntegrations } from "../useConnectedIntegrations";

beforeEach(() => {
  localStorage.clear();
});

describe("integrationsFromSessionIds", () => {
  it("detects integrations from active connection session identifiers", () => {
    expect(integrationsFromSessionIds(["codex_123", "cc_456"])).toEqual({
      codex: true,
      "claude-code": true,
    });
  });

  it("ignores missing and unrecognized session identifiers", () => {
    expect(integrationsFromSessionIds([null, "api_123", "claude_desktop_456"])).toEqual({});
  });

  it("marks an integration connected before a session row exists", async () => {
    const instance: CogneeInstance = {
      name: "test",
      instanceId: "test-instance",
      fetch: jest.fn().mockResolvedValue(
        new Response(JSON.stringify({ agents: [{ session_id: "codex_fresh" }] }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }),
      ),
    };

    const { result } = renderHook(() => useConnectedIntegrations([], "tenant-1", instance));

    await waitFor(() => expect(result.current.codex).toBe(true));
    expect(localStorage.getItem("cognee-connected-integrations-tenant-1")).toContain('"codex":true');
  });

  it("does not carry active connections across tenants", async () => {
    const instance: CogneeInstance = {
      name: "test",
      instanceId: "test-instance",
      fetch: jest
        .fn()
        .mockResolvedValueOnce(
          new Response(JSON.stringify({ agents: [{ session_id: "codex_fresh" }] }), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          }),
        )
        .mockReturnValueOnce(new Promise(() => {})),
    };

    const { result, rerender } = renderHook(
      ({ tenantId }) => useConnectedIntegrations([], tenantId, instance),
      { initialProps: { tenantId: "tenant-1" } },
    );

    await waitFor(() => expect(result.current.codex).toBe(true));
    rerender({ tenantId: "tenant-2" });

    await waitFor(() => expect(result.current.codex).toBeUndefined());
    expect(localStorage.getItem("cognee-connected-integrations-tenant-2")).toBeNull();
  });
});
