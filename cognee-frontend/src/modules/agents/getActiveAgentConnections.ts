import type { CogneeInstance } from "@/modules/instances/types";

export interface ActiveAgentConnection {
  session_id: string | null;
}

interface AgentConnectionsResponse {
  agents: ActiveAgentConnection[];
}

function isAgentConnectionsResponse(value: unknown): value is AgentConnectionsResponse {
  return (
    typeof value === "object" &&
    value !== null &&
    Array.isArray((value as { agents?: unknown }).agents)
  );
}

export async function getActiveAgentConnections(
  instance: CogneeInstance,
  signal?: AbortSignal,
): Promise<ActiveAgentConnection[]> {
  try {
    const response = await instance.fetch(
      "/v1/agents/connections?active_only=true&include_sources=false&limit=500",
      { signal },
    );
    if (!response.ok) return [];

    const payload: unknown = await response.json();
    if (!isAgentConnectionsResponse(payload)) {
      console.warn("[getActiveAgentConnections] unexpected response shape");
      return [];
    }

    return payload.agents.filter(
      (connection): connection is ActiveAgentConnection =>
        typeof connection === "object" &&
        connection !== null &&
        (connection.session_id === null || typeof connection.session_id === "string"),
    );
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") return [];
    console.warn(
      "[getActiveAgentConnections] request failed:",
      error instanceof Error ? error.message : error,
    );
    return [];
  }
}
