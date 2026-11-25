/** Public dashboard placeholder: the cloud graph's agent telemetry is not
 * rendered by the public MemoryFlowDiagram override. Keep its view model
 * shape so the shared OverviewPage can still type-check. */
import type { AgentConnection, AgentIntegrationKey } from "@/modules/agents/getAgentConnections";
import type { SessionRow } from "@/modules/sessions/getSessions";

export interface AgentSummary {
  badge: string;
  pill: "Active" | "Connected" | "Checking" | "Not connected";
  lines: string[];
  tokensPerHour: number[];
  tokensAvailable: boolean;
  savedUsd: number | null;
  tokensTotal: number;
  windowIncomplete: boolean;
  datasets: string[];
}

export function agentSummary(
  _agent: AgentIntegrationKey,
  _connected: boolean,
  _loading: boolean,
  _connections: AgentConnection[],
  _sessions: SessionRow[],
  _now: Date,
  _sessionsTruncated: boolean,
): AgentSummary {
  return {
    badge: "",
    pill: "Checking",
    lines: [],
    tokensPerHour: [],
    tokensAvailable: false,
    savedUsd: null,
    tokensTotal: 0,
    windowIncomplete: false,
    datasets: [],
  };
}
