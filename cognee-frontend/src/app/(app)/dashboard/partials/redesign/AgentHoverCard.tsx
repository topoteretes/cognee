/** The public memory diagram is a static preview; it never displays agent
 * hover content, which depends on cloud-only usage telemetry. */
import type { ReactNode } from "react";
import type { AgentSummary } from "@/app/(app)/dashboard/agentSummary";

export function AgentHoverCard(_props: {
  agentName: string;
  icon: ReactNode;
  channel: string;
  summary: AgentSummary;
  onOpenConnection: () => void;
}): null {
  return null;
}
