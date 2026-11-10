"use client";

import type { ReactElement } from "react";
import { TrackPageView } from "@/modules/analytics";
import { AGENT_CARDS } from "@/modules/integrations/agentCards";
import { AUTOMATION_CARDS } from "@/modules/integrations/automationCards";
import SetupConnectorSection from "./partials/SetupConnectorSection";
import GoogleIntegrationsSection from "./partials/GoogleIntegrationsSection";
import MoreDataSourcesSection from "./partials/MoreDataSourcesSection";
import { useAgentConnectionStatus } from "./partials/useAgentConnectionStatus";

/**
 * Open-source integrations page. The hosted page lists connector cards that
 * talk to the cloud control plane; the SDK has no control plane, one user, and
 * its own integrations router, so the data sources here are the ones that
 * router serves (GoogleIntegrationsSection, owned by this repo).
 */
const PAGE_MAX_WIDTH = 1440;

export default function IntegrationsPage(): ReactElement {
  const { connected: agentConnectionStatus, lastSeenAt, running, loading: agentConnectionsLoading } = useAgentConnectionStatus();

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", overflowY: "auto" }}>
      <TrackPageView page="Integrations" />

      <div style={{ maxWidth: PAGE_MAX_WIDTH, margin: "0 auto", width: "100%", padding: "24px 32px 40px", display: "flex", flexDirection: "column", gap: 40 }}>

        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <div>
            <h1 style={{ fontSize: 18, fontWeight: 700, color: "#EDECEA", margin: "0 0 4px", letterSpacing: "-0.01em" }}>Agents</h1>
            <p style={{ fontSize: 14, color: "rgba(237,236,234,0.55)", margin: 0 }}>Connect your AI agents and coding tools to Cognee for persistent memory.</p>
          </div>
          <SetupConnectorSection cards={AGENT_CARDS} section="agents" connectedKeys={agentConnectionStatus} lastSeenByAgent={lastSeenAt} runningAgents={running} loading={agentConnectionsLoading} />
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <div>
            <h2 style={{ fontSize: 18, fontWeight: 700, color: "#EDECEA", margin: "0 0 4px", letterSpacing: "-0.01em" }}>Automation platforms</h2>
            <p style={{ fontSize: 14, color: "rgba(237,236,234,0.55)", margin: 0 }}>Give your automation workflows access to Cognee memory via MCP.</p>
          </div>
          <SetupConnectorSection cards={AUTOMATION_CARDS} section="automation" />
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <div>
            <h2 style={{ fontSize: 18, fontWeight: 700, color: "#EDECEA", margin: "0 0 4px", letterSpacing: "-0.01em" }}>Data sources</h2>
            <p style={{ fontSize: 14, color: "rgba(237,236,234,0.55)", margin: 0 }}>Connect your Google account to turn Drive and Gmail into searchable memory.</p>
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))", gap: 16 }}>
            <GoogleIntegrationsSection />
          </div>
        </div>

        <MoreDataSourcesSection />

      </div>
    </div>
  );
}
