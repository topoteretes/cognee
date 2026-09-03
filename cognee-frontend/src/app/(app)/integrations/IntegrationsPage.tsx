"use client";

import type { ReactElement } from "react";
import { TrackPageView } from "@/modules/analytics";
import { AGENT_CARDS } from "@/modules/integrations/agentCards";
import { AUTOMATION_CARDS } from "@/modules/integrations/automationCards";
import SetupConnectorSection from "./partials/SetupConnectorSection";
import DataSourceSection from "./partials/DataSourceSection";
import MoreDataSourcesSection from "./partials/MoreDataSourcesSection";
import { useAgentConnectionStatus } from "./partials/useAgentConnectionStatus";

export default function IntegrationsPage(): ReactElement {
  const agentConnectionStatus = useAgentConnectionStatus();

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", overflowY: "auto" }}>
      <TrackPageView page="Integrations" />

      <div style={{ maxWidth: 1042, margin: "0 auto", width: "100%", padding: "24px 32px 40px", display: "flex", flexDirection: "column", gap: 40 }}>

        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <div>
            <h1 style={{ fontSize: 18, fontWeight: 700, color: "#EDECEA", margin: "0 0 4px", letterSpacing: "-0.01em" }}>Agents</h1>
            <p style={{ fontSize: 14, color: "rgba(237,236,234,0.55)", margin: 0 }}>Connect your AI agents and coding tools to Cognee for persistent memory.</p>
          </div>
          <SetupConnectorSection cards={AGENT_CARDS} connectedKeys={agentConnectionStatus} />
        </div>

        <div style={{ height: 1, background: "rgba(255,255,255,0.08)" }} />

        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <div>
            <h2 style={{ fontSize: 18, fontWeight: 700, color: "#EDECEA", margin: "0 0 4px", letterSpacing: "-0.01em" }}>Automation platforms</h2>
            <p style={{ fontSize: 14, color: "rgba(237,236,234,0.55)", margin: 0 }}>Give your automation workflows access to Cognee memory via MCP.</p>
          </div>
          <SetupConnectorSection cards={AUTOMATION_CARDS} />
        </div>

        <div style={{ height: 1, background: "rgba(255,255,255,0.08)" }} />

        <DataSourceSection />

        <div style={{ height: 1, background: "rgba(255,255,255,0.08)" }} />

        <MoreDataSourcesSection />

      </div>
    </div>
  );
}
