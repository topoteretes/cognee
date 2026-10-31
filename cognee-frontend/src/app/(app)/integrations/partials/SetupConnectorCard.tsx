"use client";

import type { ReactElement } from "react";
import type { SetupConnectorCfg } from "@/modules/integrations/types";
import { isAgentIntegrationKey } from "@/modules/agents/getAgentConnections";
import { agentStatusText } from "@/app/(app)/dashboard/agentNodes";

interface SetupConnectorCardProps {
  card: SetupConnectorCfg;
  isActive: boolean;
  hasSignal: boolean;
  isConnected: boolean;
  /** Newest known activity, plugin agents only (claude-code/codex). See
   *  lastSeenByAgent. Absent when only the sticky registry, not a
   *  connection or session row, says connected, the card then shows "Set
   *  up" with no last-seen line rather than inventing one. */
  lastSeenAt?: string;
  /** A session of this agent is running right now, see runningAgents. The
   *  same answer the dashboard's "live" node reads. */
  running?: boolean;
  /** Telemetry's first poll hasn't answered yet — see AciCard's own prop of
   *  the same name. Only ever downgrades an unconfirmed "not connected"
   *  reading to "Checking…", never masks a real "Connected" (the sticky
   *  registry `isConnected` already reads can answer before the first poll
   *  resolves). */
  loading?: boolean;
  /** The clock "active now" is measured against (see agentStatusText's
   *  freshness threshold) — a real prop rather than `new Date()` read
   *  inline, so a test can pin it against a fixed `lastSeenAt`. */
  now?: Date;
  onOpen: () => void;
}

export default function SetupConnectorCard({ card, isActive, hasSignal, isConnected, lastSeenAt, running = false, loading = false, now = new Date(), onOpen }: SetupConnectorCardProps): ReactElement {
  // Only claude-code/codex have a registration to distinguish "set up" from
  // "running right now", every MCP connector on this page keeps the plain
  // "Connected" wording it always had.
  const isPluginAgent = isAgentIntegrationKey(card.key);
  // Same wording function and same running-session answer the graph's node
  // uses, so "Live" here and on the dashboard mean one thing. A non-plugin
  // MCP connector has no lastSeenAt or running session to report, so the
  // ternaries below collapse to the plain "Connected" it always showed.
  const pluginStatus = agentStatusText(
    running ? "live" : isConnected ? "connected" : loading ? "unknown" : "disconnected",
    lastSeenAt ?? null,
    now,
  );
  const ctaLabel = isConnected && isPluginAgent ? "View connection" : card.cta;
  return (
    <button
      className="aci-card"
      // Styled in SetupConnectorSection: opaque, with its contents dimmed until
      // hovered, so the agents already talking to Cognee stand out.
      data-dim={!isConnected && !isActive ? "true" : undefined}
      data-connected={isConnected && !isActive ? "true" : undefined}
      onClick={onOpen}
      style={{
        height: "100%",
        background: isActive ? "rgba(188,155,255,0.20)" : "rgba(255,255,255,0.06)",
        backdropFilter: "blur(12px)",
        // Faint badge green on a connected agent, so it reads as switched on.
        border: `1px solid ${isActive ? "rgba(188,155,255,0.35)" : isConnected ? "rgba(34,197,94,0.3)" : "rgba(255,255,255,0.1)"}`,
        borderRadius: 12,
        padding: 20,
        cursor: "pointer",
        textAlign: "left",
        display: "flex",
        flexDirection: "column",
        gap: 12,
        transition: "background 150ms, border-color 150ms",
      }}
    >
      {/* Title on its own row, status below it: sharing one row truncated
          the name ("Cla...", "C...") on this grid's narrower cards. */}
      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12, minWidth: 0 }}>
          <div style={{ width: 40, height: 40, borderRadius: 10, background: "rgba(255,255,255,0.08)", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
            {card.icon}
          </div>
          {/* Wraps rather than truncates. The badge beside it is
              flexShrink: 0, so the name absorbed every shortfall and
              "Claude Code" read as "Cla…" — the one word on this card that
              has to survive. */}
          <span style={{ fontSize: 16, fontWeight: 500, color: "#EDECEA", fontFamily: '"TWKLausanne", sans-serif', overflowWrap: "anywhere" }}>
            {card.name}
          </span>
        </div>
        {hasSignal && (
          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            {isConnected ? (
              <span style={{ alignSelf: "flex-start", display: "flex", alignItems: "center", gap: 5, background: "rgba(34,197,94,0.14)", color: "#22C55E", fontSize: 11, fontWeight: 600, padding: "2px 8px", borderRadius: 999 }}>
                <span style={{ width: 6, height: 6, borderRadius: "50%", background: "#22C55E" }} />{pluginStatus.badge}
              </span>
            ) : (
              <span style={{ alignSelf: "flex-start", display: "flex", alignItems: "center", gap: 5, background: loading ? "rgba(255,255,255,0.06)" : "rgba(239,68,68,0.15)", color: loading ? "rgba(237,236,234,0.65)" : "#FF8A8A", fontSize: 11, fontWeight: 600, padding: "2px 8px", borderRadius: 999 }}>
                {!loading && <span style={{ width: 6, height: 6, borderRadius: "50%", background: "#EF4444" }} />}{pluginStatus.badge}
              </span>
            )}
            {isConnected && pluginStatus.detail && (
              <span style={{ fontSize: 11, color: "rgba(237,236,234,0.45)" }}>
                {pluginStatus.detail.charAt(0).toUpperCase() + pluginStatus.detail.slice(1)}
              </span>
            )}
          </div>
        )}
      </div>


      <div style={{ marginTop: "auto" }}>
        <span style={{ display: "inline-flex", alignItems: "center", background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.18)", borderRadius: 8, padding: "6px 14px", fontSize: 13, fontWeight: 600, color: "#EDECEA" }}>
          {ctaLabel}
        </span>
      </div>
    </button>
  );
}
