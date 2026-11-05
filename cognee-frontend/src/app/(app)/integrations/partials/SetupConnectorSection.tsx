"use client";

import { useState, type ReactElement } from "react";
import { trackEvent } from "@/modules/analytics";
import type { SetupConnectorCfg } from "@/modules/integrations/types";
import { AgentSetupModal } from "@/ui/elements/AgentSetupModal";
import SetupConnectorCard from "./SetupConnectorCard";

interface SetupConnectorSectionProps {
  cards: SetupConnectorCfg[];
  /** Rides on every event so the agents and automation grids read apart. */
  section: "agents" | "automation";
  /** card.key → seen-recently, from useAgentConnectionStatus. Cards with no
   *  entry here have no detection signal (e.g. VS Code, Cursor) and show no badge. */
  connectedKeys?: Record<string, boolean>;
  /** card.key → newest connection `last_active_at`, plugin agents only
   *  (claude-code/codex). See useAgentConnectionStatus. */
  lastSeenByAgent?: Partial<Record<string, string>>;
  /** Plugin agents with a running session right now — see runningAgents. */
  runningAgents?: ReadonlySet<string>;
  /** Telemetry's first poll hasn't answered yet — see useAgentConnectionStatus. */
  loading?: boolean;
}

/**
 * The same agent_* events the dashboard's cards send, with pageName telling the
 * two surfaces apart, so one set-up funnel covers both.
 */
function trackSetup(eventName: string, section: string, card: string, extra: Record<string, string> = {}): void {
  void trackEvent({ pageName: "Integrations", eventName, additionalProperties: { card, section, ...extra } });
}

export default function SetupConnectorSection({ cards, section, connectedKeys = {}, lastSeenByAgent = {}, runningAgents, loading = false }: SetupConnectorSectionProps): ReactElement {
  const [activeKey, setActiveKey] = useState<string | null>(null);

  return (
    <>
      <style>{`
        @keyframes aci-check { 0%{transform:scale(0.4);opacity:0} 100%{transform:scale(1);opacity:1} }
        @keyframes aci-popup { 0%{opacity:0;transform:scale(0.97) translateY(6px)} 100%{opacity:1;transform:scale(1) translateY(0)} }
        .aci-card:hover { background: rgba(255,255,255,0.09) !important; border-color: rgba(255,255,255,0.18) !important; }
        .aci-card[data-connected="true"]:hover { border-color: rgba(34,197,94,0.45) !important; }
        .aci-card[data-dim="true"] { background: #1F1F1F !important; }
        .aci-card[data-dim="true"]:hover { background: #262626 !important; }
        .aci-card[data-dim="true"] > * { opacity: 0.6; transition: opacity 150ms; }
        .aci-card[data-dim="true"]:hover > *, .aci-card[data-dim="true"]:focus-visible > * { opacity: 1; }
        .aci-step-row:hover { background: rgba(255,255,255,0.04); }
        .aci-step-row[data-active="true"]:hover { background: transparent; }
        /* Same track sizing as the Data sources grid, so both sections share
           card width and a short list (e.g. 2 automation platforms) packs left
           instead of stretching to fill the row. auto-fill takes as many 224px
           tracks as the page is wide enough for — six at its cap, fewer as the
           window narrows. */
        .int-agent-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(224px, 1fr));
          gap: 14px;
        }
      `}</style>

      <div className="int-agent-grid">
        {cards.map((card) => (
          <SetupConnectorCard
            key={card.key}
            card={card}
            isActive={activeKey === card.key}
            hasSignal={card.key in connectedKeys}
            isConnected={connectedKeys[card.key] === true}
            lastSeenAt={lastSeenByAgent[card.key]}
            running={runningAgents?.has(card.key) ?? false}
            loading={loading}
            onOpen={() => {
              trackSetup("agent_card_clicked", section, card.key);
              setActiveKey(card.key);
            }}
          />
        ))}
      </div>

      {activeKey && <AgentSetupModal
        key={activeKey}
        cards={cards}
        connectedKeys={connectedKeys}
        activeKey={activeKey}
        onClose={() => setActiveKey(null)}
        onStepSelect={(index) => trackSetup("agent_step_viewed", section, activeKey, { step: String(index) })}
        onCopied={(block, step) => trackSetup("agent_config_copied", section, activeKey, { block, step: String(step) })}
      />}
    </>
  );
}
