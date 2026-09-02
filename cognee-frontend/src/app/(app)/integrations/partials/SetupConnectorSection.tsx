"use client";

import { useCallback, useState, type ReactElement } from "react";
import { useCogniInstance } from "@/modules/tenant/TenantProvider";
import type { SetupConnectorCfg } from "@/modules/integrations/types";
import SetupConnectorCard from "./SetupConnectorCard";
import SetupWizardModal from "./SetupWizardModal";

interface SetupConnectorSectionProps {
  cards: SetupConnectorCfg[];
  /** card.key → seen-recently, from useAgentConnectionStatus. Cards with no
   *  entry here have no detection signal (e.g. VS Code, Cursor) and show no badge. */
  connectedKeys?: Record<string, boolean>;
}

export default function SetupConnectorSection({ cards, connectedKeys = {} }: SetupConnectorSectionProps): ReactElement {
  const { serviceUrl, apiKey, isInitializing } = useCogniInstance();
  const [activeKey, setActiveKey] = useState<string | null>(null);
  const [stepIndexMap, setStepIndexMap] = useState<Partial<Record<string, number>>>({});

  const baseUrl = serviceUrl || "https://your-tenant.aws.cognee.ai";
  const resolvedKey = apiKey || "your-api-key";

  const activeCfg = cards.find(c => c.key === activeKey);
  const activeSteps = activeCfg ? activeCfg.buildSteps(baseUrl, resolvedKey, isInitializing) : [];
  const currentStep = activeKey ? (stepIndexMap[activeKey] ?? 0) : 0;

  const closeModal = useCallback(() => setActiveKey(null), []);

  const selectStep = useCallback((index: number) => {
    if (!activeKey) return;
    setStepIndexMap(prev => ({ ...prev, [activeKey]: index }));
  }, [activeKey]);

  return (
    <>
      <style>{`
        @keyframes aci-check { 0%{transform:scale(0.4);opacity:0} 100%{transform:scale(1);opacity:1} }
        @keyframes aci-popup { 0%{opacity:0;transform:scale(0.97) translateY(6px)} 100%{opacity:1;transform:scale(1) translateY(0)} }
        .aci-card:hover { background: rgba(255,255,255,0.09) !important; border-color: rgba(255,255,255,0.18) !important; }
        .aci-step-row:hover { background: rgba(255,255,255,0.04); }
        .aci-step-row[data-active="true"]:hover { background: transparent; }
        /* Same track sizing as the Data sources grid, so both sections share
           card width and a short list (e.g. 2 automation platforms) packs left
           instead of stretching to fill the row. */
        .int-agent-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
          gap: 16px;
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
            onOpen={() => {
              setActiveKey(card.key);
              if (stepIndexMap[card.key] === undefined) setStepIndexMap(s => ({ ...s, [card.key]: 0 }));
            }}
          />
        ))}
      </div>

      {activeKey && activeCfg && (
        <SetupWizardModal
          cfg={activeCfg}
          steps={activeSteps}
          currentStep={currentStep}
          onClose={closeModal}
          onStepSelect={selectStep}
        />
      )}
    </>
  );
}
