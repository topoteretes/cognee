"use client";

import { useState, type ReactElement } from "react";
import type { SetupConnectorCfg, StepDef } from "@/modules/integrations/types";
import type { CogneeInstance } from "@/modules/instances/types";
import { AGENT_INTEGRATION_DEFS } from "@/modules/agents/getAgentConnections";
import { OsToggle } from "@/ui/elements/OsToggle";
import { SetupWizardSteps } from "./SetupWizardSteps";

interface SetupWizardModalProps {
  cfg: SetupConnectorCfg;
  steps: StepDef[];
  currentStep: number;
  cogniInstance: CogneeInstance | null;
  onClose: () => void;
  onStepSelect: (index: number) => void;
  onCopied?: (block: "terminal" | "code", step: number) => void;
}

export function SetupWizardModal({ cfg, steps, currentStep, cogniInstance, onClose, onStepSelect, onCopied }: SetupWizardModalProps): ReactElement {
  // cfg.key is a plain string across all ~15 connectors this modal renders;
  // only the two that are AGENT_INTEGRATION_DEFS rows get a status line, and
  // a table lookup is what keeps that from turning into a per-agent branch.
  const agent = AGENT_INTEGRATION_DEFS.find((def) => def.key === cfg.key)?.key;
  const [expandedSteps, setExpandedSteps] = useState<ReadonlySet<number>>(() => new Set(steps.map((_, index) => index)));

  return (
    <div role="dialog" aria-modal="true" style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.35)", backdropFilter: "blur(4px)", WebkitBackdropFilter: "blur(4px)", zIndex: 200, display: "flex", alignItems: "center", justifyContent: "center", padding: 16 }} onClick={onClose}>
      <div className="aci-popup" onClick={(e) => e.stopPropagation()} style={{ background: "rgba(15,15,15,0.92)", backdropFilter: "blur(16px)", borderRadius: 0, width: 520, maxWidth: "100%", maxHeight: "calc(100dvh - 32px)", display: "flex", flexDirection: "column", boxShadow: "0 20px 60px rgba(0,0,0,0.6), 0 0 0 1px rgba(255,255,255,0.1)", overflow: "hidden", animation: "aci-popup 200ms cubic-bezier(0.22,1,0.36,1) forwards" }}>

        <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "16px 20px", borderBottom: "1px solid rgba(255,255,255,0.1)" }}>
          <div style={{ width: 24, height: 24, flexShrink: 0, display: "flex", alignItems: "center", justifyContent: "center" }}>{cfg.icon}</div>
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{ fontSize: 15, fontWeight: 700, color: "#EDECEA", lineHeight: "20px" }}>Connect {cfg.name}</div>
            <div style={{ fontSize: 12, color: "rgba(237,236,234,0.45)", marginTop: 1 }}>
              {agent ? `${steps.length} setup steps` : `Step ${currentStep + 1} of ${steps.length}`}
            </div>
          </div>
          {/* Shell-based connectors render different commands per OS; MCP ones
              ignore it, so the toggle only appears where it changes something. */}
          {cfg.osAware && <OsToggle />}
          <button onClick={onClose} aria-label="Close" style={{ background: "none", border: "none", color: "rgba(237,236,234,0.65)", cursor: "pointer", padding: 4, borderRadius: 0, lineHeight: 1 }}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
          </button>
        </div>

        <div style={{ overflowY: "auto" }}><SetupWizardSteps
          steps={steps}
          currentStep={currentStep}
          agent={agent}
          cogniInstance={cogniInstance}
          onStepSelect={onStepSelect}
          onCopied={onCopied}
          onClose={onClose}
          expandedSteps={agent ? expandedSteps : undefined}
          onStepToggle={agent ? (index) => {
            onStepSelect(index);
            setExpandedSteps((previous) => {
            const next = new Set(previous);
            if (next.has(index)) next.delete(index); else next.add(index);
            return next;
            });
          } : undefined}
        /></div>
      </div>
    </div>
  );
}
