"use client";

import { useCallback, useState, type ReactElement } from "react";
import { useCogniInstance } from "@/modules/tenant/TenantProvider";
import { useOsPreference } from "@/ui/layout/OsPreferenceContext";
import type { SetupConnectorCfg } from "@/modules/integrations/types";
import { isAgentIntegrationKey } from "@/modules/agents/getAgentConnections";
import { useDashboardTelemetry } from "@/app/(app)/dashboard/hooks/useDashboardTelemetry";
import { SetupWizardModal } from "./SetupWizardModal";
import { ConnectedAgentModal } from "./ConnectedAgentModal";

interface AgentSetupModalProps {
  cards: SetupConnectorCfg[];
  /** card.key → seen-recently. Cards with no entry here have no detection
   *  signal and are treated as not connected. */
  connectedKeys: Record<string, boolean>;
  /** The card whose modal is open, or null for none. Not found in `cards`
   *  renders nothing — the caller owns which grid `activeKey` belongs to. */
  activeKey: string | null;
  onClose: () => void;
  onStepSelect?: (index: number) => void;
  onCopied?: (block: "terminal" | "code", step: number) => void;
}

/**
 * Which modal a card opens, and everything that modal needs (steps, current
 * step, the shared telemetry poll) — factored out of the Integrations page so
 * the dashboard's memory graph can open the exact same modal for the exact
 * same reason a card click does, instead of a second copy of this choice.
 */
export function AgentSetupModal({ cards, connectedKeys, activeKey, onClose, onStepSelect, onCopied }: AgentSetupModalProps): ReactElement | null {
  const { cogniInstance, serviceUrl, apiKey, isInitializing } = useCogniInstance();
  const { os } = useOsPreference();
  // Same react-query cache entry every other telemetry reader shares (keyed
  // by tenant, not by call site) — not a second poll.
  const { sessions, connections, connectionsTruncated } = useDashboardTelemetry();
  const [stepIndexMap, setStepIndexMap] = useState<Partial<Record<string, number>>>({});
  // Keep the setup tab in view if a connection appears while this modal is open.
  const [connectedOnOpen] = useState(() => Boolean(connectedKeys[activeKey ?? ""]));

  const activeCfg = cards.find((c) => c.key === activeKey);

  const selectStep = useCallback((index: number) => {
    if (!activeKey) return;
    onStepSelect?.(index);
    setStepIndexMap((prev) => ({ ...prev, [activeKey]: index }));
  }, [activeKey, onStepSelect]);

  if (!activeCfg) return null;

  const baseUrl = serviceUrl || "https://your-tenant.aws.cognee.ai";
  const resolvedKey = apiKey || "your-api-key";
  // See AgentConnectionSection: the gate is "no usable key yet", not just
  // "still initializing" — otherwise the placeholder key above can be copied
  // into ~/.cognee/.env over a real one.
  const steps = activeCfg.buildSteps(baseUrl, resolvedKey, isInitializing || !apiKey, os);
  const currentStep = stepIndexMap[activeCfg.key] ?? 0;
  // A connected plugin agent (claude-code/codex) opens the connection summary
  // instead of the install steps — those steps ask the user to do something
  // they've already done, which is exactly the contradiction this modal split
  // exists to remove. Every other card (MCP connectors, upload) keeps the one
  // wizard it always had.
  const activeAgentKey = isAgentIntegrationKey(activeCfg.key) ? activeCfg.key : null;
  const showConnectedModal = activeAgentKey !== null && connectedKeys[activeAgentKey] === true;

  if (showConnectedModal && activeAgentKey) {
    return (
      <ConnectedAgentModal
        agent={activeAgentKey}
        cfg={activeCfg}
        cogniInstance={cogniInstance}
        sessions={sessions}
        connections={connections}
        connectionsTruncated={connectionsTruncated}
        steps={steps}
        currentStep={currentStep}
        onClose={onClose}
        onStepSelect={selectStep}
        onCopied={onCopied}
        initialTab={connectedOnOpen ? "status" : "setup"}
      />
    );
  }
  return (
    <SetupWizardModal
      cfg={activeCfg}
      steps={steps}
      currentStep={currentStep}
      cogniInstance={cogniInstance}
      onClose={onClose}
      onStepSelect={selectStep}
      onCopied={onCopied}
    />
  );
}
