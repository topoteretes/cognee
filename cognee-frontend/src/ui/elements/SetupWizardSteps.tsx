"use client";

import type { ReactElement } from "react";
import { useRouter } from "next/navigation";
import type { StepDef } from "@/modules/integrations/types";
import type { CogneeInstance } from "@/modules/instances/types";
import type { AgentIntegrationKey } from "@/modules/agents/getAgentConnections";
import { InlineCodeBlock } from "@/ui/elements/InlineCodeBlock";
import { TerminalBlock } from "@/ui/elements/AgentSetupBlocks";
import { AgentConnectionStatus } from "@/ui/elements/AgentConnectionStatus";

interface SetupWizardStepsProps {
  steps: StepDef[];
  currentStep: number;
  /** Only set for the two plugin agents — renders the live tutorial status
   *  line below the steps, same as before this was extracted. */
  agent?: AgentIntegrationKey;
  cogniInstance: CogneeInstance | null;
  onStepSelect: (index: number) => void;
  onCopied?: (block: "terminal" | "code", step: number) => void;
  onClose: () => void;
  /** When supplied, rows can be expanded independently instead of using the wizard accordion. */
  expandedSteps?: ReadonlySet<number>;
  onStepToggle?: (index: number) => void;
  connectedAfterOpen?: boolean;
}

/**
 * Numbered setup rows shared by the install wizard and the connected modal's
 * Setup tab. Plugin steps can be expanded independently.
 */
export function SetupWizardSteps({ steps, currentStep, agent, cogniInstance, onStepSelect, onCopied, onClose, expandedSteps, onStepToggle, connectedAfterOpen = false }: SetupWizardStepsProps): ReactElement {
  const router = useRouter();

  return (
    <>
      {steps.map((step, i) => {
        const isStepActive = expandedSteps ? expandedSteps.has(i) : currentStep === i;
        const isDone = !expandedSteps && i < currentStep;
        return (
          <div key={i} className="aci-step-row" data-active={isStepActive ? "true" : undefined} onClick={() => expandedSteps ? onStepToggle?.(i) : onStepSelect(i)} style={{ borderBottom: i < steps.length - 1 ? "1px solid rgba(255,255,255,0.07)" : "none", cursor: expandedSteps || !isStepActive ? "pointer" : "default" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 12, padding: isStepActive ? "14px 20px 0" : "14px 20px" }}>
              <div style={{ width: 24, height: 24, borderRadius: "50%", flexShrink: 0, display: "flex", alignItems: "center", justifyContent: "center", background: isDone ? "#DCFCE7" : isStepActive ? "var(--color-cognee-lavender)" : "#F4F4F5", transition: "background 200ms" }}>
                {isDone
                  ? <svg width="10" height="10" viewBox="0 0 16 16" fill="none" style={{ animation: "aci-check 220ms cubic-bezier(0.22,1,0.36,1) forwards" }}><path d="M3 8.5L6.5 12L13 5" stroke="#16A34A" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round"/></svg>
                  : <span style={{ fontSize: 11, fontWeight: 700, color: isStepActive ? "#fff" : "#A1A1AA", lineHeight: 1 }}>{i + 1}</span>}
              </div>
              <span style={{ flex: 1, fontSize: 14, fontWeight: isStepActive ? 500 : 400, color: isDone ? "rgba(237,236,234,0.4)" : isStepActive ? "#EDECEA" : "rgba(237,236,234,0.35)" }}>
                {step.title}
              </span>
              {isDone && <span style={{ fontSize: 11, fontWeight: 600, letterSpacing: "0.04em", textTransform: "uppercase", background: "rgba(34,197,94,0.12)", color: "#22C55E", borderRadius: 100, padding: "2px 8px", flexShrink: 0 }}>Done</span>}
            </div>
            <div aria-hidden={!isStepActive} inert={!isStepActive} style={{ display: "grid", gridTemplateRows: isStepActive ? "1fr" : "0fr", opacity: isStepActive ? 1 : 0, transition: "grid-template-rows 260ms ease, opacity 200ms ease" }}>
              <div style={{ overflow: "hidden" }}>
                <div onClick={(e) => e.stopPropagation()} style={{ padding: "10px 20px 18px 56px" }}>
                  {step.description && <p style={{ fontSize: 13, color: "rgba(237,236,234,0.55)", margin: "0 0 12px", lineHeight: 1.6, whiteSpace: "pre-line" }}>{step.description}</p>}
                  {step.lines && <TerminalBlock lines={step.lines} loading={step.loading} placeholder="Preparing your credentials…" copyLabel={step.copyLabel} onCopied={() => onCopied?.("terminal", i)} />}
                  {step.code && <InlineCodeBlock code={step.code} toCopy={step.codeToCopy} loading={step.loading} onCopy={() => onCopied?.("code", i)} />}
                  {step.codeBlocks && (
                    <div style={{ display: "flex", flexDirection: "column", gap: step.codeBlocks.some(cb => cb.label) ? 14 : 8 }}>
                      {step.codeBlocks.map((cb, j) => (
                        cb.label ? (
                          <div key={j}>
                            <div style={{ fontSize: 13, fontWeight: 600, color: "#EDECEA", marginBottom: 6 }}>{cb.label}</div>
                            <InlineCodeBlock code={cb.code} toCopy={cb.codeToCopy} loading={cb.loading} onCopy={() => onCopied?.("code", i)} />
                          </div>
                        ) : (
                          <InlineCodeBlock key={j} code={cb.code} toCopy={cb.codeToCopy} loading={cb.loading} onCopy={() => onCopied?.("code", i)} />
                        )
                      ))}
                    </div>
                  )}
                  {step.content}
                  {i < steps.length - 1
                    ? expandedSteps ? null : <p style={{ margin: "10px 0 0", fontSize: 12, color: "#C8C8C8" }}>Click step {i + 2} when ready ↓</p>
                    : <button onClick={(e) => { e.stopPropagation(); onClose(); router.push("/sessions"); }} style={{ marginTop: 12, display: "inline-flex", alignItems: "center", gap: 5, background: "none", border: "1px solid rgba(255,255,255,0.15)", borderRadius: 0, padding: "7px 14px", fontSize: 13, fontWeight: 500, color: "rgba(237,236,234,0.7)", fontFamily: "inherit", cursor: "pointer" }}>Go to Sessions →</button>}
                </div>
              </div>
            </div>
          </div>
        );
      })}
      {agent && (
        <div style={{ padding: "0 20px 16px 56px" }}>
          {connectedAfterOpen
            ? <div role="status" style={{ color: "#22C55E", fontSize: 13 }}>✓ Connected successfully</div>
            : <AgentConnectionStatus agent={agent} cogniInstance={cogniInstance} pageName="Integrations" />}
        </div>
      )}
    </>
  );
}
