"use client";

import type { ReactElement } from "react";
import { useRouter } from "next/navigation";
import type { SetupConnectorCfg, StepDef } from "@/modules/integrations/types";
import InlineCodeBlock from "./InlineCodeBlock";

interface SetupWizardModalProps {
  cfg: SetupConnectorCfg;
  steps: StepDef[];
  currentStep: number;
  onClose: () => void;
  onStepSelect: (index: number) => void;
}

export default function SetupWizardModal({ cfg, steps, currentStep, onClose, onStepSelect }: SetupWizardModalProps): ReactElement {
  const router = useRouter();

  return (
    <div role="dialog" aria-modal="true" style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.35)", backdropFilter: "blur(4px)", WebkitBackdropFilter: "blur(4px)", zIndex: 200, display: "flex", alignItems: "center", justifyContent: "center", padding: 16 }} onClick={onClose}>
      <div className="aci-popup" onClick={(e) => e.stopPropagation()} style={{ background: "rgba(15,15,15,0.92)", backdropFilter: "blur(16px)", borderRadius: 14, width: 520, maxWidth: "100%", boxShadow: "0 20px 60px rgba(0,0,0,0.6), 0 0 0 1px rgba(255,255,255,0.1)", overflow: "hidden", animation: "aci-popup 200ms cubic-bezier(0.22,1,0.36,1) forwards" }}>

        <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "16px 20px", borderBottom: "1px solid rgba(255,255,255,0.1)" }}>
          <div style={{ width: 24, height: 24, flexShrink: 0, display: "flex", alignItems: "center", justifyContent: "center" }}>{cfg.icon}</div>
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{ fontSize: 15, fontWeight: 700, color: "#EDECEA", lineHeight: "20px" }}>Connect {cfg.name}</div>
            <div style={{ fontSize: 12, color: "rgba(237,236,234,0.45)", marginTop: 1 }}>Step {currentStep + 1} of {steps.length}</div>
          </div>
          <button onClick={onClose} style={{ background: "none", border: "none", color: "rgba(237,236,234,0.65)", cursor: "pointer", padding: 4, borderRadius: 6, lineHeight: 1 }}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
          </button>
        </div>

        {steps.map((step, i) => {
          const isStepActive = currentStep === i;
          const isDone = i < currentStep;
          return (
            <div key={i} className="aci-step-row" data-active={isStepActive ? "true" : undefined} onClick={() => onStepSelect(i)} style={{ borderBottom: i < steps.length - 1 ? "1px solid rgba(255,255,255,0.07)" : "none", cursor: isStepActive ? "default" : "pointer" }}>
              <div style={{ display: "flex", alignItems: "center", gap: 12, padding: isStepActive ? "14px 20px 0" : "14px 20px" }}>
                <div style={{ width: 24, height: 24, borderRadius: "50%", flexShrink: 0, display: "flex", alignItems: "center", justifyContent: "center", background: isDone ? "#DCFCE7" : isStepActive ? "#6510F4" : "#F4F4F5", transition: "background 200ms" }}>
                  {isDone
                    ? <svg width="10" height="10" viewBox="0 0 16 16" fill="none" style={{ animation: "aci-check 220ms cubic-bezier(0.22,1,0.36,1) forwards" }}><path d="M3 8.5L6.5 12L13 5" stroke="#16A34A" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round"/></svg>
                    : <span style={{ fontSize: 11, fontWeight: 700, color: isStepActive ? "#fff" : "#A1A1AA", lineHeight: 1 }}>{i + 1}</span>}
                </div>
                <span style={{ flex: 1, fontSize: 14, fontWeight: isStepActive ? 500 : 400, color: isDone ? "rgba(237,236,234,0.4)" : isStepActive ? "#EDECEA" : "rgba(237,236,234,0.35)" }}>
                  {step.title}
                </span>
                {isDone && <span style={{ fontSize: 10, fontWeight: 700, letterSpacing: "0.06em", textTransform: "uppercase", background: "#DCFCE7", color: "#15803D", borderRadius: 100, padding: "2px 8px", flexShrink: 0 }}>Done</span>}
              </div>
              <div style={{ display: "grid", gridTemplateRows: isStepActive ? "1fr" : "0fr", opacity: isStepActive ? 1 : 0, transition: "grid-template-rows 260ms ease, opacity 200ms ease" }}>
                <div style={{ overflow: "hidden" }}>
                  <div onClick={(e) => e.stopPropagation()} style={{ padding: "10px 20px 18px 56px" }}>
                    {step.description && <p style={{ fontSize: 13, color: "rgba(237,236,234,0.55)", margin: "0 0 12px", lineHeight: 1.6, whiteSpace: "pre-line" }}>{step.description}</p>}
                    {step.code && <InlineCodeBlock code={step.code} toCopy={step.codeToCopy} loading={step.loading} />}
                    {step.codeBlocks && (
                      <div style={{ display: "flex", flexDirection: "column", gap: step.codeBlocks.some(cb => cb.label) ? 14 : 8 }}>
                        {step.codeBlocks.map((cb, j) => (
                          cb.label ? (
                            <div key={j}>
                              <div style={{ fontSize: 13, fontWeight: 600, color: "#EDECEA", marginBottom: 6 }}>{cb.label}</div>
                              <InlineCodeBlock code={cb.code} toCopy={cb.codeToCopy} loading={cb.loading} />
                            </div>
                          ) : (
                            <InlineCodeBlock key={j} code={cb.code} toCopy={cb.codeToCopy} loading={cb.loading} />
                          )
                        ))}
                      </div>
                    )}
                    {step.content}
                    {i < steps.length - 1
                      ? <p style={{ margin: "10px 0 0", fontSize: 12, color: "#C8C8C8" }}>Click step {i + 2} when ready ↓</p>
                      : <button onClick={(e) => { e.stopPropagation(); onClose(); router.push("/sessions"); }} style={{ marginTop: 12, display: "inline-flex", alignItems: "center", gap: 5, background: "none", border: "1px solid rgba(255,255,255,0.15)", borderRadius: 8, padding: "7px 14px", fontSize: 13, fontWeight: 500, color: "rgba(237,236,234,0.7)", fontFamily: "inherit", cursor: "pointer" }}>Go to Sessions →</button>}
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
