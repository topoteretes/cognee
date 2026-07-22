"use client";

import React from "react";
import type { AciStepDef, AciAgentKey } from "./agentConnectionSteps";
import { InlineCodeBlock } from "./InlineCodeBlock";
import { SkillCopyBlock } from "./SkillCopyBlock";

interface AciStepRowProps {
  step: AciStepDef;
  index: number;
  total: number;
  isActive: boolean;
  isDone: boolean;
  card: AciAgentKey;
  onClick: () => void;
  onNavigate: (path: string) => void;
}

export function AciStepRow({
  step,
  index,
  total,
  isActive,
  isDone,
  card,
  onClick,
  onNavigate,
}: AciStepRowProps): React.ReactElement {
  return (
    <div
      className="aci-step-row"
      data-active={isActive ? "true" : undefined}
      style={{ borderBottom: index < total - 1 ? "1px solid rgba(255,255,255,0.07)" : "none" }}
    >
      {/* Row header — keyboard-accessible button; disabled on the active step */}
      <button
        onClick={onClick}
        disabled={isActive}
        aria-expanded={isActive}
        style={{
          display: "flex", alignItems: "center", gap: 12,
          padding: isActive ? "14px 20px 0" : "14px 20px",
          width: "100%", background: "none", border: "none",
          cursor: isActive ? "default" : "pointer",
          fontFamily: "inherit", textAlign: "left",
        }}
      >
        <div style={{
          width: 24, height: 24, borderRadius: "50%", flexShrink: 0,
          display: "flex", alignItems: "center", justifyContent: "center",
          background: isDone ? "rgba(34,197,94,0.18)" : isActive ? "var(--color-cognee-purple)" : "rgba(255,255,255,0.1)",
          transition: "background 200ms ease",
        }}>
          {isDone ? (
            <svg width="10" height="10" viewBox="0 0 16 16" fill="none" style={{ animation: "aci-check 220ms cubic-bezier(0.22,1,0.36,1) forwards" }}>
              <path d="M3 8.5L6.5 12L13 5" stroke="#22C55E" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          ) : (
            <span style={{ fontSize: 11, fontWeight: 700, color: isActive ? "#fff" : "rgba(237,236,234,0.65)", lineHeight: 1 }}>
              {index + 1}
            </span>
          )}
        </div>
        <span style={{ flex: 1, fontSize: 14, fontWeight: isActive ? 500 : 400, color: isDone ? "rgba(237,236,234,0.45)" : isActive ? "#EDECEA" : "rgba(237,236,234,0.30)" }}>
          {step.title}
        </span>
        {isDone && (
          <span style={{ fontSize: 10, fontWeight: 700, letterSpacing: "0.06em", textTransform: "uppercase", background: "rgba(34,197,94,0.18)", color: "#22C55E", borderRadius: 100, padding: "2px 8px", flexShrink: 0, animation: "aci-check 200ms ease forwards" }}>
            Done
          </span>
        )}
      </button>

      {/* Expanded content — grid-template-rows animates height without JS measurement */}
      <div
        className="aci-step-body"
        style={{
          display: "grid",
          gridTemplateRows: isActive ? "1fr" : "0fr",
          opacity: isActive ? 1 : 0,
          transition: "grid-template-rows 260ms ease, opacity 200ms ease",
        }}
      >
        <div style={{ overflow: "hidden" }}>
          <div onClick={(e) => e.stopPropagation()} style={{ padding: "10px 20px 18px 56px" }}>
            {step.description && (
              <p style={{ fontSize: 13, color: "rgba(237,236,234,0.65)", margin: "0 0 12px", lineHeight: 1.6 }}>
                {step.description}
              </p>
            )}
            {step.code && (
              <InlineCodeBlock code={step.code} toCopy={step.codeToCopy} loading={step.loading} card={card} block={step.title} />
            )}
            {step.codeBlocks && (
              <div style={{ display: "flex", flexDirection: "column", gap: step.codeBlocks.some((cb) => cb.label) ? 14 : 8 }}>
                {step.codeBlocks.map((cb, j) =>
                  cb.label ? (
                    <div key={j}>
                      <div style={{ fontSize: 13, fontWeight: 600, color: "#EDECEA", marginBottom: 6 }}>{cb.label}</div>
                      <InlineCodeBlock code={cb.code} toCopy={cb.toCopy} card={card} block={cb.label} />
                    </div>
                  ) : (
                    <InlineCodeBlock key={j} code={cb.code} toCopy={cb.toCopy} card={card} block={step.title} />
                  ),
                )}
              </div>
            )}
            {step.skillPath && step.skillContent && (
              <SkillCopyBlock path={step.skillPath} content={step.skillContent} card={card} />
            )}
            {index < total - 1 ? (
              <p style={{ margin: "10px 0 0", fontSize: 12, color: "rgba(237,236,234,0.65)" }}>
                Click step {index + 2} when ready ↓
              </p>
            ) : (
              <button
                onClick={(e) => { e.stopPropagation(); onNavigate("/sessions"); }}
                style={{ marginTop: 12, display: "inline-flex", alignItems: "center", gap: 5, background: "none", border: "1px solid rgba(255,255,255,0.2)", borderRadius: 8, padding: "7px 14px", fontSize: 13, fontWeight: 500, color: "#EDECEA", fontFamily: "inherit", cursor: "pointer" }}
              >
                Go to Sessions →
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
