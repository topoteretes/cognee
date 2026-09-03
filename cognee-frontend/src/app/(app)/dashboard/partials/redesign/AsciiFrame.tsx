"use client";

import React from "react";
import { FONT, T } from "./mono";

interface AsciiFrameProps {
  /** Panel title, e.g. "Cost". Pass null to omit the header row entirely —
   *  the panel body then owns its own top line (used by the Memory panel). */
  label: React.ReactNode | null;
  /** Right-aligned meta slot in the header, e.g. a range hint or disclaimer. */
  meta?: React.ReactNode;
  children: React.ReactNode;
  /** Optional min height so a row of panels shares a baseline. */
  minHeight?: number;
  style?: React.CSSProperties;
}

/**
 * Soft brand card used across the Overview redesign (cognee.ai idiom): a
 * subtle dark surface, hairline border, and a light header with a sentence-case
 * title plus an optional muted meta slot. Named AsciiFrame for import stability;
 * the terminal framing it started as is gone.
 */
export function AsciiFrame({ label, meta, children, minHeight, style }: AsciiFrameProps): React.ReactElement {
  return (
    <section
      style={{
        ...FONT,
        display: "flex",
        flexDirection: "column",
        minHeight,
        background: T.panel,
        border: `1px solid ${T.frame}`,
        borderRadius: 0,
        overflow: "hidden",
        ...style,
      }}
    >
      {label !== null && (
        <header
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            gap: 12,
            padding: "0 16px",
            height: 44,
            flexShrink: 0,
            boxSizing: "border-box",
          }}
        >
          <span style={{ fontSize: 14, fontWeight: 600, color: T.text, letterSpacing: "-0.01em" }}>{label}</span>
          {meta != null && (
            <div style={{ fontSize: 12, color: T.muted, whiteSpace: "nowrap" }}>{meta}</div>
          )}
        </header>
      )}

      {/* Flex column so a panel body can fill the frame with `flex: 1`. A
          percentage height cannot: the section is sized by `minHeight`, so its
          height is indefinite and `height: 100%` collapses to `auto`. */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column", padding: label === null ? "16px 16px 16px" : "4px 16px 16px", color: T.text, minWidth: 0 }}>{children}</div>
    </section>
  );
}
