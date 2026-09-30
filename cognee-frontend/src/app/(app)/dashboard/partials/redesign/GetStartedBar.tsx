"use client";

import React, { useState } from "react";
import { FONT, T } from "./mono";

interface GetStartedBarProps {
  subtitle: string;
  /** Connector count shown on the right, e.g. 5. */
  connectors: number;
  defaultOpen?: boolean;
  children: React.ReactNode;
}

/**
 * The collapsed "Get started" strip from WO-0 — a single terminal row that
 * expands to reveal the existing agent/brain connection cards. Collapsed by
 * default so returning users land straight on the Overview; the connector count
 * keeps the affordance discoverable.
 */
export function GetStartedBar({ subtitle, connectors, defaultOpen = false, children }: GetStartedBarProps): React.ReactElement {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <div style={{ border: `1px solid ${T.frame}`, borderRadius: 0, background: T.panel, overflow: "hidden" }}>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        style={{
          ...FONT,
          width: "100%",
          display: "flex",
          alignItems: "center",
          gap: 12,
          padding: "12px 16px",
          background: "none",
          border: "none",
          cursor: "pointer",
          textAlign: "left",
        }}
      >
        <span style={{ color: T.lavender, display: "inline-block", width: 12, transition: "transform 140ms", transform: open ? "rotate(90deg)" : "none" }}>➤</span>
        <span style={{ fontSize: 14, fontWeight: 700, color: T.text }}>Get started</span>
        <span style={{ marginLeft: 4, fontSize: 13, color: "rgba(237,236,234,0.48)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{subtitle}</span>
        {/* One compact control group: connector count + expand/collapse toggle. */}
        <span style={{ marginLeft: "auto", display: "inline-flex", alignItems: "center", gap: 9, flexShrink: 0, background: "rgba(255,255,255,0.04)", border: `1px solid ${T.frame}`, borderRadius: 0, padding: "5px 10px" }}>
          <span style={{ fontSize: 12, color: T.muted }}>{connectors} connector{connectors === 1 ? "" : "s"}</span>
          <span style={{ width: 1, height: 12, background: T.frameStrong }} />
          <span style={{ fontSize: 12, fontWeight: 500, color: T.lavender }}>{open ? "Collapse ▴" : "Expand ▾"}</span>
        </span>
      </button>

      {open && <div style={{ padding: "12px 16px 16px", borderTop: `1px solid ${T.frame}` }}>{children}</div>}
    </div>
  );
}
