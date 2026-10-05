"use client";

import type { ReactElement } from "react";
import type { SetupConnectorCfg } from "@/modules/integrations/types";

interface SetupConnectorCardProps {
  card: SetupConnectorCfg;
  isActive: boolean;
  hasSignal: boolean;
  isConnected: boolean;
  onOpen: () => void;
}

export default function SetupConnectorCard({ card, isActive, hasSignal, isConnected, onOpen }: SetupConnectorCardProps): ReactElement {
  return (
    <button
      className="aci-card"
      onClick={onOpen}
      style={{
        height: "100%",
        background: isActive ? "rgba(188,155,255,0.20)" : "rgba(255,255,255,0.06)",
        backdropFilter: "blur(12px)",
        border: `1px solid ${isActive ? "rgba(188,155,255,0.35)" : "rgba(255,255,255,0.1)"}`,
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
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12, minWidth: 0 }}>
          <div style={{ width: 40, height: 40, borderRadius: 10, background: "rgba(255,255,255,0.08)", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
            {card.icon}
          </div>
          <span style={{ fontSize: 16, fontWeight: 500, color: "#EDECEA", fontFamily: '"TWKLausanne", sans-serif', overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
            {card.name}
          </span>
        </div>
        {hasSignal && (
          isConnected ? (
            <span style={{ flexShrink: 0, display: "flex", alignItems: "center", gap: 5, background: "rgba(34,197,94,0.14)", color: "#22C55E", fontSize: 11, fontWeight: 600, padding: "2px 8px", borderRadius: 999 }}>
              <span style={{ width: 6, height: 6, borderRadius: "50%", background: "#22C55E" }} />Connected
            </span>
          ) : (
            <span style={{ flexShrink: 0, background: "rgba(255,255,255,0.06)", color: "rgba(237,236,234,0.35)", fontSize: 11, fontWeight: 500, padding: "2px 8px", borderRadius: 999 }}>Not connected yet</span>
          )
        )}
      </div>

      <p style={{ fontSize: 13, color: "rgba(237,236,234,0.55)", margin: 0 }}>{card.description}</p>

      <div style={{ marginTop: "auto" }}>
        <span style={{ display: "inline-flex", alignItems: "center", background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.18)", borderRadius: 8, padding: "6px 14px", fontSize: 13, fontWeight: 600, color: "#EDECEA" }}>
          {card.cta}
        </span>
      </div>
    </button>
  );
}
