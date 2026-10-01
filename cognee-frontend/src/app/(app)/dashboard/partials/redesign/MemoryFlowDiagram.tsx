"use client";

import React from "react";
import { FONT, T } from "./mono";

/** Open-source stub — the interactive hub-and-spoke diagram (data sources,
 *  live agent/health status, team wiring) reveals private workspace topology
 *  and is a Cognee Cloud feature. This renders a static illustration behind a
 *  blurred overlay with a call to action instead of syncing the real graph. */

export type NodeStatus = "live" | "connected" | "reconnect" | "disconnected";

export interface FlowNodeData {
  name: string;
  logo: string;
  status: NodeStatus;
  avatar?: { initials: string; color: string };
}

export type FlowSource = FlowNodeData;
export type FlowAgent = FlowNodeData;
export type FlowUserNode = FlowNodeData;

interface MemoryFlowDiagramProps {
  sources: FlowSource[];
  agents: FlowAgent[];
  healthy: boolean;
  onInvite?: () => void;
  onCoreClick?: () => void;
  onNodeNavigate?: () => void;
  onTeamsClick?: () => void;
}

const DOT = { width: 8, height: 8, borderRadius: "50%", background: T.frameStrong } as const;

export function MemoryFlowDiagram(_props: MemoryFlowDiagramProps): React.ReactElement {
  return (
    <div style={{ position: "relative" }}>
      <div
        aria-hidden
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 20,
          padding: "36px 20px",
          filter: "blur(6px)",
          userSelect: "none",
        }}
      >
        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          {[0, 1, 2].map((i) => (
            <div key={i} style={{ ...DOT, opacity: 0.6 - i * 0.15 }} />
          ))}
        </div>
        <div style={{ width: 64, height: 64, borderRadius: "50%", background: T.chrome, border: `1px solid ${T.frame}` }} />
        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          {[0, 1, 2].map((i) => (
            <div key={i} style={{ ...DOT, opacity: 0.6 - i * 0.15 }} />
          ))}
        </div>
      </div>

      <div
        style={{
          position: "absolute",
          inset: 0,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          gap: 10,
          textAlign: "center",
          padding: 20,
        }}
      >
        <span style={{ ...FONT, fontSize: 14, fontWeight: 500, color: T.text }}>
          Live memory graph is a Cognee Cloud feature
        </span>
        <span style={{ ...FONT, fontSize: 13, color: T.muted, maxWidth: 360 }}>
          Build your own dashboard from the API, or use the hosted one in Cognee Cloud.
        </span>
        <a
          href="https://www.cognee.ai"
          target="_blank"
          rel="noopener noreferrer"
          style={{
            ...FONT,
            marginTop: 4,
            background: T.lavender,
            color: "#000000",
            borderRadius: 8,
            padding: "8px 20px",
            fontSize: 13,
            fontWeight: 600,
            textDecoration: "none",
          }}
        >
          Open Cognee Cloud
        </a>
      </div>
    </div>
  );
}
