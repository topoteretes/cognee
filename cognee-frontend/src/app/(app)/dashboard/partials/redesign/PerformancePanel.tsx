"use client";

import React from "react";
import { AsciiFrame } from "./AsciiFrame";
import { FONT, T } from "./mono";

/** Open-source stub — recall/coverage scoring reads the tenant's real brains
 *  and is a Cognee Cloud feature. Renders a text-only notice instead of
 *  syncing the real panel. */

export interface TopicScore { name: string; pct: number | null }

interface PerformancePanelProps {
  recallPct: number | null;
  topics: TopicScore[];
  onUpload?: () => void;
  onViewAnalysis?: () => void;
}

export function PerformancePanel(_props: PerformancePanelProps): React.ReactElement {
  return (
    <AsciiFrame label="Memory Coverage" minHeight={260}>
      <div style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 10, textAlign: "center", padding: 20 }}>
        <span style={{ ...FONT, fontSize: 14, fontWeight: 500, color: T.text }}>
          Memory Coverage is a Cognee Cloud feature
        </span>
        <span style={{ ...FONT, fontSize: 13, color: T.muted, maxWidth: 320 }}>
          Build your own dashboard from the API, or use the hosted one in Cognee Cloud.
        </span>
        <a
          href="https://www.cognee.ai"
          target="_blank"
          rel="noopener noreferrer"
          style={{ ...FONT, marginTop: 4, background: T.lavender, color: "#000000", borderRadius: 8, padding: "8px 20px", fontSize: 13, fontWeight: 600, textDecoration: "none" }}
        >
          Open Cognee Cloud
        </a>
      </div>
    </AsciiFrame>
  );
}
