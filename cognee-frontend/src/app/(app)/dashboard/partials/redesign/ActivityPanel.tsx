"use client";

import React from "react";
import type { SessionRow } from "@/modules/sessions/getSessions";
import type { PipelineRun } from "@/ui/elements/AgentActivityTerminal";
import type { Agent, Dataset } from "@/ui/layout/FilterContext";
import { AsciiFrame } from "./AsciiFrame";
import { FONT, T } from "./mono";

/** Open-source stub — the per-event activity log reveals real workspace
 *  usage and is a Cognee Cloud feature. Renders a text-only notice instead
 *  of syncing the real log. */

interface ActivityPanelProps {
  runs: PipelineRun[];
  sessions: SessionRow[];
  agents: Agent[];
  datasets: Dataset[];
  onViewFullLog?: () => void;
}

export function ActivityPanel(_props: ActivityPanelProps): React.ReactElement {
  return (
    <AsciiFrame label="Activity" minHeight={260}>
      <div style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 10, textAlign: "center", padding: 20 }}>
        <span style={{ ...FONT, fontSize: 14, fontWeight: 500, color: T.text }}>
          Activity is a Cognee Cloud feature
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
