"use client";

import React from "react";
import type { PipelineRun, SessionRow } from "@/modules/sessions/getSessions";
import type { Agent, Dataset } from "@/ui/layout/FilterContext";
import { AsciiFrame } from "./AsciiFrame";
import { FONT, T } from "./mono";

/** Open-source stub — the per-event activity log reveals real workspace
 *  usage and is a Cognee Cloud feature. Renders a text-only notice instead
 *  of syncing the real log. */

/**
 * Mirrors the real panel's props exactly, requiredness included.
 *
 * The stub ignores every value, so it is tempting to mark them all optional —
 * but that makes the public type-check gate strictly weaker than the private
 * one: a call site that forgot a required prop would fail locally and pass the
 * sync, which is the wrong way round for a gate whose job is to catch drift.
 * Keep this in step with the real ActivityPanelProps.
 */
interface ActivityPanelProps {
  runs: PipelineRun[];
  sessions: SessionRow[];
  agents: Agent[];
  datasets: Dataset[];
  loading: boolean;
  error: boolean;
  partial: boolean;
  onViewFullLog: () => void;
  onOpenSession: (sessionId: string) => void;
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
