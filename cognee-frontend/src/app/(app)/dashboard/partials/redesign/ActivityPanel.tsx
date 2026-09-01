"use client";

import React, { useMemo } from "react";
import { channelForSessionId, datasetNameById, estimateCostUsd, runStatus, sessionBrainLabel, type RunStatus, type SessionRow } from "@/modules/sessions/getSessions";
import { actorColor, ownerDisplayName } from "@/ui/elements/AgentActivityTerminal";
import type { PipelineRun } from "@/ui/elements/AgentActivityTerminal";
import type { Agent, Dataset } from "@/ui/layout/FilterContext";
import { formatDate } from "@/utils/formatDate";
import { AsciiFrame } from "./AsciiFrame";
import { FONT, MONO_FONT, T } from "./mono";

interface ActivityPanelProps {
  runs: PipelineRun[];
  sessions: SessionRow[];
  agents: Agent[];
  /** Resolves a row's dataset id to its display name. */
  datasets: Dataset[];
  /** Fired by "View full log" (→ the full /activity page). */
  onViewFullLog?: () => void;
}

export type Action = "recall" | "remember" | "improve" | "forget";
export const ACTION_COLOR: Record<Action, string> = {
  recall: T.green,
  remember: T.blue,
  improve: T.purple,
  forget: T.amber,
};

export type Status = RunStatus | "abandoned" | "unknown";
export const STATUS_META: Record<Status, { label: string; color: string }> = {
  completed: { label: "completed", color: T.green },
  running: { label: "running", color: T.purple },
  failed: { label: "failed", color: T.red },
  abandoned: { label: "abandoned", color: T.faint },
  unknown: { label: "unknown", color: T.muted },
};

export function sessionStatus(raw: string): Status {
  const s = raw.toLowerCase();
  if (s === "completed" || s === "running" || s === "failed" || s === "abandoned") return s;
  return "unknown";
}

export function StatusPill({ status }: { status: Status }): React.ReactElement {
  const meta = STATUS_META[status];
  return (
    <span style={{ ...FONT, display: "inline-block", fontSize: 11, fontWeight: 600, letterSpacing: "0.04em", textTransform: "uppercase", color: meta.color, background: `color-mix(in srgb, ${meta.color} 12%, transparent)`, borderRadius: 100, padding: "2px 8px", whiteSpace: "nowrap" }}>
      {meta.label}
    </span>
  );
}

export function formatCostUsd(cost: number | null): string {
  return cost === null ? "—" : `$${cost.toFixed(4)}`;
}

interface Event {
  key: string;
  ts: number;
  time: string;
  status: Status;
  runId: string;
  user: string;
  /** How the session reached Cognee — UI, API, or a named agent/IDE integration. */
  via: string;
  dataset: string;
  action: Action;
  tokens: number | null;
  cost: number | null;
}

interface TimeParts { ts: number; label: string }

function dateAndTime(dateStr: string | null): TimeParts {
  if (!dateStr) return { ts: 0, label: "—" };
  const ts = new Date(dateStr).getTime();
  return { ts, label: formatDate(dateStr) };
}

export function actionForPipeline(name: string): Action {
  if (name.includes("search") || name.includes("recall")) return "recall";
  if (name.includes("cognify") || name.includes("improve")) return "improve";
  return "remember";
}

/**
 * ACTIVITY panel — a compact live log (time · action · status · user · via ·
 * dataset · run id · tokens · cost) drawn from pipeline runs and search
 * sessions, matching WO-0. This is the glance
 * view; "full log →" hands off to the full AgentActivityTerminal / sessions
 * page which does the richer per-event correlation and evidence expansion.
 */
export function ActivityPanel({ runs, sessions, agents, datasets, onViewFullLog }: ActivityPanelProps): React.ReactElement {
  const events = useMemo(() => {
    const agentName = new Map(agents.map((a) => [a.id, ownerDisplayName(a.email)]));
    const nameById = datasetNameById(datasets);
    const list: Event[] = [];

    for (const r of runs) {
      // Operation rows (kind: "operation", pipeline_name: null) are recall/
      // search/remember/etc. — already represented below via `sessions` at
      // the session-lifecycle granularity. Counting them again here would
      // double the tokens/cost totals, so this feed only takes named
      // pipelines (cognify, add, memify, indexing), mirroring the same
      // kind-based split AgentActivityTerminal uses for its own log.
      if (r.kind !== "pipeline") continue;
      const { ts, label } = dateAndTime(r.created_at);
      // SDK-399 tokens are independently nullable: null means "not measured".
      const tokens = r.tokens_in === null && r.tokens_out === null ? null : (r.tokens_in ?? 0) + (r.tokens_out ?? 0);
      list.push({
        key: `run-${r.pipeline_run_id || r.id}`,
        ts,
        time: label,
        status: runStatus(r.status),
        runId: r.pipeline_run_id || r.id,
        user: ownerDisplayName(r.owner_email),
        via: "—", // pipeline runs carry no access-channel signal
        // Unknown, not unscoped: dataset_id is nullable at the source, so a
        // pipeline run without one carries no evidence either way and must not
        // borrow the session rows' UNSCOPED_BRAIN_LABEL.
        dataset: r.dataset_name || (r.dataset_id ? nameById.get(r.dataset_id) : undefined) || "—",
        action: actionForPipeline(r.pipeline_name || ""),
        tokens,
        // Estimated the same way as everywhere else in the dashboard — see
        // CostPanel — never the authoritative per-operation dollar figure,
        // which the backend doesn't expose.
        cost: tokens === null ? null : estimateCostUsd(r.tokens_in ?? 0, r.tokens_out ?? 0),
      });
    }
    for (const s of sessions) {
      const { ts, label } = dateAndTime(s.started_at || s.last_activity_at);
      list.push({
        key: `ses-${s.session_id}`,
        ts,
        time: label,
        status: sessionStatus(s.effective_status || s.status || ""),
        runId: s.session_id,
        // Mirror the terminal's actor resolution: search-ui- sessions carry a
        // human email as user_id, so fall back to the email's display name
        // rather than collapsing every human search to the literal "agent".
        user: agentName.get(s.user_id) ?? (s.user_id.includes("@") ? ownerDisplayName(s.user_id) : "agent"),
        via: channelForSessionId(s.session_id),
        dataset: sessionBrainLabel(s.dataset_id, nameById),
        action: "recall",
        tokens: (s.tokens_in || 0) + (s.tokens_out || 0),
        cost: estimateCostUsd(s.tokens_in || 0, s.tokens_out || 0),
      });
    }

    // Full-width block below Cost/Performance — room for more rows than the
    // old narrow third column showed.
    return list.sort((a, b) => b.ts - a.ts).slice(0, 8);
  }, [runs, sessions, agents, datasets]);

  return (
    <AsciiFrame label="Activity" minHeight={260}>
      <div style={{ display: "flex", flexDirection: "column", flex: 1, minHeight: 0 }}>
        {/* Column headers */}
        <div style={{ ...FONT, fontSize: 12, color: T.muted, display: "flex", gap: 16, paddingBottom: 6, borderBottom: `1px solid ${T.frame}` }}>
          <span style={{ width: 116, flexShrink: 0 }}>time</span>
          <span style={{ width: 64, flexShrink: 0 }}>action</span>
          <span style={{ width: 92, flexShrink: 0 }}>status</span>
          <span style={{ width: 130, flexShrink: 0 }}>user</span>
          <span style={{ width: 104, flexShrink: 0 }}>via</span>
          <span style={{ flex: 1, minWidth: 0 }}>brain</span>
          <span style={{ width: 110, flexShrink: 0 }}>run id</span>
          <span style={{ width: 60, flexShrink: 0, textAlign: "right" }}>tokens</span>
          <span style={{ width: 68, flexShrink: 0, textAlign: "right" }}>cost (est.)</span>
        </div>

        <div style={{ flex: 1, display: "flex", flexDirection: "column" }}>
          {events.length === 0 && (
            <span style={{ ...FONT, fontSize: 12, color: T.faint, paddingTop: 10 }}>No activity in this range yet</span>
          )}
          {events.map((e) => (
            <div
              key={e.key}
              style={{ ...FONT, fontSize: 12, display: "flex", gap: 16, alignItems: "center", padding: "5px 0", borderBottom: `1px solid ${T.frame}`, background: "transparent", transition: "background 150ms ease" }}
              onMouseEnter={(ev) => { ev.currentTarget.style.background = "rgba(255,255,255,0.06)"; }}
              onMouseLeave={(ev) => { ev.currentTarget.style.background = "transparent"; }}
            >
              <span style={{ width: 116, flexShrink: 0, color: T.muted, fontVariantNumeric: "tabular-nums", whiteSpace: "nowrap" }}>{e.time}</span>
              <span style={{ width: 64, flexShrink: 0, color: ACTION_COLOR[e.action] }}>{e.action}</span>
              <span style={{ width: 92, flexShrink: 0 }}><StatusPill status={e.status} /></span>
              <span style={{ width: 130, flexShrink: 0, display: "flex", alignItems: "center", gap: 6 }}>
                <span style={{ width: 6, height: 6, borderRadius: "50%", background: actorColor(e.user), flexShrink: 0 }} />
                <span style={{ color: T.text, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{e.user}</span>
              </span>
              <span style={{ width: 104, flexShrink: 0, color: T.muted, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{e.via}</span>
              <span style={{ flex: 1, minWidth: 0, color: T.muted, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{e.dataset}</span>
              <span style={{ ...MONO_FONT, width: 110, flexShrink: 0, fontSize: 11, color: T.muted, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={e.runId}>{e.runId}</span>
              <span style={{ ...MONO_FONT, width: 60, flexShrink: 0, fontSize: 11, textAlign: "right", color: e.tokens === null ? T.faint : T.muted, fontVariantNumeric: "tabular-nums" }}>{e.tokens === null ? "—" : e.tokens.toLocaleString()}</span>
              <span style={{ ...MONO_FONT, width: 68, flexShrink: 0, fontSize: 11, textAlign: "right", color: e.cost === null ? T.faint : T.muted, fontVariantNumeric: "tabular-nums" }}>{formatCostUsd(e.cost)}</span>
            </div>
          ))}
        </div>

        <div style={{ marginTop: "auto", flexShrink: 0, padding: "8px 0 0" }}>
          <button
            type="button"
            onClick={onViewFullLog}
            style={{ ...FONT, fontSize: 12, color: T.lavender, background: "none", border: "none", cursor: "pointer", padding: 0 }}
          >
            View full log →
          </button>
        </div>
      </div>
    </AsciiFrame>
  );
}
