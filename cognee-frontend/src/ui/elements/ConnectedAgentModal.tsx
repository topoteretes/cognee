"use client";

import { useCallback, useState, type ReactElement } from "react";
import Link from "next/link";
import type { SetupConnectorCfg, StepDef } from "@/modules/integrations/types";
import type { CogneeInstance } from "@/modules/instances/types";
import {
  AGENT_INTEGRATION_BY_KEY,
  earliestConnectionAt,
  lastSeenByAgent,
  latestConnectionForAgent,
  newestFirst,
} from "@/modules/agents/getAgentConnections";
import type { AgentConnection, AgentIntegrationKey } from "@/modules/agents/getAgentConnections";
import type { RecentQa, RecentTrace } from "@/modules/agents/getAgentConnectionDetail";
import { isSessionRunning } from "@/modules/sessions/getSessions";
import type { SessionRow } from "@/modules/sessions/getSessions";
import { agentStatusText } from "@/app/(app)/dashboard/agentNodes";
import {
  datasetRolePhrase,
  matchingSessions,
  memoryLineText,
  sessionDurationPhrase,
  sessionStartedLabel,
  sessionStatusLabel,
  writesToDatasetsDetailed,
} from "@/modules/agents/agentConnectionPresentation";
import { timeAgo } from "@/utils/timeAgo";
import { formatClockTime } from "@/utils/formatDate";
import { OsToggle } from "@/ui/elements/OsToggle";
import { SetupWizardSteps } from "./SetupWizardSteps";
import { useAgentConnectionDetail } from "./useAgentConnectionDetail";

/** The detail view only shows the newest few; the Activity page holds the rest. */
const SESSION_LIST_LIMIT = 5;

const COPY_FEEDBACK_MS = 1500;

/** Own component (not a render helper) because the "Copied" feedback is
 *  per-row state — a shared boolean would flash every row's button at once. */
function CopySessionIdButton({ sessionId }: { sessionId: string }): ReactElement {
  const [copied, setCopied] = useState(false);

  const handleCopy = useCallback(() => {
    if (typeof navigator === "undefined" || !navigator.clipboard) return;
    navigator.clipboard
      .writeText(sessionId)
      .then(() => {
        setCopied(true);
        setTimeout(() => setCopied(false), COPY_FEEDBACK_MS);
      })
      .catch(() => {
        // Denied permission or no secure context — the id is still visible
        // in the row for a manual copy, nothing else to fall back to.
      });
  }, [sessionId]);

  return (
    <button
      onClick={handleCopy}
      aria-label="Copy session id"
      style={{ background: "none", border: "none", color: "rgba(237,236,234,0.4)", cursor: "pointer", padding: 2, fontSize: 11, lineHeight: 1, flexShrink: 0 }}
    >
      {copied ? "Copied" : (
        <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <rect x="9" y="9" width="13" height="13" rx="2" /><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
        </svg>
      )}
    </button>
  );
}

interface ConnectedAgentModalProps {
  agent: AgentIntegrationKey;
  cfg: SetupConnectorCfg;
  cogniInstance: CogneeInstance | null;
  sessions: SessionRow[];
  connections: AgentConnection[];
  /** Whether `connections` is a full page (see useDashboardTelemetry). Gates
   *  "First connected": a truncated page can be missing this agent's actual
   *  first registration, and showing a wrong date is worse than showing none. */
  connectionsTruncated: boolean;
  steps: StepDef[];
  currentStep: number;
  onClose: () => void;
  onStepSelect: (index: number) => void;
  onCopied?: (block: "terminal" | "code", step: number) => void;
  initialTab?: "status" | "setup";
}

function capitalize(text: string): string {
  return text.charAt(0).toUpperCase() + text.slice(1);
}

/** Recent memory shows qas over traces whenever there are any — a trace is
 *  "the agent called a memory tool", a qa is "the agent asked something and
 *  got an answer", and the second is the more useful one to a person
 *  glancing at what their agent has been doing. */
function recentMemoryItems(qas: RecentQa[], traces: RecentTrace[]): { kind: "qa"; item: RecentQa }[] | { kind: "trace"; item: RecentTrace }[] {
  if (qas.length > 0) return qas.map((item) => ({ kind: "qa" as const, item }));
  return traces.map((item) => ({ kind: "trace" as const, item }));
}

const RECENT_MEMORY_LIMIT = 5;

/**
 * What a card that already reads "Connected" opens instead of the install
 * steps: those steps ask the user to do something they have already done,
 * and reopening them the moment a card says "Connected" is the two screens
 * contradicting each other this whole change exists to fix.
 *
 * Status and setup occupy separate tabs so opening setup cannot stack a second
 * long section below the connection summary.
 */
export function ConnectedAgentModal({
  agent,
  cfg,
  cogniInstance,
  sessions,
  connections,
  connectionsTruncated,
  steps,
  currentStep,
  onClose,
  onStepSelect,
  onCopied,
  initialTab = "status",
}: ConnectedAgentModalProps): ReactElement {
  const [tab, setTab] = useState<"status" | "setup">(initialTab);
  const [expandedSteps, setExpandedSteps] = useState<ReadonlySet<number>>(() => new Set([steps.length - 1]));
  const now = new Date();
  const channel = AGENT_INTEGRATION_BY_KEY.get(agent)?.channel ?? agent;

  const agentSessions = matchingSessions(agent, sessions)
    .sort((a, b) => {
      const aRunning = isSessionRunning(a);
      const bRunning = isSessionRunning(b);
      if (aRunning !== bRunning) return aRunning ? -1 : 1;
      return newestFirst(a.last_activity_at, b.last_activity_at);
    });
  const isLive = agentSessions.some(isSessionRunning);
  const lastSeenAt = lastSeenByAgent(connections, sessions)[agent] ?? null;
  const status = agentStatusText(isLive ? "live" : "connected", lastSeenAt, now);
  const visibleSessions = agentSessions.slice(0, SESSION_LIST_LIMIT);
  const datasets = writesToDatasetsDetailed(agent, connections);
  const datasetRoles = new Set(datasets.map((d) => d.role));
  const rolesDiffer = datasetRoles.size > 1;
  const memoryConnection = latestConnectionForAgent(agent, connections);
  const memoryLine = memoryConnection ? memoryLineText(memoryConnection.memory_mode, datasets) : null;
  const writesToFallback = datasets.length > 0 ? `Writes to ${datasets.length === 1 ? "" : `${datasets.length} datasets`}` : null;
  const headerLine = memoryLine ?? writesToFallback;
  const firstConnectedAt = connectionsTruncated ? null : earliestConnectionAt(agent, connections);

  const detailAgentId = memoryConnection?.user_id || null;
  const { qas, traces, loading: memoryLoading, failed: memoryFailed } = useAgentConnectionDetail(cogniInstance, detailAgentId);
  const memoryItems = recentMemoryItems(qas, traces).slice(0, RECENT_MEMORY_LIMIT);

  return (
    <div role="dialog" aria-modal="true" style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.35)", backdropFilter: "blur(4px)", WebkitBackdropFilter: "blur(4px)", zIndex: 200, display: "flex", alignItems: "center", justifyContent: "center", padding: 16 }} onClick={onClose}>
      <div className="aci-popup" onClick={(e) => e.stopPropagation()} style={{ background: "rgba(15,15,15,0.92)", backdropFilter: "blur(16px)", borderRadius: 0, width: 520, maxWidth: "100%", maxHeight: "calc(100dvh - 32px)", display: "flex", flexDirection: "column", boxShadow: "0 20px 60px rgba(0,0,0,0.6), 0 0 0 1px rgba(255,255,255,0.1)", overflow: "hidden", animation: "aci-popup 200ms cubic-bezier(0.22,1,0.36,1) forwards" }}>

        <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "16px 20px", borderBottom: "1px solid rgba(255,255,255,0.1)" }}>
          <div style={{ width: 24, height: 24, flexShrink: 0, display: "flex", alignItems: "center", justifyContent: "center" }}>{cfg.icon}</div>
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{ fontSize: 15, fontWeight: 700, color: "#EDECEA", lineHeight: "20px" }}>{cfg.name} connected</div>
            <div style={{ fontSize: 12, color: "#22C55E", marginTop: 1 }}>
              {status.detail ? capitalize(status.detail) : "Connected"}
            </div>
          </div>
          <button onClick={onClose} aria-label="Close" style={{ background: "none", border: "none", color: "rgba(237,236,234,0.65)", cursor: "pointer", padding: 4, borderRadius: 0, lineHeight: 1 }}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
          </button>
        </div>

        <div role="tablist" aria-label={`${cfg.name} connection`} style={{ display: "flex", borderBottom: "1px solid rgba(255,255,255,0.1)" }}>
          {(["status", "setup"] as const).map((item) => (
            <button key={item} type="button" role="tab" id={`agent-${item}-tab`} aria-controls={`agent-${item}-panel`} aria-selected={tab === item} onClick={() => setTab(item)} style={{ flex: 1, padding: "11px 20px", background: "none", border: "none", borderBottom: tab === item ? "2px solid var(--color-cognee-lavender)" : "2px solid transparent", color: tab === item ? "#EDECEA" : "rgba(237,236,234,0.5)", cursor: "pointer", font: "inherit", fontSize: 13 }}>
              {item === "status" ? "Status" : "Setup"}
            </button>
          ))}
        </div>

        {tab === "status" ? <div role="tabpanel" id="agent-status-panel" aria-labelledby="agent-status-tab" style={{ padding: "16px 20px", display: "flex", flexDirection: "column", gap: 16, overflowY: "auto" }}>
          <div>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 8 }}>
              <span style={{ fontSize: 12, fontWeight: 600, color: "rgba(237,236,234,0.55)", textTransform: "uppercase", letterSpacing: "0.04em" }}>
                Sessions ({agentSessions.length})
              </span>
              <Link
                href={`/activity?via=${encodeURIComponent(channel)}`}
                style={{ fontSize: 12, color: "var(--color-cognee-lavender)", textDecoration: "underline" }}
              >
                View all sessions →
              </Link>
            </div>
            {visibleSessions.length === 0 ? (
              <p style={{ fontSize: 13, color: "rgba(237,236,234,0.45)", margin: 0 }}>No sessions yet.</p>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                {visibleSessions.map((session) => {
                  const running = isSessionRunning(session);
                  const statusText = sessionStatusLabel(session);
                  const startedLabel = sessionStartedLabel(session.started_at, now);
                  const duration = sessionDurationPhrase(session);
                  const activityLabel = session.last_activity_at ? `last activity ${timeAgo(session.last_activity_at)}` : null;
                  const errors = session.error_count ?? 0;
                  const restParts = [startedLabel, duration, activityLabel].filter((p): p is string => p !== null);
                  return (
                    <div key={session.session_id} style={{ display: "flex", flexDirection: "column", gap: 2 }}>
                      <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 13, fontWeight: 400, color: "rgba(237,236,234,0.65)" }}>
                        <span style={{ width: 6, height: 6, borderRadius: "50%", background: running ? "#22C55E" : "rgba(237,236,234,0.3)", flexShrink: 0 }} />
                        <span>
                          {statusText && <span style={{ color: running ? "#22C55E" : "rgba(237,236,234,0.65)" }}>{statusText}</span>}
                          {restParts.length > 0 && (statusText ? " · " : "") + restParts.join(" · ")}
                          {errors > 0 && (
                            <span style={{ color: "#F87171" }}>
                              {(statusText || restParts.length > 0) ? " · " : ""}{errors} error{errors === 1 ? "" : "s"}
                            </span>
                          )}
                        </span>
                      </div>
                      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                        <span
                          title={session.session_id}
                          style={{ fontFamily: "monospace", fontSize: 11, color: "rgba(237,236,234,0.4)", wordBreak: "break-all" }}
                        >
                          {session.session_id}
                        </span>
                        <CopySessionIdButton sessionId={session.session_id} />
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          {/* Memory mode and dataset access, one block: the header line
              states the mode and (when every dataset shares one role) that
              role, falling back to the plain "Writes to N datasets" count
              when the connection has no recognized mode or role to name.
              The chip list below still names every dataset, only carrying a
              per-chip role when they don't all agree, so the same fact
              never appears twice in two different words. */}
          {(headerLine || datasets.length > 0) && (
            <div style={{ fontSize: 13, color: "rgba(237,236,234,0.65)" }}>
              {headerLine && <div>{headerLine}</div>}
              {datasets.length > 0 && (
                <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginTop: headerLine ? 6 : 0 }}>
                  {datasets.map((dataset) => {
                    const roleWord = rolesDiffer && dataset.role ? datasetRolePhrase(dataset.role) : null;
                    return (
                      <span
                        key={dataset.name}
                        style={{ padding: "2px 8px", borderRadius: 999, border: "1px solid rgba(255,255,255,0.12)", color: "rgba(237,236,234,0.8)" }}
                      >
                        {dataset.name}{roleWord ? ` · ${roleWord}` : ""}
                      </span>
                    );
                  })}
                </div>
              )}
            </div>
          )}

          {firstConnectedAt && (
            <div style={{ fontSize: 12, color: "rgba(237,236,234,0.5)" }}>First connected {formatClockTime(firstConnectedAt, now)}</div>
          )}
          {/* Newest few remembered exchanges from this agent's most recent
              session — never every session, since that's all the pod
              hydrates (see useAgentConnectionDetail). */}
          {detailAgentId && (memoryLoading || memoryFailed || memoryItems.length > 0) && (
            <div>
              <div style={{ fontSize: 12, fontWeight: 600, color: "rgba(237,236,234,0.55)", textTransform: "uppercase", letterSpacing: "0.04em", marginBottom: 8 }}>
                Recent memory
              </div>
              {memoryLoading ? <p style={{ fontSize: 13, color: "rgba(237,236,234,0.45)", margin: 0 }}>Loading recent memory...</p> : memoryFailed ? <p style={{ fontSize: 13, color: "rgba(237,236,234,0.45)", margin: 0 }}>Could not load recent memory</p> : (
                <>
                  <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                    {memoryItems.map((entry) => {
                      if (entry.kind === "qa") {
                        return (
                          <div
                            key={entry.item.qaId}
                            title={entry.item.question}
                            style={{ fontSize: 13, color: "#EDECEA", display: "-webkit-box", WebkitLineClamp: 2, WebkitBoxOrient: "vertical", overflow: "hidden" }}
                          >
                            {entry.item.question}
                            {entry.item.time && <span style={{ color: "rgba(237,236,234,0.4)" }}> · {timeAgo(entry.item.time)}</span>}
                          </div>
                        );
                      }
                      const isError = entry.item.status === "error" || !!entry.item.errorMessage;
                      return (
                        <div key={entry.item.traceId} style={{ fontSize: 13, color: isError ? "#F87171" : "#EDECEA" }}>
                          {isError ? entry.item.errorMessage || entry.item.label : entry.item.label}
                          {entry.item.time && <span style={{ color: "rgba(237,236,234,0.4)" }}> · {timeAgo(entry.item.time)}</span>}
                        </div>
                      );
                    })}
                  </div>
                  <p style={{ fontSize: 11, color: "rgba(237,236,234,0.35)", margin: "6px 0 0" }}>Most recent session only</p>
                </>
              )}
            </div>
          )}
        </div> : <div role="tabpanel" id="agent-setup-panel" aria-labelledby="agent-setup-tab" style={{ overflowY: "auto" }}>
              {/* Shell-based connectors render different commands per OS; MCP
                  ones ignore it, matching SetupWizardModal's own header toggle. */}
              {cfg.osAware && (
                <div style={{ padding: "12px 20px 0" }}>
                  <OsToggle />
                </div>
              )}
              <SetupWizardSteps
                steps={steps}
                currentStep={currentStep}
                agent={agent}
                cogniInstance={cogniInstance}
                onStepSelect={onStepSelect}
                onCopied={onCopied}
                onClose={onClose}
                connectedAfterOpen={initialTab === "setup"}
                expandedSteps={expandedSteps}
                onStepToggle={(index) => {
                  onStepSelect(index);
                  setExpandedSteps((previous) => {
                  const next = new Set(previous);
                  if (next.has(index)) next.delete(index); else next.add(index);
                  return next;
                  });
                }}
              />
        </div>}
      </div>
    </div>
  );
}
