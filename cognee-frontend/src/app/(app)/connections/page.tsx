"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useCogniInstance } from "@/modules/tenant/TenantProvider";
import { useFilter } from "@/ui/layout/FilterContext";
import { listSessions, type SessionRow } from "@/modules/sessions/getSessions";
import { trackEvent, TrackPageView } from "@/modules/analytics";

interface Agent {
  id: string;
  email: string;
  agent_type: string;
  agent_short_id: string;
  is_agent: boolean;
  is_default: boolean;
  status: string;
  api_key_count: number;
  created_at: string | null;
  last_active: string | null;
}

interface CreatedAgent {
  agentId: string;
  agentEmail: string;
  agentApiKey: string;
}

interface Dataset { id: string; name: string; ownerId?: string }

function timeAgo(dateStr: string): string {
  const diff = Date.now() - new Date(dateStr).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
}

function StatusBadge({ status }: { status: string }) {
  const color = status === "LIVE" ? "#22C55E" : status === "STAGING" ? "#F59E0B" : "#A1A1AA";
  const label = status === "LIVE" ? "LIVE" : status === "STAGING" ? "STAGING" : "INACTIVE";
  const textColor = status === "LIVE" ? "#16A34A" : status === "STAGING" ? "#D97706" : "#71717A";
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 4, fontSize: 12, fontWeight: 600, color: textColor }}>
      <span style={{ width: 6, height: 6, borderRadius: "50%", background: color }} />
      {label}
    </span>
  );
}

function AgentIconSmall() {
  return <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0 }}><rect x="4" y="4" width="16" height="16" rx="3" /><circle cx="12" cy="12" r="3" /></svg>;
}

function PersonIcon() {
  return <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0 }}><circle cx="12" cy="8" r="4" /><path d="M5.5 21a6.5 6.5 0 0113 0" /></svg>;
}

function OrgIcon() {
  return <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0 }}><rect x="3" y="7" width="18" height="13" rx="2" /><path d="M3 12h18" /></svg>;
}

function HoverTooltip({ label, items }: { label: string; items: string[] }) {
  if (items.length === 0) return <span style={{ fontSize: 13, color: "#A1A1AA" }}>—</span>;
  return (
    <span style={{ position: "relative", fontSize: 13, color: "#6510F4", cursor: "default" }} className="hover-tooltip-trigger">
      {label}
      <span className="hover-tooltip-popup" style={{
        display: "none", position: "absolute", bottom: "calc(100% + 6px)", left: "50%", transform: "translateX(-50%)",
        background: "#18181B", color: "#fff", borderRadius: 6, padding: "6px 10px", fontSize: 12, whiteSpace: "nowrap",
        zIndex: 50, pointerEvents: "none", lineHeight: "18px",
      }}>
        {items.map((name, i) => <span key={i} style={{ display: "block" }}>{name}</span>)}
      </span>
      <style>{`.hover-tooltip-trigger:hover .hover-tooltip-popup { display: block !important; }`}</style>
    </span>
  );
}

type Tab = "agents" | "my-datasets" | "org-shared";

export default function ConnectionsPage() {
  const { cogniInstance, isInitializing } = useCogniInstance();
  const { datasets: contextDatasets } = useFilter();
  const [agents, setAgents] = useState<Agent[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>(contextDatasets as Dataset[]);
  const [loading, setLoading] = useState(true);
  const [tab, setTab] = useState<Tab>("agents");
  const [selectedAgentId, setSelectedAgentId] = useState<string | null>(null);
  const [showShareModal, setShowShareModal] = useState(false);
  const [sharing, setSharing] = useState(false);
  // agentId → datasetId → Set<"read"|"write">
  const [agentPermissions, setAgentPermissions] = useState<Record<string, Record<string, Set<string>>>>({});;
  const [showCreateAgentModal, setShowCreateAgentModal] = useState(false);
  const [createAgentName, setCreateAgentName] = useState("");
  const [creatingAgent, setCreatingAgent] = useState(false);
  const [createdAgent, setCreatedAgent] = useState<CreatedAgent | null>(null);
  const [createAgentError, setCreateAgentError] = useState<string | null>(null);
  const [copiedKey, setCopiedKey] = useState(false);
  const [showDeleteAgentModal, setShowDeleteAgentModal] = useState(false);
  const [deletingAgent, setDeletingAgent] = useState(false);
  const [allSessions, setAllSessions] = useState<SessionRow[]>([]);
  useEffect(() => {
    if (!cogniInstance || isInitializing) return;
    setDatasets(contextDatasets as Dataset[]);
    Promise.all([
      cogniInstance.fetch("/v1/activity/agents").then((r) => r.ok ? r.json() : []).catch(() => []),
      listSessions(cogniInstance, { range: "30d", limit: 100 }),
    ]).then(async ([a, sessionsPage]) => {
      setAllSessions(sessionsPage?.sessions ?? []);
      const agentData = Array.isArray(a) ? a : [];
      setAgents(agentData);
      const firstAgent = agentData.find((x: Agent) => x.is_agent);
      if (firstAgent) setSelectedAgentId(firstAgent.id);

      // OSS exposes grant/revoke through /v1/permissions/datasets but does
      // not currently expose a per-agent dataset list endpoint.
      const permMap: Record<string, Record<string, Set<string>>> = {};
      setAgentPermissions(permMap);
    }).finally(() => setLoading(false));
  }, [cogniInstance, isInitializing, contextDatasets]);

  // Keep local datasets in sync with FilterContext
  useEffect(() => {
    if (contextDatasets.length > 0) {
      setDatasets(contextDatasets as Dataset[]);
    }
  }, [contextDatasets]);

  const agentUsers = agents.filter((a) => a.is_agent && !a.is_default);
  const selectedAgent = agents.find((a) => a.id === selectedAgentId);

  // Agent's datasets (owned by agent OR shared with agent)
  const agentPerms = selectedAgent ? (agentPermissions[selectedAgent.id] || {}) : {};
  const agentDatasets = selectedAgent
    ? datasets.filter((d) => d.ownerId === selectedAgent.id || d.id in agentPerms)
    : [];

  // My datasets (owned by default user)
  const defaultUser = agents.find((a) => a.is_default);
  const myDatasets = defaultUser
    ? datasets.filter((d) => d.ownerId === defaultUser.id)
    : datasets;

  // Reverse mapping: dataset ID → { agents: string[], users: string[] }
  const datasetSharedWith = (() => {
    const map: Record<string, { agents: string[]; users: string[] }> = {};
    const ensure = (id: string) => { if (!map[id]) map[id] = { agents: [], users: [] }; };
    for (const agent of agentUsers) {
      const perms = agentPermissions[agent.id] || {};
      for (const dsId of Object.keys(perms)) {
        ensure(dsId);
        if (!map[dsId].agents.includes(agent.agent_type)) map[dsId].agents.push(agent.agent_type);
      }
      // Also include datasets owned by this agent
      for (const ds of datasets) {
        if (ds.ownerId === agent.id) {
          ensure(ds.id);
          if (!map[ds.id].agents.includes(agent.agent_type)) map[ds.id].agents.push(agent.agent_type);
        }
      }
    }
    // Track user (non-agent) owners
    for (const ds of datasets) {
      const owner = agents.find((a) => a.id === ds.ownerId);
      if (owner && !owner.is_agent) {
        ensure(ds.id);
        const label = owner.is_default ? "You" : owner.email?.split("@")[0] || "User";
        if (!map[ds.id].users.includes(label)) map[ds.id].users.push(label);
      }
    }
    return map;
  })();

  async function handleShareDataset(agentId: string, datasetId: string) {
    if (!cogniInstance) return;
    setSharing(true);
    try {
      await cogniInstance.fetch(`/v1/permissions/datasets/${agentId}?permission_name=read`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify([datasetId]),
      });
      trackEvent({ pageName: "Connections", eventName: "dataset_shared_with_agent", additionalProperties: { dataset_id: datasetId, agent_id: agentId } });
      setAgentPermissions((prev) => {
        const next = { ...prev };
        const agentPerms = { ...(prev[agentId] || {}) };
        agentPerms[datasetId] = new Set(["read"]);
        next[agentId] = agentPerms;
        return next;
      });
    } catch (err) {
      console.error("Share failed:", err);
    } finally {
      setSharing(false);
    }
  }

  async function handleCreateAgent() {
    if (!cogniInstance || !createAgentName.trim()) return;
    setCreatingAgent(true);
    setCreateAgentError(null);
    try {
      const res = await cogniInstance.fetch(`/v1/agents/create?name=${encodeURIComponent(createAgentName.trim())}`, { method: "POST" });
      if (!res.ok) {
        const body = await res.text();
        throw new Error(body || `Failed to create agent (${res.status})`);
      }
      const data: CreatedAgent = await res.json();
      setCreatedAgent(data);
      trackEvent({ pageName: "Connections", eventName: "agent_created", additionalProperties: { agent_id: data.agentId, agent_email: data.agentEmail } });
      // Refresh agents list
      cogniInstance.fetch("/v1/activity/agents").then((r) => r.ok ? r.json() : []).then((a) => {
        const agentData = Array.isArray(a) ? a : [];
        setAgents(agentData);
      }).catch(() => {});
    } catch (err) {
      setCreateAgentError(err instanceof Error ? err.message : "Failed to create agent");
    } finally {
      setCreatingAgent(false);
    }
  }

  function closeCreateAgentModal() {
    setShowCreateAgentModal(false);
    setCreateAgentName("");
    setCreatedAgent(null);
    setCreateAgentError(null);
    setCopiedKey(false);
  }

  if (loading || isInitializing) {
    return <><TrackPageView page="Connections" /><div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", background: "#FFFFFF" }}><video src="/videos/mascot-waiting.mp4" autoPlay loop muted playsInline style={{ width: 200, height: "auto" }} /></div></>;
  }

  const TABS: { id: Tab; label: string; icon: React.ReactNode }[] = [
    { id: "agents", label: "Agents", icon: <AgentIconSmall /> },
    { id: "my-datasets", label: "My Datasets", icon: <PersonIcon /> },
    { id: "org-shared", label: "Org Shared", icon: <OrgIcon /> },
  ];

  return (
    <div style={{ padding: 32, display: "flex", flexDirection: "column", gap: 24, fontFamily: '"Inter", system-ui, sans-serif', height: "100%" }}>
      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          <h1 style={{ fontSize: 20, fontWeight: 300, color: "#18181B", margin: 0, fontFamily: '"TWK Lausanne", system-ui, sans-serif' }}>Connections</h1>
          <span style={{ fontSize: 14, color: "#71717A" }}>Manage agents, personal datasets, and shared organization data.</span>
        </div>
        <Link
          href="#"
          onClick={(event) => {
            event.preventDefault();
            setShowCreateAgentModal(true);
          }}
          className="hover:bg-cognee-purple-hover cursor-pointer"
          style={{ background: "#6510F4", color: "#fff", border: "none", borderRadius: 8, padding: "8px 16px", fontSize: 13, fontWeight: 500, display: "flex", alignItems: "center", gap: 6, fontFamily: "inherit", flexShrink: 0 }}
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="12" y1="5" x2="12" y2="19" /><line x1="5" y1="12" x2="19" y2="12" /></svg>
          Create agent
        </Link>
      </div>

      {/* Tabs */}
      <div style={{ display: "flex", gap: 0, borderBottom: "1px solid #E4E4E7" }}>
        {TABS.map((t) => (
          <button
            key={t.id}
            onClick={() => { trackEvent({ pageName: "Connections", eventName: "connections_tab_switched", additionalProperties: { tab: t.id } }); setTab(t.id); }}
            className="cursor-pointer"
            style={{
              display: "flex", alignItems: "center", gap: 6,
              padding: "10px 20px",
              background: "none", border: "none", borderBottom: tab === t.id ? "2px solid #6510F4" : "2px solid transparent",
              fontSize: 14, fontWeight: tab === t.id ? 500 : 400,
              color: tab === t.id ? "#6510F4" : "#71717A",
              fontFamily: "inherit", marginBottom: -1,
            }}
          >
            <span style={{ color: tab === t.id ? "#6510F4" : "#A1A1AA" }}>{t.icon}</span>
            {t.label}
          </button>
        ))}
      </div>

      {/* Share modal */}
      {showShareModal && selectedAgent && (() => {
        const currentPerms = agentPermissions[selectedAgent.id] || {};
        const sharedDs = datasets.filter((d) => d.id in currentPerms || d.ownerId === selectedAgent.id);
        const unsharedDs = datasets.filter((d) => !(d.id in currentPerms) && d.ownerId !== selectedAgent.id);
        return (
          <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.3)", zIndex: 100, display: "flex", alignItems: "center", justifyContent: "center" }} onClick={() => setShowShareModal(false)}>
            <div onClick={(e) => e.stopPropagation()} style={{ background: "#fff", borderRadius: 12, padding: 24, width: 480, maxHeight: "80vh", display: "flex", flexDirection: "column", gap: 16, boxShadow: "0 16px 48px rgba(0,0,0,0.12)" }}>
              <h2 style={{ fontSize: 18, fontWeight: 600, color: "#18181B", margin: 0 }}>Manage dataset access</h2>
              <p style={{ fontSize: 13, color: "#71717A", margin: 0 }}>Share datasets with <strong>{selectedAgent.agent_type}</strong>.</p>

              {/* Shared datasets */}
              <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                <span style={{ fontSize: 11, fontWeight: 600, color: "#71717A", textTransform: "uppercase", letterSpacing: 0.3 }}>Has access ({sharedDs.length})</span>
                <div style={{ border: "1px solid #E4E4E7", borderRadius: 8, overflow: "hidden" }}>
                  {sharedDs.length === 0 ? (
                    <div style={{ padding: "14px 12px", textAlign: "center", fontSize: 13, color: "#A1A1AA" }}>No datasets shared yet</div>
                  ) : (
                    <div style={{ maxHeight: 180, overflowY: "auto" }}>
                      {sharedDs.map((d) => (
                        <div key={d.id} style={{ display: "flex", alignItems: "center", padding: "10px 12px", borderBottom: "1px solid #F4F4F5", gap: 8 }}>
                          <div style={{ width: 6, height: 6, borderRadius: "50%", background: "#22C55E", flexShrink: 0 }} />
                          <span style={{ flex: 1, fontSize: 13, fontWeight: 500, color: "#18181B", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{d.name}</span>
                          <span style={{ fontSize: 11, color: "#A1A1AA", flexShrink: 0 }}>{d.ownerId === selectedAgent.id ? "Owned" : "Read"}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>

              {/* Unshared datasets */}
              <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                <span style={{ fontSize: 11, fontWeight: 600, color: "#71717A", textTransform: "uppercase", letterSpacing: 0.3 }}>No access ({unsharedDs.length})</span>
                <div style={{ border: "1px solid #E4E4E7", borderRadius: 8, overflow: "hidden" }}>
                  {unsharedDs.length === 0 ? (
                    <div style={{ padding: "14px 12px", textAlign: "center", fontSize: 13, color: "#A1A1AA" }}>All datasets are shared</div>
                  ) : (
                    <div style={{ maxHeight: 180, overflowY: "auto" }}>
                      {unsharedDs.map((d) => (
                        <div key={d.id} style={{ display: "flex", alignItems: "center", padding: "10px 12px", borderBottom: "1px solid #F4F4F5", gap: 8 }}>
                          <div style={{ width: 6, height: 6, borderRadius: "50%", background: "#D4D4D8", flexShrink: 0 }} />
                          <span style={{ flex: 1, fontSize: 13, fontWeight: 500, color: "#18181B", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{d.name}</span>
                          <button
                            onClick={() => handleShareDataset(selectedAgent.id, d.id)}
                            disabled={sharing}
                            className="cursor-pointer hover:bg-cognee-hover"
                            style={{ background: "#fff", border: "1px solid #E4E4E7", borderRadius: 6, padding: "4px 10px", fontSize: 11, fontWeight: 500, color: "#6510F4", fontFamily: "inherit", flexShrink: 0 }}
                          >
                            Share
                          </button>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>

              <div style={{ display: "flex", justifyContent: "flex-end" }}>
                <button onClick={() => setShowShareModal(false)} className="cursor-pointer" style={{ background: "#6510F4", border: "none", borderRadius: 8, padding: "8px 20px", fontSize: 13, fontWeight: 500, color: "#fff", fontFamily: "inherit" }}>Done</button>
              </div>
            </div>
          </div>
        );
      })()}

      {/* Create Agent modal */}
      {showCreateAgentModal && (
        <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.3)", zIndex: 100, display: "flex", alignItems: "center", justifyContent: "center" }} onClick={closeCreateAgentModal}>
          <div onClick={(e) => e.stopPropagation()} style={{ background: "#fff", borderRadius: 12, padding: 24, width: 480, display: "flex", flexDirection: "column", gap: 16, boxShadow: "0 16px 48px rgba(0,0,0,0.12)" }}>
            {!createdAgent ? (
              <>
                <h2 style={{ fontSize: 18, fontWeight: 600, color: "#18181B", margin: 0 }}>Create Agent</h2>
                <p style={{ fontSize: 13, color: "#71717A", margin: 0 }}>
                  Give your agent a name. An API key will be generated automatically.
                </p>
                <input
                  type="text"
                  placeholder="e.g. MyAgent, research-bot, support-agent"
                  value={createAgentName}
                  onChange={(e) => setCreateAgentName(e.target.value)}
                  onKeyDown={(e) => { if (e.key === "Enter" && createAgentName.trim()) handleCreateAgent(); }}
                  autoFocus
                  style={{ border: "1px solid #E4E4E7", borderRadius: 8, padding: "10px 14px", fontSize: 14, fontFamily: "inherit", outline: "none" }}
                />
                {createAgentError && (
                  <span style={{ fontSize: 12, color: "#DC2626" }}>{createAgentError}</span>
                )}
                <div style={{ display: "flex", gap: 8, justifyContent: "flex-end" }}>
                  <button onClick={closeCreateAgentModal} className="cursor-pointer" style={{ background: "#fff", border: "1px solid #E4E4E7", borderRadius: 8, padding: "8px 16px", fontSize: 13, fontWeight: 500, color: "#3F3F46", fontFamily: "inherit" }}>Cancel</button>
                  <button
                    onClick={handleCreateAgent}
                    disabled={!createAgentName.trim() || creatingAgent}
                    className="cursor-pointer"
                    style={{ background: createAgentName.trim() ? "#6510F4" : "#E4E4E7", border: "none", borderRadius: 8, padding: "8px 16px", fontSize: 13, fontWeight: 500, color: createAgentName.trim() ? "#fff" : "#A1A1AA", fontFamily: "inherit" }}
                  >
                    {creatingAgent ? "Creating..." : "Create Agent"}
                  </button>
                </div>
              </>
            ) : (
              <>
                <h2 style={{ fontSize: 18, fontWeight: 600, color: "#18181B", margin: 0 }}>Agent Created</h2>
                <div style={{ display: "flex", gap: 8, background: "#FEF3C7", border: "1px solid #FDE68A", borderRadius: 8, padding: "10px 14px", alignItems: "flex-start" }}>
                  <span style={{ fontSize: 16, flexShrink: 0 }}>&#9888;</span>
                  <span style={{ fontSize: 12, color: "#92400E", lineHeight: "18px" }}>
                    Copy the API key now. It will not be shown again.
                  </span>
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                  <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                    <span style={{ fontSize: 12, fontWeight: 500, color: "#71717A" }}>Agent ID</span>
                    <span style={{ fontSize: 13, fontFamily: '"Fira Code", "SF Mono", monospace', color: "#18181B", background: "#F4F4F5", borderRadius: 6, padding: "8px 12px", wordBreak: "break-all" }}>{createdAgent.agentId}</span>
                  </div>
                  <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                    <span style={{ fontSize: 12, fontWeight: 500, color: "#71717A" }}>Agent Email</span>
                    <span style={{ fontSize: 13, fontFamily: '"Fira Code", "SF Mono", monospace', color: "#18181B", background: "#F4F4F5", borderRadius: 6, padding: "8px 12px" }}>{createdAgent.agentEmail}</span>
                  </div>
                  <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                    <span style={{ fontSize: 12, fontWeight: 500, color: "#71717A" }}>API Key</span>
                    <div style={{ display: "flex", alignItems: "center", gap: 8, background: "#F4F4F5", borderRadius: 6, padding: "8px 12px" }}>
                      <span style={{ fontSize: 13, fontFamily: '"Fira Code", "SF Mono", monospace', color: "#18181B", flex: 1, wordBreak: "break-all" }}>{createdAgent.agentApiKey}</span>
                      <button
                        onClick={() => { navigator.clipboard.writeText(createdAgent.agentApiKey); setCopiedKey(true); setTimeout(() => setCopiedKey(false), 2000); }}
                        className="cursor-pointer"
                        style={{ background: copiedKey ? "#22C55E22" : "#fff", border: "1px solid #E4E4E7", borderRadius: 6, padding: "4px 10px", fontSize: 11, fontWeight: 500, color: copiedKey ? "#22C55E" : "#3F3F46", fontFamily: "inherit", flexShrink: 0 }}
                      >
                        {copiedKey ? "Copied!" : "Copy"}
                      </button>
                    </div>
                  </div>
                </div>
                <div style={{ display: "flex", justifyContent: "flex-end" }}>
                  <button onClick={closeCreateAgentModal} className="cursor-pointer" style={{ background: "#6510F4", border: "none", borderRadius: 8, padding: "8px 20px", fontSize: 13, fontWeight: 500, color: "#fff", fontFamily: "inherit" }}>Done</button>
                </div>
              </>
            )}
          </div>
        </div>
      )}

      {/* Delete Agent modal */}
      {showDeleteAgentModal && selectedAgent && (
        <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.3)", zIndex: 100, display: "flex", alignItems: "center", justifyContent: "center" }} onClick={() => !deletingAgent && setShowDeleteAgentModal(false)}>
          <div onClick={(e) => e.stopPropagation()} style={{ background: "#fff", borderRadius: 12, padding: 24, width: 420, display: "flex", flexDirection: "column", gap: 16, boxShadow: "0 16px 48px rgba(0,0,0,0.12)" }}>
            <h2 style={{ fontSize: 18, fontWeight: 600, color: "#18181B", margin: 0 }}>Delete Agent</h2>
            <p style={{ fontSize: 13, color: "#71717A", margin: 0, lineHeight: "20px" }}>
              Are you sure you want to delete <strong>{selectedAgent.agent_type}</strong>? This will permanently remove the agent and revoke its API key. This action cannot be undone.
            </p>
            <div style={{ display: "flex", gap: 8, justifyContent: "flex-end" }}>
              <button
                onClick={() => setShowDeleteAgentModal(false)}
                disabled={deletingAgent}
                className="cursor-pointer"
                style={{ background: "#fff", border: "1px solid #E4E4E7", borderRadius: 8, padding: "8px 16px", fontSize: 13, fontWeight: 500, color: "#3F3F46", fontFamily: "inherit" }}
              >
                Cancel
              </button>
              <button
                onClick={async () => {
                  if (!cogniInstance) return;
                  setDeletingAgent(true);
                  try {
                    const res = await cogniInstance.fetch(`/v1/agents/${selectedAgent.id}`, { method: "DELETE" });
                    if (!res.ok) throw new Error(`Failed to delete agent (${res.status})`);
                    trackEvent({ pageName: "Connections", eventName: "agent_deleted", additionalProperties: { agent_id: selectedAgent.id, agent_type: selectedAgent.agent_type } });
                    setAgents((prev) => prev.filter((a) => a.id !== selectedAgent.id));
                    setSelectedAgentId(null);
                    setShowDeleteAgentModal(false);
                  } catch (err) {
                    console.error("Delete agent failed:", err);
                  } finally {
                    setDeletingAgent(false);
                  }
                }}
                disabled={deletingAgent}
                className="cursor-pointer"
                style={{ background: "#EF4444", border: "none", borderRadius: 8, padding: "8px 16px", fontSize: 13, fontWeight: 500, color: "#fff", fontFamily: "inherit" }}
              >
                {deletingAgent ? "Deleting..." : "Delete"}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Agents tab */}
      {tab === "agents" && agentUsers.length === 0 && (
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 12, padding: "60px 20px", flex: 1 }}>
          <span style={{ color: "#A1A1AA" }}><AgentIconSmall /></span>
          <span style={{ fontSize: 15, fontWeight: 500, color: "#18181B" }}>No agents connected yet</span>
          <span style={{ fontSize: 13, color: "#71717A", textAlign: "center", maxWidth: 360 }}>Create an agent to get an API key, or follow the integration guides to set up a connection.</span>
          <div style={{ display: "flex", gap: 8, marginTop: 4 }}>
            <Link
              href="#"
              onClick={(event) => {
                event.preventDefault();
                setShowCreateAgentModal(true);
              }}
              className="cursor-pointer"
              style={{ display: "inline-flex", alignItems: "center", gap: 6, background: "#6510F4", color: "#fff", border: "none", borderRadius: 8, padding: "8px 20px", fontSize: 13, fontWeight: 500, fontFamily: "inherit" }}
            >
              Create Agent
            </Link>
            <Link
              href="/connect-agent"
              style={{ display: "inline-flex", alignItems: "center", gap: 6, background: "#fff", color: "#3F3F46", border: "1px solid #E4E4E7", borderRadius: 8, padding: "8px 20px", fontSize: 13, fontWeight: 500, textDecoration: "none" }}
            >
              Integration Guides
            </Link>
          </div>
        </div>
      )}

      {tab === "agents" && agentUsers.length > 0 && (
        <div style={{ display: "flex", gap: 0, flex: 1, minHeight: 0 }}>
          {/* Agent sidebar */}
          <div style={{ width: 220, display: "flex", flexDirection: "column", gap: 2, flexShrink: 0 }}>
            {agentUsers.map((a) => (
                <div
                  key={a.id}
                  onClick={() => { trackEvent({ pageName: "Connections", eventName: "connections_agent_selected", additionalProperties: { agent_id: a.id, agent_type: a.agent_type } }); setSelectedAgentId(a.id); }}
                  className="cursor-pointer"
                  style={{
                    display: "flex", alignItems: "center", gap: 10, padding: "12px 16px",
                    borderRadius: 8, borderLeft: selectedAgentId === a.id ? "3px solid #6510F4" : "3px solid transparent",
                    background: selectedAgentId === a.id ? "#F0EDFF" : "transparent",
                    transition: "all 150ms",
                  }}
                >
                  <span style={{ color: selectedAgentId === a.id ? "#6510F4" : "#71717A", flexShrink: 0 }}><AgentIconSmall /></span>
                  <span style={{ fontSize: 13, fontWeight: selectedAgentId === a.id ? 500 : 400, color: selectedAgentId === a.id ? "#6510F4" : "#18181B", flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{a.agent_type}</span>
                </div>
              ))}
          </div>

          {/* Agent detail */}
          <div style={{ flex: 1, paddingLeft: 24, display: "flex", flexDirection: "column", gap: 16, minWidth: 0 }}>
            {selectedAgent ? (
              <>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12 }}>
                  <div style={{ display: "flex", flexDirection: "column", gap: 2, minWidth: 0 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                      <span style={{ fontSize: 18, fontWeight: 600, color: "#18181B", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{selectedAgent.agent_type}</span>
                      <StatusBadge status={selectedAgent.status} />
                    </div>
                    <span style={{ fontSize: 13, color: "#71717A" }}>
                      {agentDatasets.length} dataset{agentDatasets.length !== 1 ? "s" : ""}
                      {selectedAgent.last_active ? ` · Last active ${timeAgo(selectedAgent.last_active)}` : " · Never connected"}
                    </span>
                  </div>
                  <div style={{ display: "flex", gap: 6, flexShrink: 0 }}>
                    <button
                      onClick={() => setShowDeleteAgentModal(true)}
                      className="cursor-pointer hover:opacity-100"
                      style={{ display: "flex", alignItems: "center", justifyContent: "center", background: "none", border: "1px solid #FECACA", borderRadius: 6, padding: "6px 8px", opacity: 0.7, transition: "opacity 150ms" }}
                      title="Delete agent"
                    >
                      <svg width="14" height="14" viewBox="0 0 16 16" fill="none"><path d="M3 4h10M6 4V3h4v1M5 4v8.5a.5.5 0 00.5.5h5a.5.5 0 00.5-.5V4" stroke="#EF4444" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" /></svg>
                    </button>
                    <button
                      onClick={() => { trackEvent({ pageName: "Connections", eventName: "connections_add_dataset_clicked", additionalProperties: { agent_id: selectedAgent.id, agent_type: selectedAgent.agent_type } }); setShowShareModal(true); }}
                      className="cursor-pointer hover:bg-cognee-purple-hover"
                      style={{ background: "#6510F4", color: "#fff", border: "none", borderRadius: 6, padding: "6px 12px", fontSize: 12, fontWeight: 500 }}
                    >
                      Share dataset
                    </button>
                  </div>
                </div>

                {/* Agent activity metrics */}
                <AgentMetrics sessions={allSessions.filter((s) => s.user_id === selectedAgent.id)} />

                {/* Agent datasets table */}
                <div style={{ background: "#fff", border: "1px solid #E4E4E7", borderRadius: 12, overflow: "hidden" }}>
                  <div style={{ display: "flex", padding: "10px 16px", borderBottom: "1px solid #E4E4E7" }}>
                    <span style={{ flex: 1, fontSize: 11, fontWeight: 500, letterSpacing: "0.5px", color: "#71717A", textTransform: "uppercase", minWidth: 0 }}>Dataset</span>
                    <span style={{ width: 80, fontSize: 11, fontWeight: 500, letterSpacing: "0.5px", color: "#71717A", textTransform: "uppercase", flexShrink: 0 }}>Owner</span>
                    <span style={{ width: 90, fontSize: 11, fontWeight: 500, letterSpacing: "0.5px", color: "#71717A", textTransform: "uppercase", flexShrink: 0 }}>Access</span>
                    <span style={{ width: 70, fontSize: 11, fontWeight: 500, letterSpacing: "0.5px", color: "#71717A", textTransform: "uppercase", flexShrink: 0 }}>Status</span>
                  </div>
                  {agentDatasets.length === 0 ? (
                    <div style={{ padding: "20px 16px", textAlign: "center" }}>
                      <span style={{ fontSize: 13, color: "#A1A1AA" }}>No datasets shared yet. Click &quot;Share dataset&quot; to grant access.</span>
                    </div>
                  ) : (
                    agentDatasets.map((d, i) => {
                      const isOwned = selectedAgent && d.ownerId === selectedAgent.id;
                      return (
                      <div key={d.id} style={{ display: "flex", alignItems: "center", padding: "10px 16px", borderBottom: i < agentDatasets.length - 1 ? "1px solid #F4F4F5" : "none" }}>
                        <div style={{ flex: 1, display: "flex", alignItems: "center", gap: 8, minWidth: 0 }}>
                          <div style={{ width: 6, height: 6, borderRadius: "50%", background: "#22C55E", flexShrink: 0 }} />
                          <span style={{ fontSize: 13, fontWeight: 500, color: "#18181B", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{d.name}</span>
                        </div>
                        <span style={{ width: 80, fontSize: 12, color: "#3F3F46", flexShrink: 0 }}>{isOwned ? "Agent" : "Shared"}</span>
                        <span style={{ width: 90, fontSize: 12, color: "#3F3F46", flexShrink: 0 }}>{isOwned ? "Read & Write" : (() => { const p = agentPerms[d.id]; if (!p) return "Read"; const parts = []; if (p.has("read")) parts.push("Read"); if (p.has("write")) parts.push("Write"); return parts.join(" & ") || "Read"; })()}</span>
                        <span style={{ width: 70, fontSize: 12, color: "#22C55E", fontWeight: 500, flexShrink: 0 }}>Indexed</span>
                      </div>
                      );
                    })
                  )}
                </div>

                {/* Recent sessions */}
                <AgentSessionHistory sessions={allSessions.filter((s) => s.user_id === selectedAgent.id)} />
              </>
            ) : (
              <span style={{ fontSize: 14, color: "#A1A1AA", padding: 20 }}>Select an agent to view details.</span>
            )}
          </div>
        </div>
      )}

      {/* My Datasets tab */}
      {tab === "my-datasets" && (
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            <span style={{ fontSize: 16, fontWeight: 600, color: "#18181B" }}>My datasets</span>
            <span style={{ fontSize: 13, color: "#A1A1AA" }}>Datasets you own</span>
          </div>
          <div style={{ background: "#fff", border: "1px solid #E4E4E7", borderRadius: 12, overflow: "hidden" }}>
            <div style={{ display: "flex", padding: "12px 20px", borderBottom: "1px solid #E4E4E7" }}>
              <span style={{ flex: 1, fontSize: 12, fontWeight: 500, letterSpacing: "0.5px", color: "#71717A", textTransform: "uppercase" }}>Name</span>
              <span style={{ width: 120, fontSize: 12, fontWeight: 500, letterSpacing: "0.5px", color: "#71717A", textTransform: "uppercase", flexShrink: 0 }}>Status</span>
              <span style={{ width: 100, fontSize: 12, fontWeight: 500, letterSpacing: "0.5px", color: "#71717A", textTransform: "uppercase", flexShrink: 0 }}>Shared</span>
            </div>
            {myDatasets.map((d, i) => (
              <div key={d.id} style={{ display: "flex", alignItems: "center", padding: "14px 20px", borderBottom: i < myDatasets.length - 1 ? "1px solid #F4F4F5" : "none" }}>
                <div style={{ flex: 1, display: "flex", alignItems: "center", gap: 8 }}>
                  <div style={{ width: 6, height: 6, borderRadius: "50%", background: "#22C55E", flexShrink: 0 }} />
                  <span style={{ fontSize: 14, fontWeight: 500, color: "#18181B" }}>{d.name}</span>
                </div>
                <span style={{ width: 120, fontSize: 13, color: "#22C55E", fontWeight: 500, flexShrink: 0 }}>Indexed</span>
                <span style={{ width: 100, flexShrink: 0 }}>
                  {(() => {
                    const s = datasetSharedWith[d.id];
                    const count = s?.agents.length || 0;
                    if (count === 0) return <span style={{ fontSize: 13, color: "#A1A1AA" }}>—</span>;
                    return <HoverTooltip label={`${count} agent${count !== 1 ? "s" : ""}`} items={s.agents} />;
                  })()}
                </span>
              </div>
            ))}
            {myDatasets.length === 0 && (
              <div style={{ padding: "24px 20px", textAlign: "center" }}><span style={{ fontSize: 13, color: "#A1A1AA" }}>No personal datasets.</span></div>
            )}
          </div>
        </div>
      )}

      {/* Org Shared tab */}
      {tab === "org-shared" && (
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            <span style={{ fontSize: 16, fontWeight: 600, color: "#18181B" }}>Shared datasets</span>
            <span style={{ fontSize: 13, color: "#A1A1AA" }}>Visible to all members</span>
          </div>
          <div style={{ background: "#fff", border: "1px solid #E4E4E7", borderRadius: 12, overflow: "hidden" }}>
            <div style={{ display: "flex", padding: "10px 16px", borderBottom: "1px solid #E4E4E7" }}>
              <span style={{ flex: 1, fontSize: 11, fontWeight: 500, letterSpacing: "0.5px", color: "#71717A", textTransform: "uppercase", minWidth: 0 }}>Name</span>
              <span style={{ width: 160, fontSize: 11, fontWeight: 500, letterSpacing: "0.5px", color: "#71717A", textTransform: "uppercase", flexShrink: 0 }}>Used by</span>
              <span style={{ width: 70, fontSize: 11, fontWeight: 500, letterSpacing: "0.5px", color: "#71717A", textTransform: "uppercase", flexShrink: 0 }}>Status</span>
            </div>
            {datasets.map((d, i) => {
              const s = datasetSharedWith[d.id];
              const userCount = s?.users.length || 0;
              const agentCount = s?.agents.length || 0;
              const parts: string[] = [];
              if (userCount > 0) parts.push(`${userCount} user${userCount !== 1 ? "s" : ""}`);
              if (agentCount > 0) parts.push(`${agentCount} agent${agentCount !== 1 ? "s" : ""}`);
              const allNames = [...(s?.users || []), ...(s?.agents || [])];
              return (
                <div key={d.id} style={{ display: "flex", alignItems: "center", padding: "10px 16px", borderBottom: i < datasets.length - 1 ? "1px solid #F4F4F5" : "none" }}>
                  <div style={{ flex: 1, display: "flex", alignItems: "center", gap: 8, minWidth: 0 }}>
                    <div style={{ width: 6, height: 6, borderRadius: "50%", background: "#22C55E", flexShrink: 0 }} />
                    <span style={{ fontSize: 13, fontWeight: 500, color: "#18181B", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{d.name}</span>
                  </div>
                  <span style={{ width: 160, flexShrink: 0 }}>
                    {parts.length > 0
                      ? <HoverTooltip label={parts.join(" & ")} items={allNames} />
                      : <span style={{ fontSize: 13, color: "#A1A1AA" }}>—</span>}
                  </span>
                  <span style={{ width: 70, fontSize: 12, color: "#22C55E", fontWeight: 500, flexShrink: 0 }}>Indexed</span>
                </div>
              );
            })}
            {datasets.length === 0 && (
              <div style={{ padding: "24px 20px", textAlign: "center" }}><span style={{ fontSize: 13, color: "#A1A1AA" }}>No shared datasets.</span></div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Agent activity metrics ────────────────────────────────────────────

function AgentMetrics({ sessions }: { sessions: SessionRow[] }) {
  if (sessions.length === 0) return null;
  const total = sessions.length;
  const completed = sessions.filter((s) => s.effective_status === "completed").length;
  const failed = sessions.filter((s) => s.effective_status === "failed").length;
  const successRate = total > 0 ? ((completed / (completed + failed || 1)) * 100) : 100;
  const totalTokens = sessions.reduce((acc, s) => acc + s.tokens_in + s.tokens_out, 0);
  const totalErrors = sessions.reduce((acc, s) => acc + s.error_count, 0);

  function fmt(n: number): string {
    if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
    if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`;
    return String(n);
  }

  const metrics = [
    { label: "Sessions", value: String(total), color: "#18181B" },
    { label: "Success rate", value: `${successRate.toFixed(1)}%`, color: successRate >= 95 ? "#16A34A" : successRate >= 80 ? "#D97706" : "#DC2626" },
    { label: "Tokens", value: fmt(totalTokens), color: "#18181B" },
    { label: "Errors", value: String(totalErrors), color: totalErrors > 0 ? "#DC2626" : "#18181B" },
  ];

  return (
    <div style={{ display: "flex", gap: 12 }}>
      {metrics.map((m) => (
        <div key={m.label} style={{ flex: 1, background: "#fff", border: "1px solid #E4E4E7", borderRadius: 10, padding: "14px 16px", display: "flex", flexDirection: "column", gap: 4 }}>
          <span style={{ fontSize: 11, fontWeight: 600, color: "#A1A1AA", letterSpacing: "0.06em", textTransform: "uppercase" }}>{m.label}</span>
          <span style={{ fontSize: 22, fontWeight: 600, color: m.color, fontVariantNumeric: "tabular-nums", lineHeight: "26px" }}>{m.value}</span>
        </div>
      ))}
    </div>
  );
}

// ── Agent session history ─────────────────────────────────────────────

function AgentSessionHistory({ sessions }: { sessions: SessionRow[] }) {
  const sorted = [...sessions].sort((a, b) => {
    const ta = a.last_activity_at ? new Date(a.last_activity_at).getTime() : 0;
    const tb = b.last_activity_at ? new Date(b.last_activity_at).getTime() : 0;
    return tb - ta;
  });
  const visible = sorted.slice(0, 10);

  if (visible.length === 0) {
    return (
      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        <span style={{ fontSize: 14, fontWeight: 600, color: "#18181B" }}>Recent Sessions</span>
        <div style={{ background: "#fff", border: "1px solid #E4E4E7", borderRadius: 12, padding: "24px 20px", textAlign: "center" }}>
          <span style={{ fontSize: 13, color: "#A1A1AA" }}>No sessions recorded yet.</span>
        </div>
      </div>
    );
  }

  const statusColor: Record<string, string> = { running: "#3B82F6", completed: "#22C55E", failed: "#EF4444", abandoned: "#D97706" };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <span style={{ fontSize: 14, fontWeight: 600, color: "#18181B" }}>Recent Sessions</span>
        {sessions.length > 10 && (
          <Link href="/activity" style={{ fontSize: 12, color: "#6510F4", textDecoration: "none", fontWeight: 500 }}>
            View all in Activity &rarr;
          </Link>
        )}
      </div>
      <div style={{ background: "#fff", border: "1px solid #E4E4E7", borderRadius: 12, overflow: "hidden" }}>
        <div style={{ display: "flex", padding: "10px 20px", borderBottom: "1px solid #E4E4E7" }}>
          <span style={{ width: 100, fontSize: 12, fontWeight: 500, letterSpacing: "0.5px", color: "#71717A", textTransform: "uppercase", flexShrink: 0 }}>Status</span>
          <span style={{ flex: 1, fontSize: 12, fontWeight: 500, letterSpacing: "0.5px", color: "#71717A", textTransform: "uppercase" }}>Session</span>
          <span style={{ width: 100, fontSize: 12, fontWeight: 500, letterSpacing: "0.5px", color: "#71717A", textTransform: "uppercase", flexShrink: 0 }}>Model</span>
          <span style={{ width: 80, fontSize: 12, fontWeight: 500, letterSpacing: "0.5px", color: "#71717A", textTransform: "uppercase", flexShrink: 0, textAlign: "right" }}>Tokens</span>
          <span style={{ width: 80, fontSize: 12, fontWeight: 500, letterSpacing: "0.5px", color: "#71717A", textTransform: "uppercase", flexShrink: 0, textAlign: "right" }}>Time</span>
        </div>
        {visible.map((s, i) => {
          const status = s.effective_status || s.status || "unknown";
          const dot = statusColor[status] || "#A1A1AA";
          const tokens = s.tokens_in + s.tokens_out;
          return (
            <Link
              key={`${s.session_id}-${s.user_id}`}
              href={`/activity?session=${encodeURIComponent(s.session_id)}`}
              className="cursor-pointer hover:bg-cognee-hover"
              style={{ display: "flex", alignItems: "center", padding: "12px 20px", borderBottom: i < visible.length - 1 ? "1px solid #F4F4F5" : "none", textDecoration: "none", color: "inherit", transition: "background 150ms" }}
            >
              <div style={{ width: 100, flexShrink: 0, display: "flex", alignItems: "center", gap: 6 }}>
                <span style={{ width: 6, height: 6, borderRadius: "50%", background: dot, flexShrink: 0 }} />
                <span style={{ fontSize: 12, color: "#52525B", textTransform: "capitalize" }}>{status}</span>
              </div>
              <span style={{ flex: 1, fontSize: 13, color: "#18181B", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", fontFamily: "monospace" }}>
                {s.session_id.length > 24 ? `${s.session_id.slice(0, 24)}...` : s.session_id}
              </span>
              <span style={{ width: 100, fontSize: 13, color: "#3F3F46", flexShrink: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{s.last_model ?? "—"}</span>
              <span style={{ width: 80, fontSize: 13, color: "#18181B", flexShrink: 0, textAlign: "right", fontVariantNumeric: "tabular-nums" }}>
                {tokens >= 1000 ? `${(tokens / 1000).toFixed(1)}k` : String(tokens)}
              </span>
              <span style={{ width: 80, fontSize: 12, color: "#A1A1AA", flexShrink: 0, textAlign: "right" }}>
                {s.last_activity_at ? timeAgo(s.last_activity_at) : "—"}
              </span>
            </Link>
          );
        })}
      </div>
    </div>
  );
}
