"use client";

import { useEffect, useState } from "react";
import { useCogniInstance } from "@/modules/tenant/TenantProvider";
import getDatasets from "@/modules/datasets/getDatasets";

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

type Tab = "agents" | "my-datasets" | "org-shared";

export default function ConnectionsPage() {
  const { cogniInstance, isInitializing } = useCogniInstance();
  const [agents, setAgents] = useState<Agent[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(true);
  const [tab, setTab] = useState<Tab>("agents");
  const [selectedAgentId, setSelectedAgentId] = useState<string | null>(null);
  const [showShareModal, setShowShareModal] = useState(false);
  const [sharing, setSharing] = useState(false);
  const [shareDatasetId, setShareDatasetId] = useState<string | null>(null);
  const [sharedDatasetIds, setSharedDatasetIds] = useState<Record<string, Set<string>>>({});

  useEffect(() => {
    if (!cogniInstance || isInitializing) return;
    Promise.all([
      cogniInstance.fetch("/v1/activity/agents").then((r) => r.ok ? r.json() : []).catch(() => []),
      getDatasets(cogniInstance).then((d: Dataset[]) => Array.isArray(d) ? d : []).catch(() => []),
    ]).then(([a, d]) => {
      const agentData = Array.isArray(a) ? a : [];
      setAgents(agentData);
      setDatasets(d);
      const firstAgent = agentData.find((x: Agent) => x.is_agent);
      if (firstAgent) setSelectedAgentId(firstAgent.id);
    }).finally(() => setLoading(false));
  }, [cogniInstance, isInitializing]);

  const agentUsers = agents.filter((a) => a.is_agent);
  const selectedAgent = agents.find((a) => a.id === selectedAgentId);

  // Agent's datasets (owned by agent OR shared with agent)
  const agentSharedIds = selectedAgent ? (sharedDatasetIds[selectedAgent.id] || new Set<string>()) : new Set<string>();
  const agentDatasets = selectedAgent
    ? datasets.filter((d) => d.ownerId === selectedAgent.id || agentSharedIds.has(d.id))
    : [];

  // My datasets (owned by default user)
  const defaultUser = agents.find((a) => a.is_default);
  const myDatasets = defaultUser
    ? datasets.filter((d) => d.ownerId === defaultUser.id)
    : datasets;

  async function handleShareDataset(datasetId: string, principalId: string) {
    if (!cogniInstance) return;
    setSharing(true);
    try {
      await cogniInstance.fetch(`/v1/permissions/datasets/${principalId}?permission_name=read`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify([datasetId]),
      });
      // Track the shared dataset so it shows immediately in the agent's table
      setSharedDatasetIds((prev) => {
        const next = { ...prev };
        const existing = new Set(prev[principalId] || []);
        existing.add(datasetId);
        next[principalId] = existing;
        return next;
      });
    } catch (err) {
      console.error("Share failed:", err);
    } finally {
      setSharing(false);
      setShowShareModal(false);
      setShareDatasetId(null);
    }
  }

  if (loading || isInitializing) {
    return <div style={{ padding: 32, display: "flex", alignItems: "center", justifyContent: "center", height: "100%" }}><span style={{ fontSize: 14, color: "#71717A" }}>Loading...</span></div>;
  }

  const TABS: { id: Tab; label: string; icon: React.ReactNode }[] = [
    { id: "agents", label: "Agents", icon: <AgentIconSmall /> },
    { id: "my-datasets", label: "My Datasets", icon: <PersonIcon /> },
    { id: "org-shared", label: "Org Shared", icon: <OrgIcon /> },
  ];

  return (
    <div style={{ padding: "32px 48px", display: "flex", flexDirection: "column", gap: 24, fontFamily: '"Inter", system-ui, sans-serif', height: "100%" }}>
      {/* Header */}
      <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        <h1 style={{ fontSize: 24, fontWeight: 600, color: "#18181B", margin: 0 }}>Connections</h1>
        <span style={{ fontSize: 14, color: "#71717A" }}>Manage agents, personal datasets, and shared organization data.</span>
      </div>

      {/* Tabs */}
      <div style={{ display: "flex", gap: 0, borderBottom: "1px solid #E4E4E7" }}>
        {TABS.map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
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
      {showShareModal && selectedAgent && (
        <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.3)", zIndex: 100, display: "flex", alignItems: "center", justifyContent: "center" }} onClick={() => setShowShareModal(false)}>
          <div onClick={(e) => e.stopPropagation()} style={{ background: "#fff", borderRadius: 12, padding: 24, width: 420, display: "flex", flexDirection: "column", gap: 16, boxShadow: "0 16px 48px rgba(0,0,0,0.12)" }}>
            <h2 style={{ fontSize: 18, fontWeight: 600, color: "#18181B", margin: 0 }}>Share dataset with {selectedAgent.agent_type}</h2>
            <p style={{ fontSize: 13, color: "#71717A", margin: 0 }}>Select a dataset to grant read access.</p>
            <div style={{ display: "flex", flexDirection: "column", gap: 4, maxHeight: 300, overflow: "auto" }}>
              {datasets.map((d) => (
                <div
                  key={d.id}
                  onClick={() => setShareDatasetId(d.id)}
                  className="cursor-pointer hover:bg-cognee-hover"
                  style={{ display: "flex", alignItems: "center", gap: 8, padding: "10px 12px", borderRadius: 8, border: shareDatasetId === d.id ? "1px solid #6510F4" : "1px solid #F4F4F5", transition: "border-color 150ms" }}
                >
                  <div style={{ width: 6, height: 6, borderRadius: "50%", background: "#22C55E", flexShrink: 0 }} />
                  <span style={{ fontSize: 13, fontWeight: 500, color: "#18181B" }}>{d.name}</span>
                  {shareDatasetId === d.id && (
                    <svg width="14" height="14" viewBox="0 0 16 16" fill="none" style={{ marginLeft: "auto" }}><path d="M3.5 8.5L6.5 11.5L12.5 4.5" stroke="#6510F4" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" /></svg>
                  )}
                </div>
              ))}
            </div>
            <div style={{ display: "flex", gap: 8, justifyContent: "flex-end" }}>
              <button onClick={() => { setShowShareModal(false); setShareDatasetId(null); }} className="cursor-pointer" style={{ background: "#fff", border: "1px solid #E4E4E7", borderRadius: 8, padding: "8px 16px", fontSize: 13, fontWeight: 500, color: "#3F3F46", fontFamily: "inherit" }}>Cancel</button>
              <button
                onClick={() => shareDatasetId && handleShareDataset(shareDatasetId, selectedAgent.id)}
                disabled={!shareDatasetId || sharing}
                className="cursor-pointer"
                style={{ background: shareDatasetId ? "#6510F4" : "#E4E4E7", border: "none", borderRadius: 8, padding: "8px 16px", fontSize: 13, fontWeight: 500, color: shareDatasetId ? "#fff" : "#A1A1AA", fontFamily: "inherit" }}
              >
                {sharing ? "Sharing..." : "Share"}
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
          <span style={{ fontSize: 13, color: "#71717A", textAlign: "center", maxWidth: 360 }}>Connect an agent framework to start syncing data. Go to API Keys to create a key, or use the onboarding to set up a connection.</span>
        </div>
      )}

      {tab === "agents" && agentUsers.length > 0 && (
        <div style={{ display: "flex", gap: 0, flex: 1, minHeight: 0 }}>
          {/* Agent sidebar */}
          <div style={{ width: 260, display: "flex", flexDirection: "column", gap: 2, flexShrink: 0 }}>
            {agentUsers.map((a) => (
                <div
                  key={a.id}
                  onClick={() => setSelectedAgentId(a.id)}
                  className="cursor-pointer"
                  style={{
                    display: "flex", alignItems: "center", gap: 10, padding: "12px 16px",
                    borderRadius: 8, borderLeft: selectedAgentId === a.id ? "3px solid #6510F4" : "3px solid transparent",
                    background: selectedAgentId === a.id ? "#F0EDFF" : "transparent",
                    transition: "all 150ms",
                  }}
                >
                  <span style={{ color: selectedAgentId === a.id ? "#6510F4" : "#71717A" }}><AgentIconSmall /></span>
                  <span style={{ fontSize: 14, fontWeight: selectedAgentId === a.id ? 500 : 400, color: selectedAgentId === a.id ? "#6510F4" : "#18181B", flex: 1 }}>{a.agent_type}</span>
                  <StatusBadge status={a.status} />
                </div>
              ))}
          </div>

          {/* Agent detail */}
          <div style={{ flex: 1, paddingLeft: 32, display: "flex", flexDirection: "column", gap: 20 }}>
            {selectedAgent ? (
              <>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
                  <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                      <span style={{ fontSize: 22, fontWeight: 600, color: "#18181B" }}>{selectedAgent.agent_type}</span>
                      <StatusBadge status={selectedAgent.status} />
                    </div>
                    <span style={{ fontSize: 14, color: "#71717A" }}>
                      {agentDatasets.length} dataset{agentDatasets.length !== 1 ? "s" : ""}
                      {selectedAgent.created_at && ` · Last active ${timeAgo(selectedAgent.created_at)}`}
                    </span>
                  </div>
                  <button
                    onClick={() => setShowShareModal(true)}
                    className="cursor-pointer hover:bg-cognee-purple-hover"
                    style={{ background: "#6510F4", color: "#fff", border: "none", borderRadius: 6, padding: "8px 16px", fontSize: 13, fontWeight: 500 }}
                  >
                    Share dataset
                  </button>
                </div>

                {/* Agent datasets table */}
                <div style={{ background: "#fff", border: "1px solid #E4E4E7", borderRadius: 12, overflow: "hidden" }}>
                  <div style={{ display: "flex", padding: "12px 20px", borderBottom: "1px solid #E4E4E7" }}>
                    <span style={{ flex: 1, fontSize: 12, fontWeight: 500, letterSpacing: "0.5px", color: "#71717A", textTransform: "uppercase" }}>Dataset</span>
                    <span style={{ width: 100, fontSize: 12, fontWeight: 500, letterSpacing: "0.5px", color: "#71717A", textTransform: "uppercase", flexShrink: 0 }}>Owner</span>
                    <span style={{ width: 100, fontSize: 12, fontWeight: 500, letterSpacing: "0.5px", color: "#71717A", textTransform: "uppercase", flexShrink: 0 }}>Access</span>
                    <span style={{ width: 100, fontSize: 12, fontWeight: 500, letterSpacing: "0.5px", color: "#71717A", textTransform: "uppercase", flexShrink: 0 }}>Status</span>
                  </div>
                  {agentDatasets.length === 0 ? (
                    <div style={{ padding: "24px 20px", textAlign: "center" }}>
                      <span style={{ fontSize: 13, color: "#A1A1AA" }}>No datasets shared yet. Click &quot;Share dataset&quot; to grant access.</span>
                    </div>
                  ) : (
                    agentDatasets.map((d, i) => {
                      const isOwned = selectedAgent && d.ownerId === selectedAgent.id;
                      return (
                      <div key={d.id} style={{ display: "flex", alignItems: "center", padding: "14px 20px", borderBottom: i < agentDatasets.length - 1 ? "1px solid #F4F4F5" : "none" }}>
                        <div style={{ flex: 1, display: "flex", alignItems: "center", gap: 8 }}>
                          <div style={{ width: 6, height: 6, borderRadius: "50%", background: "#22C55E", flexShrink: 0 }} />
                          <span style={{ fontSize: 14, fontWeight: 500, color: "#18181B" }}>{d.name}</span>
                        </div>
                        <span style={{ width: 100, fontSize: 13, color: "#3F3F46", flexShrink: 0 }}>{isOwned ? "Agent" : "Shared"}</span>
                        <span style={{ width: 100, fontSize: 13, color: "#3F3F46", flexShrink: 0 }}>{isOwned ? "Read & Write" : "Read"}</span>
                        <span style={{ width: 100, fontSize: 13, color: "#22C55E", fontWeight: 500, flexShrink: 0 }}>Indexed</span>
                      </div>
                      );
                    })
                  )}
                </div>
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
                <span style={{ width: 100, fontSize: 13, color: "#A1A1AA", flexShrink: 0 }}>—</span>
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
            <div style={{ display: "flex", padding: "12px 20px", borderBottom: "1px solid #E4E4E7" }}>
              <span style={{ flex: 1, fontSize: 12, fontWeight: 500, letterSpacing: "0.5px", color: "#71717A", textTransform: "uppercase" }}>Name</span>
              <span style={{ width: 200, fontSize: 12, fontWeight: 500, letterSpacing: "0.5px", color: "#71717A", textTransform: "uppercase", flexShrink: 0 }}>Used by</span>
              <span style={{ width: 100, fontSize: 12, fontWeight: 500, letterSpacing: "0.5px", color: "#71717A", textTransform: "uppercase", flexShrink: 0 }}>Status</span>
            </div>
            {datasets.map((d, i) => {
              // Find the owner agent
              const owner = agents.find((a) => a.id === d.ownerId);
              const ownerLabel = owner?.is_agent ? owner.agent_type : owner?.is_default ? "You" : owner?.email?.split("@")[0] || "—";
              return (
                <div key={d.id} style={{ display: "flex", alignItems: "center", padding: "14px 20px", borderBottom: i < datasets.length - 1 ? "1px solid #F4F4F5" : "none" }}>
                  <div style={{ flex: 1, display: "flex", alignItems: "center", gap: 8 }}>
                    <div style={{ width: 6, height: 6, borderRadius: "50%", background: "#22C55E", flexShrink: 0 }} />
                    <span style={{ fontSize: 14, fontWeight: 500, color: "#18181B" }}>{d.name}</span>
                  </div>
                  <div style={{ width: 200, display: "flex", alignItems: "center", gap: 6, flexShrink: 0 }}>
                    {owner?.is_agent && (
                      <span style={{ background: "#F0EDFF", borderRadius: 4, padding: "2px 8px", fontSize: 12, fontWeight: 500, color: "#6510F4" }}>{ownerLabel}</span>
                    )}
                    {!owner?.is_agent && (
                      <span style={{ fontSize: 13, color: "#3F3F46" }}>{ownerLabel}</span>
                    )}
                  </div>
                  <span style={{ width: 100, fontSize: 13, color: "#22C55E", fontWeight: 500, flexShrink: 0 }}>Indexed</span>
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
