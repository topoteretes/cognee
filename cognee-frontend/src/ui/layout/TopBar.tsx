"use client";

import { useEffect, useState, useCallback } from "react";
import { usePathname } from "next/navigation";
import getUser from "@/modules/users/getUser";
import getLocalUser from "@/modules/users/getLocalUser";
import CogneeUser from "@/modules/users/CogneeUser";
import isCloudEnvironment from "@/utils/isCloudEnvironment";
import createWorkspace from "@/modules/tenant/createWorkspace";
import HelpMenu from "./HelpMenu";
import ProfileMenu from "./ProfileMenu";
import { useFilter } from "./FilterContext";
import useBoolean from "@/utils/useBoolean";
import useOutsideClick from "@/utils/useOutsideClick";

// ── Icons ──

function Chevron({ color = "#A1A1AA", up = false }: { color?: string; up?: boolean }) {
  return <svg width="12" height="12" viewBox="0 0 12 12" fill="none" style={{ flexShrink: 0 }}><path d={up ? "M3 7.5L6 4.5L9 7.5" : "M3 4.5L6 7.5L9 4.5"} stroke={color} strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" /></svg>;
}
function Check() {
  return <svg width="12" height="12" viewBox="0 0 12 12" fill="none" style={{ marginLeft: "auto", flexShrink: 0 }}><path d="M2.5 6L5 8.5L9.5 3.5" stroke="#6510F4" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" /></svg>;
}
function AgentIcon({ color = "#52525B" }: { color?: string }) {
  return <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0 }}><circle cx="12" cy="8" r="4" /><path d="M5.5 21a6.5 6.5 0 0113 0" /></svg>;
}
function DatasetIcon({ color = "#52525B" }: { color?: string }) {
  return <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0 }}><ellipse cx="12" cy="5" rx="9" ry="3" /><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3" /><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5" /></svg>;
}
function PlusIcon() {
  return <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#A1A1AA" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0 }}><line x1="12" y1="5" x2="12" y2="19" /><line x1="5" y1="12" x2="19" y2="12" /></svg>;
}
function Slash() {
  return <span style={{ color: "#D4D4D8", fontSize: 16, flexShrink: 0 }}>/</span>;
}

function StatusDot({ status }: { status: string }) {
  const color = status === "LIVE" ? "#10B981" : status === "STAGING" ? "#F59E0B" : "#A1A1AA";
  const label = status === "LIVE" ? "Live" : status === "STAGING" ? "Staging" : "Inactive";
  return <span style={{ color, fontSize: 11, fontWeight: 500, whiteSpace: "nowrap" }}>● {label}</span>;
}

// ── Reusable Dropdown ──

function Dropdown({ trigger, children, width = 280 }: { trigger: React.ReactNode; children: React.ReactNode; width?: number }) {
  const { value: isOpen, toggle, setFalse: close } = useBoolean(false);
  const closeCallback = useCallback(() => close(), [close]);
  const ref = useOutsideClick<HTMLDivElement>(closeCallback, isOpen);

  return (
    <div ref={ref} style={{ position: "relative" }}>
      <div onClick={toggle} className="cursor-pointer flex items-center">{trigger}</div>
      {isOpen && (
        <div onClick={close} style={{ position: "absolute", top: 32, left: 0, width, background: "#fff", borderRadius: 10, boxShadow: "0px 8px 30px #0000001F, 0px 0px 0px 1px #0000000F", padding: 6, zIndex: 50 }}>
          {children}
        </div>
      )}
    </div>
  );
}

// ── Page name from path ──

const ROUTE_LABELS: Record<string, string> = {
  "/": "Overview", "/dashboard": "Overview",
  "/datasets": "Datasets", "/search": "Search",
  "/knowledge-graph": "Knowledge Graph",
  "/connect-agent": "Connect Agent",
  "/connections": "Connections", "/api-keys": "API Keys",
  "/activity": "Activity", "/settings": "Settings",
  "/onboarding": "Onboarding",
};

// ── TopBar ──

export default function TopBar() {
  const [user, setUser] = useState<CogneeUser>();
  const cloud = isCloudEnvironment();
  const pathname = usePathname();
  const { workspace, workspaces, setWorkspace, selectedAgent, selectedDataset, setSelectedAgent, setSelectedDataset, agents, datasets } = useFilter();

  // Create workspace modal state
  const [showCreateWsModal, setShowCreateWsModal] = useState(false);
  const [wsName, setWsName] = useState("");
  const [wsCreating, setWsCreating] = useState(false);
  const [wsError, setWsError] = useState<string | null>(null);

  async function handleCreateWorkspace() {
    if (!wsName.trim()) return;
    setWsCreating(true);
    setWsError(null);
    const result = await createWorkspace(wsName.trim());
    setWsCreating(false);
    if (result.success) {
      setWsName("");
      setShowCreateWsModal(false);
      window.location.reload();
    } else {
      setWsError(result.error || "Failed to create workspace");
    }
  }

  useEffect(() => {
    if (cloud) { getUser().then(setUser); }
    else { getLocalUser().then((u) => { if (u) setUser(u); }); }
  }, [cloud]);

  const agentList = agents.filter((a) => a.is_agent);

  // Derive page label
  const basePath = "/" + (pathname.split("/").filter(Boolean)[0] || "");
  const pageName = ROUTE_LABELS[basePath] || basePath.slice(1).charAt(0).toUpperCase() + basePath.slice(2).replace(/-/g, " ");

  // Check if we're on a dataset detail page
  const datasetDetailMatch = pathname.match(/^\/datasets\/(.+)$/);
  const isDatasetDetail = !!datasetDetailMatch;

  return (
    <header className="flex items-center justify-between border-b border-cognee-border bg-white flex-shrink-0" style={{ height: 53, paddingInline: 24, fontFamily: '"Inter", system-ui, sans-serif' }}>
      {/* Left: breadcrumbs */}
      <div className="flex items-center" style={{ gap: 8 }}>

        {/* 1. Workspace switcher */}
        <Dropdown
          trigger={
            <div className="flex items-center" style={{ gap: 6 }}>
              <div className="flex items-center justify-center rounded-[4px] flex-shrink-0" style={{ width: 20, height: 20, background: workspace.color }}>
                <span style={{ color: "#fff", fontSize: 10, fontWeight: 700 }}>{workspace.initial}</span>
              </div>
              <span style={{ fontSize: 14, fontWeight: 500, color: "#18181B", flexShrink: 0 }}>{workspace.name}</span>
              <span style={{ background: "#F0EDFF", borderRadius: 2, padding: "2px 8px", fontSize: 11, fontWeight: 500, color: "#6C5CE7", flexShrink: 0 }}>Free</span>
              <Chevron />
            </div>
          }
          width={220}
        >
          {workspaces.map((ws) => (
            <div key={ws.id} onClick={() => setWorkspace(ws)} className="cursor-pointer" style={{ display: "flex", alignItems: "center", gap: 8, padding: "8px 10px", borderRadius: 6, background: workspace.id === ws.id ? "#F0EDFF" : "transparent" }}>
              <div style={{ width: 16, height: 16, borderRadius: 3, background: ws.color, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                <span style={{ color: "#fff", fontSize: 8, fontWeight: 700 }}>{ws.initial}</span>
              </div>
              <span style={{ fontSize: 13, fontWeight: workspace.id === ws.id ? 500 : 400, color: workspace.id === ws.id ? "#6510F4" : "#3F3F46" }}>{ws.name}</span>
              {ws.type === "personal" && <span style={{ fontSize: 11, color: "#A1A1AA", marginLeft: "auto" }}>Personal</span>}
              {workspace.id === ws.id && <Check />}
            </div>
          ))}
          <div style={{ height: 1, background: "#E4E4E7", margin: "4px 0" }} />
          <div onClick={(e) => { e.stopPropagation(); setShowCreateWsModal(true); }} className="cursor-pointer" style={{ display: "flex", alignItems: "center", gap: 8, padding: "8px 10px", borderRadius: 6 }}>
            <PlusIcon />
            <span style={{ fontSize: 13, color: "#A1A1AA" }}>Create workspace</span>
          </div>
        </Dropdown>

        {/* 2. Dataset selector — hidden on pages where datasets are not relevant */}
        {!["/datasets", "/api-keys", "/connect-agent", "/connections", "/settings", "/onboarding"].includes(basePath) && (
          <>
            <Slash />
            <Dropdown
              trigger={
                <div className="flex items-center" style={{ gap: 6 }}>
                  <DatasetIcon color="#52525B" />
                  <span style={{ fontSize: 14, fontWeight: 500, color: "#18181B" }}>{selectedDataset ? selectedDataset.name : "All datasets"}</span>
                  <Chevron />
                </div>
              }
              width={240}
            >
              <div onClick={() => setSelectedDataset(null)} className="cursor-pointer" style={{ display: "flex", alignItems: "center", gap: 8, padding: "8px 12px", borderRadius: 6, background: !selectedDataset ? "#F0EDFF" : "transparent" }}>
                <span style={{ fontSize: 13, fontWeight: !selectedDataset ? 500 : 400, color: !selectedDataset ? "#6510F4" : "#3F3F46" }}>All datasets</span>
                {!selectedDataset && <Check />}
              </div>
              <div style={{ height: 1, background: "#E4E4E7", margin: "4px 0" }} />
              {datasets.map((d) => (
                <div key={d.id} onClick={() => setSelectedDataset(d)} className="cursor-pointer" style={{ display: "flex", alignItems: "center", gap: 8, padding: "8px 12px", borderRadius: 6, background: selectedDataset?.id === d.id ? "#F0EDFF" : "transparent" }}>
                  <DatasetIcon color={selectedDataset?.id === d.id ? "#6510F4" : "#3F3F46"} />
                  <span style={{ fontSize: 13, fontWeight: selectedDataset?.id === d.id ? 500 : 400, color: selectedDataset?.id === d.id ? "#6510F4" : "#3F3F46" }}>{d.name}</span>
                  {selectedDataset?.id === d.id && <Check />}
                </div>
              ))}
            </Dropdown>
          </>
        )}

        {/* 3. Page name */}
        {isDatasetDetail ? (
          <><Slash /><span style={{ fontSize: 14, fontWeight: 500, color: "#18181B" }}>Documents</span></>
        ) : basePath !== "/" && basePath !== "/dashboard" ? (
          <><Slash /><span style={{ fontSize: 14, fontWeight: 500, color: "#18181B" }}>{pageName}</span></>
        ) : null}
      </div>

      {/* Right: help + profile */}
      <div className="flex items-center gap-3">
        <HelpMenu />
        <ProfileMenu
          userName={user?.name || ""}
          userEmail={user?.email || ""}
          logoutHref={cloud ? "/api/signout" : "/api/local-signout"}
        />
      </div>

      {/* Create workspace modal */}
      {showCreateWsModal && (
        <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.3)", zIndex: 100, display: "flex", alignItems: "center", justifyContent: "center" }} onClick={() => { setShowCreateWsModal(false); setWsError(null); setWsName(""); }}>
          <div onClick={(e) => e.stopPropagation()} style={{ background: "#fff", borderRadius: 12, padding: 24, width: 420, display: "flex", flexDirection: "column", gap: 16, boxShadow: "0 16px 48px rgba(0,0,0,0.12)", fontFamily: '"Inter", system-ui, sans-serif' }}>
            <h2 style={{ fontSize: 18, fontWeight: 600, color: "#18181B", margin: 0 }}>Create workspace</h2>
            <p style={{ fontSize: 13, color: "#71717A", margin: 0 }}>Workspaces let you organize users and resources into isolated environments.</p>
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              <label style={{ fontSize: 13, fontWeight: 500, color: "#3F3F46" }}>Workspace name</label>
              <input
                autoFocus
                type="text"
                value={wsName}
                onChange={(e) => setWsName(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter") handleCreateWorkspace(); }}
                placeholder="e.g. My Team, Production, Research..."
                style={{ width: "100%", height: 40, border: `1px solid ${wsError ? "#EF4444" : "#E4E4E7"}`, borderRadius: 8, paddingInline: 14, fontSize: 14, color: "#18181B", fontFamily: "inherit", outline: "none", boxSizing: "border-box" }}
                onFocus={(e) => { e.target.style.borderColor = "#6510F4"; e.target.style.boxShadow = "0 0 0 3px rgba(101,16,244,0.1)"; }}
                onBlur={(e) => { e.target.style.borderColor = wsError ? "#EF4444" : "#E4E4E7"; e.target.style.boxShadow = "none"; }}
              />
              {wsError && <span style={{ fontSize: 12, color: "#EF4444" }}>{wsError}</span>}
            </div>
            <div style={{ display: "flex", gap: 8, justifyContent: "flex-end" }}>
              <button onClick={() => { setShowCreateWsModal(false); setWsError(null); setWsName(""); }} className="cursor-pointer" style={{ background: "#fff", border: "1px solid #E4E4E7", borderRadius: 8, padding: "8px 16px", fontSize: 13, fontWeight: 500, color: "#3F3F46", fontFamily: "inherit" }}>Cancel</button>
              <button onClick={handleCreateWorkspace} disabled={wsCreating || !wsName.trim()} className="cursor-pointer" style={{ background: wsName.trim() ? "#6510F4" : "#E4E4E7", border: "none", borderRadius: 8, padding: "8px 16px", fontSize: 13, fontWeight: 500, color: wsName.trim() ? "#fff" : "#A1A1AA", fontFamily: "inherit" }}>
                {wsCreating ? "Creating..." : "Create"}
              </button>
            </div>
          </div>
        </div>
      )}
    </header>
  );
}
