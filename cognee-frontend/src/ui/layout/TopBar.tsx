"use client";

import { useEffect, useState, useCallback } from "react";
import { usePathname } from "next/navigation";
import getUser from "@/modules/users/getUser";
import getLocalUser from "@/modules/users/getLocalUser";
import CogneeUser from "@/modules/users/CogneeUser";
import isCloudEnvironment from "@/utils/isCloudEnvironment";

import Image from "next/image";
import HelpMenu from "./HelpMenu";
import ProfileMenu from "./ProfileMenu";
import { useFilter } from "./FilterContext";
import { useTenant } from "@/modules/tenant/TenantContext";
import useBoolean from "@/utils/useBoolean";
import useOutsideClick from "@/utils/useOutsideClick";

// ── Icons ──

function Chevron({ color = "#A1A1AA", up = false }: { color?: string; up?: boolean }) {
  return <svg width="12" height="12" viewBox="0 0 12 12" fill="none" style={{ flexShrink: 0 }}><path d={up ? "M3 7.5L6 4.5L9 7.5" : "M3 4.5L6 7.5L9 4.5"} stroke={color} strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" /></svg>;
}
function Check() {
  return <svg width="12" height="12" viewBox="0 0 12 12" fill="none" style={{ marginLeft: "auto", flexShrink: 0 }}><path d="M2.5 6L5 8.5L9.5 3.5" stroke="#6510F4" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" /></svg>;
}
function DatasetIcon({ color = "#52525B" }: { color?: string }) {
  return <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0 }}><ellipse cx="12" cy="5" rx="9" ry="3" /><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3" /><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5" /></svg>;
}
function Slash() {
  return <span style={{ color: "#D4D4D8", fontSize: 16, flexShrink: 0 }}>/</span>;
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
  "/datasets": "Brains", "/search": "Search",
  "/knowledge-graph": "Knowledge Graph",
  "/connect-agent": "Connect Agent",
  "/connections": "Connections", "/api-keys": "API Keys",
  "/settings": "Settings",
  "/onboarding": "Onboarding", "/members": "Members",
};

// ── TopBar ──

function PlusIcon() {
  return <svg width="12" height="12" viewBox="0 0 12 12" fill="none" style={{ flexShrink: 0 }}><line x1="6" y1="2" x2="6" y2="10" stroke="#6510F4" strokeWidth="1.5" strokeLinecap="round" /><line x1="2" y1="6" x2="10" y2="6" stroke="#6510F4" strokeWidth="1.5" strokeLinecap="round" /></svg>;
}

export default function TopBar() {
  const [user, setUser] = useState<CogneeUser>();
  const cloud = isCloudEnvironment();
  const pathname = usePathname();
  const { workspace, workspaces, setWorkspace, selectedDataset, setSelectedDataset, datasets } = useFilter();
  const { requestCreateWorkspace, availableTenants } = useTenant();

  useEffect(() => {
    if (cloud) { getUser().then(setUser); }
    else { getLocalUser().then((u) => { if (u) setUser(u); }); }
  }, [cloud]);

  // Derive page label
  const basePath = "/" + (pathname.split("/").filter(Boolean)[0] || "");
  const pageName = ROUTE_LABELS[basePath] || basePath.slice(1).charAt(0).toUpperCase() + basePath.slice(2).replace(/-/g, " ");

  // Check if we're on a dataset detail page
  const isDatasetDetail = /^\/datasets\/.+$/.test(pathname);

  // Pages where the dataset selector is NOT relevant
  const datasetHiddenPaths = ["/datasets", "/api-keys", "/connect-agent", "/connections", "/settings", "/onboarding", "/members"];

  return (
    <header className="flex items-center justify-between border-b border-cognee-border bg-white flex-shrink-0" style={{ height: 53, paddingInline: 24, fontFamily: '"Inter", system-ui, sans-serif', position: "relative", zIndex: 300 }}>
      {/* Left: logo + breadcrumbs */}
      <div className="flex items-center" style={{ gap: 8 }}>
        {/* Logo fixed-width box so workspace aligns with right edge of navbar (240px - 24px padding) */}
        <div style={{ width: 240, flexShrink: 0, display: "flex", alignItems: "center" }}>
          <Image src="/cognee-logo-black.svg" alt="Cognee" width={110} height={24} style={{ flexShrink: 0 }} />
        </div>

        {/* 1. Workspace switcher */}
        <Dropdown
          trigger={
            <div className="flex items-center" style={{ gap: 6 }}>
              <div className="flex items-center justify-center rounded-[4px] flex-shrink-0" style={{ width: 20, height: 20, background: workspace.color }}>
                <span style={{ color: "#fff", fontSize: 10, fontWeight: 700 }}>{workspace.initial}</span>
              </div>
              <span style={{ fontSize: 14, fontWeight: 500, color: "#18181B", flexShrink: 0 }}>{workspace.name}</span>
              <Chevron />
            </div>
          }
          width={220}
        >
          {workspaces.map((ws) => {
            const tenantInfo = availableTenants.find((t) => t.id === ws.id);
            const blocked = tenantInfo ? !tenantInfo.ownerHasSubscription : false;
            return (
              <div key={ws.id} onClick={() => !blocked && setWorkspace(ws)} className={blocked ? "" : "cursor-pointer"} style={{ display: "flex", alignItems: "center", gap: 8, padding: "8px 10px", borderRadius: 6, background: workspace.id === ws.id ? "#F0EDFF" : "transparent", opacity: blocked ? 0.5 : 1, cursor: blocked ? "default" : "pointer" }}>
                <div style={{ width: 16, height: 16, borderRadius: 3, background: blocked ? "#D4D4D8" : ws.color, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                  <span style={{ color: "#fff", fontSize: 8, fontWeight: 700 }}>{ws.initial}</span>
                </div>
                <div style={{ display: "flex", flexDirection: "column" }}>
                  <span style={{ fontSize: 13, fontWeight: workspace.id === ws.id ? 500 : 400, color: blocked ? "#A1A1AA" : workspace.id === ws.id ? "#6510F4" : "#3F3F46" }}>{ws.name}</span>
                  {blocked && <span style={{ fontSize: 10, color: "#A1A1AA" }}>No active subscription</span>}
                </div>
                {workspace.id === ws.id && !blocked && <Check />}
              </div>
            );
          })}
          {cloud && (
            <>
              <div style={{ height: 1, background: "#E4E4E7", margin: "4px 0" }} />
              <div
                onClick={requestCreateWorkspace}
                className="cursor-pointer"
                style={{ display: "flex", alignItems: "center", gap: 8, padding: "8px 10px", borderRadius: 6 }}
              >
                <PlusIcon />
                <span style={{ fontSize: 13, fontWeight: 500, color: "#6510F4" }}>
                  Create new workspace
                </span>
              </div>
            </>
          )}
        </Dropdown>

        {/* 2. Dataset selector — hidden on pages where datasets are not relevant */}
        {!datasetHiddenPaths.includes(basePath) && (
          <>
            <Slash />
            <Dropdown
              trigger={
                <div className="flex items-center" style={{ gap: 6 }}>
                  <DatasetIcon color="#52525B" />
                  <span style={{ fontSize: 14, fontWeight: 500, color: "#18181B" }}>{selectedDataset ? selectedDataset.name : "All brains"}</span>
                  <Chevron />
                </div>
              }
              width={240}
            >
              <div onClick={() => setSelectedDataset(null)} className="cursor-pointer" style={{ display: "flex", alignItems: "center", gap: 8, padding: "8px 12px", borderRadius: 6, background: !selectedDataset ? "#F0EDFF" : "transparent" }}>
                <span style={{ fontSize: 13, fontWeight: !selectedDataset ? 500 : 400, color: !selectedDataset ? "#6510F4" : "#3F3F46" }}>All brains</span>
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

    </header>
  );
}
