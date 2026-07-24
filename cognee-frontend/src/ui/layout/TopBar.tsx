"use client";

import { useEffect, useState, useCallback } from "react";
import { usePathname } from "next/navigation";
import { useCurrentUser } from "@/modules/users/useCurrentUser";
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

function Chevron({ color = "rgba(255,255,255,0.45)", up = false }: { color?: string; up?: boolean }) {
  return <svg width="12" height="12" viewBox="0 0 12 12" fill="none" style={{ flexShrink: 0 }}><path d={up ? "M3 7.5L6 4.5L9 7.5" : "M3 4.5L6 7.5L9 4.5"} stroke={color} strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" /></svg>;
}
function Check() {
  return <svg width="12" height="12" viewBox="0 0 12 12" fill="none" style={{ marginLeft: "auto", flexShrink: 0 }}><path d="M2.5 6L5 8.5L9.5 3.5" stroke="#6510F4" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" /></svg>;
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
        <div onClick={close} style={{ position: "absolute", top: 32, left: 0, width, background: "#1a1a1a", borderRadius: 10, boxShadow: "0px 8px 30px rgba(0,0,0,0.5), 0px 0px 0px 1px rgba(255,255,255,0.08)", padding: 6, zIndex: 50, maxHeight: "min(480px, calc(100vh - 120px))", overflowY: "auto" }}>
          {children}
        </div>
      )}
    </div>
  );
}

function Slash() {
  return <span style={{ color: "rgba(255,255,255,0.2)", fontSize: 16, flexShrink: 0 }}>/</span>;
}

// ── Page name from path ──

const ROUTE_LABELS: Record<string, string> = {
  "/": "Overview", "/dashboard": "Overview",
  "/datasets": "Brain", "/sessions": "Sessions", "/search": "Search",
  "/knowledge-graph": "Mindmap",
  "/schema": "Memory Schema",
  "/integrations": "Integrations",
  "/api-keys": "API Keys",
  "/settings": "Settings",
  "/onboarding": "Onboarding", "/members": "Members",
};

// ── TopBar ──

function PlusIcon() {
  return <svg width="12" height="12" viewBox="0 0 12 12" fill="none" style={{ flexShrink: 0 }}><line x1="6" y1="2" x2="6" y2="10" stroke="#6510F4" strokeWidth="1.5" strokeLinecap="round" /><line x1="2" y1="6" x2="10" y2="6" stroke="#6510F4" strokeWidth="1.5" strokeLinecap="round" /></svg>;
}

export default function TopBar() {
  const [localUser, setLocalUser] = useState<CogneeUser>();
  const cloud = isCloudEnvironment();
  const { data: cloudUser } = useCurrentUser(cloud);
  const user = cloud ? cloudUser : localUser;
  const pathname = usePathname();
  const { workspace, workspaces, setWorkspace } = useFilter();
  const { requestCreateWorkspace, availableTenants } = useTenant();

  useEffect(() => {
    if (!cloud) getLocalUser().then((u) => { if (u) setLocalUser(u); });
  }, [cloud]);

  // Derive page label
  const basePath = "/" + (pathname.split("/").filter(Boolean)[0] || "");
  const pageName = ROUTE_LABELS[basePath] || basePath.slice(1).charAt(0).toUpperCase() + basePath.slice(2).replace(/-/g, " ");

  // Check if we're on a dataset detail page
  const isDatasetDetail = /^\/datasets\/.+$/.test(pathname);

  return (
    <header className="flex items-center justify-between flex-shrink-0" style={{ height: 53, paddingInline: 24, position: "relative", zIndex: 300, background: "rgba(0,0,0,0.65)", backdropFilter: "blur(12px)", borderBottom: "1px solid rgba(255,255,255,0.08)" }}>
      {/* Left: logo + breadcrumbs */}
      <div className="flex items-center" style={{ gap: 8 }}>
        {/* Logo fixed-width box so workspace aligns with right edge of navbar (240px - 24px padding) */}
        <div style={{ width: 240, flexShrink: 0, display: "flex", alignItems: "center" }}>
          <Image src="/cognee-logo-black.svg" alt="Cognee" width={110} height={24} style={{ flexShrink: 0, filter: "invert(1)" }} />
        </div>

        {/* 1. Workspace switcher */}
        <Dropdown
          trigger={
            <div className="flex items-center" style={{ gap: 6 }}>
              <div className="flex items-center justify-center rounded-[4px] flex-shrink-0" style={{ width: 20, height: 20, background: workspace.color }}>
                <span style={{ color: "#fff", fontSize: 10, fontWeight: 700 }}>{workspace.initial}</span>
              </div>
              <span style={{ fontSize: 14, fontWeight: 500, color: "#EDECEA", flexShrink: 0 }}>{workspace.name}</span>
              <Chevron />
            </div>
          }
          width={220}
        >
          {workspaces.map((ws) => {
            const tenantInfo = availableTenants.find((t) => t.id === ws.id);
            const blocked = tenantInfo ? !tenantInfo.ownerHasSubscription : false;
            return (
              <div key={ws.id} onClick={() => !blocked && setWorkspace(ws)} className={blocked ? "" : "cursor-pointer"} style={{ display: "flex", alignItems: "center", gap: 8, padding: "8px 10px", borderRadius: 6, background: workspace.id === ws.id ? "rgba(188,155,255,0.20)" : "transparent", opacity: blocked ? 0.5 : 1, cursor: blocked ? "default" : "pointer" }}>
                <div style={{ width: 16, height: 16, borderRadius: 3, background: blocked ? "#555" : ws.color, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                  <span style={{ color: "#fff", fontSize: 8, fontWeight: 700 }}>{ws.initial}</span>
                </div>
                <div style={{ display: "flex", flexDirection: "column" }}>
                  <span style={{ fontSize: 13, fontWeight: workspace.id === ws.id ? 500 : 400, color: blocked ? "#71717A" : workspace.id === ws.id ? "rgba(188,155,255,0.60)" : "#EDECEA" }}>{ws.name}</span>
                  {blocked && <span style={{ fontSize: 10, color: "rgba(237,236,234,0.55)" }}>No active subscription</span>}
                </div>
                {workspace.id === ws.id && !blocked && <Check />}
              </div>
            );
          })}
          {cloud && (
            <>
              <div style={{ height: 1, background: "rgba(255,255,255,0.1)", margin: "4px 0" }} />
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

        {/* 2. Page name */}
        {isDatasetDetail ? (
          <><Slash /><span style={{ fontSize: 14, fontWeight: 500, color: "rgba(237,236,234,0.7)" }}>Documents</span></>
        ) : basePath !== "/" && basePath !== "/dashboard" ? (
          <><Slash /><span style={{ fontSize: 14, fontWeight: 500, color: "rgba(237,236,234,0.7)" }}>{pageName}</span></>
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
