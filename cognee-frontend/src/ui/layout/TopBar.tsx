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
import { useNavbar } from "./NavbarContext";
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

// Compact Cognee brain-mark shown on the collapsed desktop rail, where the full
// wordmark no longer fits. The path is the mark glyph lifted from the brand
// logo so it stays pixel-consistent with the wordmark it replaces.
function CogneeMark({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      width="21"
      height="24"
      viewBox="607 464 1441 1627"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      style={{ flexShrink: 0 }}
      role="img"
      aria-label="Cognee"
    >
      <path d="M1514.86 2090.35C1427.39 2090.35 1356.25 2014.59 1356.25 1921.47V632.789C1356.25 609.821 1341.17 593.817 1327.65 593.817C1314.13 593.817 1299.05 609.821 1299.05 632.789V1921.47C1299.05 2014.59 1227.91 2090.35 1140.45 2090.35C1052.98 2090.35 981.846 2014.59 981.846 1921.47V913.389C981.846 890.421 966.766 874.417 953.246 874.417C939.725 874.417 924.645 890.421 924.645 913.389V1547.34C924.645 1640.45 853.508 1716.22 766.042 1716.22C678.577 1716.22 607.439 1640.45 607.439 1547.34V1240.76C607.439 1147.64 683.257 1071.88 776.442 1071.88C782.786 1071.88 788.819 1072.81 794.643 1074.47V913.389C794.643 820.271 865.78 744.509 953.246 744.509C1040.71 744.509 1111.85 820.271 1111.85 913.389V1921.47C1111.85 1944.44 1126.93 1960.44 1140.45 1960.44C1153.97 1960.44 1169.05 1944.44 1169.05 1921.47V632.789C1169.05 539.671 1240.19 463.909 1327.65 463.909C1415.12 463.909 1486.26 539.671 1486.26 632.789V1921.47C1486.26 1944.44 1501.34 1960.44 1514.86 1960.44C1528.38 1960.44 1543.46 1944.44 1543.46 1921.47V913.389C1543.46 820.271 1614.59 744.509 1702.06 744.509C1789.52 744.509 1860.66 820.271 1860.66 913.389V1074.47C1866.49 1072.81 1872.52 1071.88 1878.86 1071.88C1972.05 1071.88 2047.87 1147.64 2047.87 1240.76V1547.34C2047.87 1640.45 1976.73 1716.22 1889.26 1716.22C1801.8 1716.22 1730.66 1640.45 1730.66 1547.34V913.389C1730.66 890.421 1715.58 874.417 1702.06 874.417C1688.54 874.417 1673.46 890.421 1673.46 913.389V1921.47C1673.46 2014.59 1602.32 2090.35 1514.86 2090.35ZM1860.66 1199.19V1547.34C1860.66 1570.3 1875.74 1586.31 1889.26 1586.31C1902.78 1586.31 1917.86 1570.3 1917.86 1547.34V1240.76C1917.86 1219.66 1899.97 1201.78 1878.86 1201.78C1872.52 1201.78 1866.49 1200.85 1860.66 1199.19ZM794.643 1199.19C788.819 1200.85 782.786 1201.78 776.442 1201.78C755.33 1201.78 737.442 1219.66 737.442 1240.76V1547.34C737.442 1570.3 752.522 1586.31 766.042 1586.31C779.562 1586.31 794.643 1570.3 794.643 1547.34V1199.19Z" fill="#fff" />
    </svg>
  );
}

export default function TopBar() {
  const [localUser, setLocalUser] = useState<CogneeUser>();
  const cloud = isCloudEnvironment();
  const { data: cloudUser } = useCurrentUser(cloud);
  const user = cloud ? cloudUser : localUser;
  const pathname = usePathname();
  const { workspace, workspaces, setWorkspace } = useFilter();
  const { requestCreateWorkspace, availableTenants } = useTenant();
  const { isCollapsed } = useNavbar();

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
        {/* Logo box width tracks the sidebar so the workspace switcher stays
            aligned with the navbar's right edge in both states. On the collapsed
            desktop rail the full wordmark is swapped for the compact mark. */}
        <div
          className={`items-center transition-[width] duration-200 ease-in-out w-[240px] ${isCollapsed ? "sm:w-[72px] sm:justify-center" : "sm:w-[240px]"}`}
          style={{ flexShrink: 0, display: "flex" }}
        >
          <Image
            src="/cognee-logo-black.svg"
            alt="Cognee"
            width={110}
            height={24}
            className={`block ${isCollapsed ? "sm:hidden" : "sm:block"}`}
            style={{ flexShrink: 0, filter: "invert(1)" }}
          />
          <CogneeMark className={`hidden ${isCollapsed ? "sm:block" : "sm:hidden"}`} />
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
