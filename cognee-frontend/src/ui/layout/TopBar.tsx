"use client";

import { useEffect, useState, useCallback } from "react";
import { usePathname } from "next/navigation";
import { useCurrentUser } from "@/modules/users/useCurrentUser";
import { useUser } from "@/modules/users/UserContext";
import { displayName } from "@/modules/users/profileName";
import getLocalUser from "@/modules/users/getLocalUser";
import CogneeUser from "@/modules/users/CogneeUser";
import isCloudEnvironment from "@/utils/isCloudEnvironment";

import Image from "next/image";
import Link from "next/link";
import HelpMenu from "./HelpMenu";
import ProfileMenu from "./ProfileMenu";
import { useFilter } from "./FilterContext";
import { useTenant } from "@/modules/tenant/TenantContext";
import isTenantSuspended from "@/modules/tenant/isTenantSuspended";
import WorkspaceBadge from "./WorkspaceBadge";
import type { WorkspaceBadgeState } from "./WorkspaceBadge";
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
  "/datasets": "Datasets", "/sessions": "Sessions", "/search": "Search",
  "/knowledge-graph": "Mindmap",
  // /business is the same page as /knowledge-graph (page.tsx re-exports
  // BusinessPage) but had no entry here, so visiting it directly fell back
  // to a derived "Business" breadcrumb while the nav-linked /knowledge-graph
  // correctly showed "Mindmap" for identical content.
  "/business": "Mindmap",
  "/integrations": "Integrations",
  "/api-keys": "API Keys",
  "/settings": "Settings",
  "/onboarding": "Onboarding", "/members": "Members",
  "/memory-gap-analysis": "Memory coverage",
  "/analytics": "Analytics",
  "/activity": "Activity",
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
  const { userMe } = useUser();
  const pathname = usePathname();
  const { workspace, workspaces, setWorkspace } = useFilter();
  const { requestCreateWorkspace, availableTenants, wakeState, tenant } = useTenant();

  useEffect(() => {
    if (!cloud) getLocalUser().then((u) => { if (u) setLocalUser(u); });
  }, [cloud]);

  // Derive page label
  const basePath = "/" + (pathname.split("/").filter(Boolean)[0] || "");
  const pageName = ROUTE_LABELS[basePath] || basePath.slice(1).charAt(0).toUpperCase() + basePath.slice(2).replace(/-/g, " ");

  // Check if we're on a dataset detail page
  const isDatasetDetail = /^\/datasets\/.+$/.test(pathname);

  // The trigger badge carries the state too, because the picker is shut for all
  // but a second or two of a wake and the caption inside it is unreachable then.
  //
  // Keyed on the connected tenant, never on `workspace`: that one is derived
  // from the list and falls back to its first entry when the active tenant is
  // missing from it, so keying the suspension lookup on it made the trigger
  // report a DIFFERENT workspace's state — the same trap the picker rows below
  // spell out and avoid. When the two disagree there is nothing trustworthy to
  // say, so the badge says nothing.
  const connectedId = tenant?.tenant_id;
  const triggerIsConnected = !!connectedId && connectedId === workspace.id;
  const connectedInfo = availableTenants.find((t) => t.id === connectedId);
  const triggerBadgeState: WorkspaceBadgeState = !triggerIsConnected
    ? "active"
    : connectedInfo && !connectedInfo.ownerHasSubscription
      ? "blocked"
      : wakeState === "waking"
        ? "waking"
        : isTenantSuspended(availableTenants, connectedId)
          ? "asleep"
          : "active";

  return (
    <header className="flex items-center justify-between flex-shrink-0" style={{ height: 53, paddingInline: 24, position: "relative", zIndex: 300, background: "rgba(0,0,0,0.65)", backdropFilter: "blur(12px)", borderBottom: "1px solid rgba(255,255,255,0.08)" }}>
      {/* Left: logo + breadcrumbs */}
      <div className="flex items-center" style={{ gap: 8 }}>
        {/* Logo fixed-width box so workspace aligns with right edge of navbar (240px - 24px padding) */}
        <div style={{ width: 240, flexShrink: 0, display: "flex", alignItems: "center" }}>
          <Image src="/cognee-logo-black.svg" alt="Cognee" width={110} height={32} loading="eager" style={{ width: 110, height: "auto", flexShrink: 0, filter: "invert(1)" }} />
        </div>

        {/* 1. Workspace switcher */}
        <Dropdown
          trigger={
            <div className="flex items-center" style={{ gap: 6 }}>
              <WorkspaceBadge initial={workspace.initial} color={workspace.color} size={20} state={triggerBadgeState} />
              <span style={{ fontSize: 14, fontWeight: 500, color: "#EDECEA", flexShrink: 0 }}>{workspace.name}</span>
              <Chevron />
            </div>
          }
          width={220}
        >
          {workspaces.map((ws) => {
            const tenantInfo = availableTenants.find((t) => t.id === ws.id);
            const blocked = tenantInfo ? !tenantInfo.ownerHasSubscription : false;
            // Scaled to zero. Deliberately NOT styled like `blocked`: blocked is
            // a dead end (grayed out, click disabled), while an asleep workspace
            // opens normally and just takes a moment. WorkspaceBadge draws it
            // hollow rather than grey, and lightens the workspace colour so the
            // outline survives on the dark chrome across the whole palette.
            const asleep = !blocked && isTenantSuspended(availableTenants, ws.id);
            // Keyed on the connected tenant, not on `workspace`: that is derived
            // from the list and falls back to its first entry when the active
            // tenant is missing from it, which would caption the wrong row.
            // Not gated on `asleep` either — a wake discovered by the pod probe
            // (a stale list, or a workspace that fell asleep with the tab open)
            // never shows as suspended in the list it came from.
            const isWaking = !blocked && tenant?.tenant_id === ws.id && wakeState === "waking";
            // A wake that gave up leaves the list still saying "suspended"
            // (only the success path refetches it), so without this the row
            // invites a click to wake a workspace the content area has just
            // reported as failed — and it is the active row, so the click is a
            // no-op anyway.
            const wakeFailed = !blocked && tenant?.tenant_id === ws.id && wakeState === "failed";
            return (
              <div key={ws.id} onClick={() => !blocked && setWorkspace(ws)} className={blocked ? "" : "cursor-pointer"} style={{ display: "flex", alignItems: "center", gap: 8, padding: "8px 10px", borderRadius: 6, background: workspace.id === ws.id ? "rgba(188,155,255,0.20)" : "transparent", opacity: blocked ? 0.5 : 1, cursor: blocked ? "default" : "pointer" }}>
                <WorkspaceBadge
                  initial={ws.initial}
                  color={ws.color}
                  size={16}
                  state={blocked ? "blocked" : isWaking ? "waking" : (asleep || wakeFailed) ? "asleep" : "active"}
                />
                <div style={{ display: "flex", flexDirection: "column" }}>
                  <span style={{ fontSize: 13, fontWeight: workspace.id === ws.id ? 500 : 400, color: blocked ? "#71717A" : workspace.id === ws.id ? "rgba(188,155,255,0.60)" : "#EDECEA" }}>{ws.name}</span>
                  {blocked && <span style={{ fontSize: 10, color: "rgba(237,236,234,0.55)" }}>No active subscription</span>}
                  {(asleep || isWaking || wakeFailed) && (
                    <span style={{ fontSize: 10, color: "rgba(237,236,234,0.55)" }}>
                      {wakeFailed ? "Didn't wake up" : isWaking ? "Waking up…" : "Asleep · wakes when you open it"}
                    </span>
                  )}
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
          <>
            <Slash />
            <Link href="/datasets" className="hover:opacity-70" style={{ fontSize: 14, fontWeight: 500, color: "rgba(237,236,234,0.55)" }}>Datasets</Link>
            <Slash />
            <span style={{ fontSize: 14, fontWeight: 500, color: "rgba(237,236,234,0.7)" }}>Documents</span>
          </>
        ) : basePath !== "/" && basePath !== "/dashboard" ? (
          <><Slash /><span style={{ fontSize: 14, fontWeight: 500, color: "rgba(237,236,234,0.7)" }}>{pageName}</span></>
        ) : null}
      </div>

      {/* Right: help + profile */}
      <div className="flex items-center gap-3">
        <HelpMenu />
        <ProfileMenu
          // The name the person set on their profile, else the sign-in provider's.
          // Without this the header kept showing the old one after a rename, while
          // everyone else in the workspace already saw the new one.
          userName={cloud ? displayName(userMe?.profile ?? null, user?.name) : user?.name || ""}
          userEmail={user?.email || ""}
          logoutHref={cloud ? "/api/signout" : "/api/local-signout"}
        />
      </div>

    </header>
  );
}
