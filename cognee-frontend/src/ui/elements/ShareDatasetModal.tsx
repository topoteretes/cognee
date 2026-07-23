"use client";

import { useState } from "react";
import { Loader } from "@mantine/core";
import { useCogniInstance, useTenant } from "@/modules/tenant/TenantProvider";
import { useFilter } from "@/ui/layout/FilterContext";
import { trackEvent } from "@/modules/analytics";

type SharePermission = "read" | "write";

interface ShareDatasetModalProps {
  datasetId: string;
  datasetName: string;
  onClose: () => void;
  /** Page name used for analytics events. */
  pageName?: string;
}

/**
 * Modal for sharing a brain (dataset): with the whole workspace (grant to the
 * tenant principal — covers all current and future members) or read-only with
 * individual agents and users. Shared between the Brains list and the dataset
 * detail page. Sharing is optimistic — the backend has no "list existing
 * shares" endpoint yet, so the shared set is tracked only for this session.
 */
export default function ShareDatasetModal({ datasetId, datasetName, onClose, pageName = "Brains" }: ShareDatasetModalProps) {
  const { cogniInstance } = useCogniInstance();
  const { tenant } = useTenant();
  const { agents } = useFilter();
  const [sharedWith, setSharedWith] = useState<Set<string>>(new Set());
  const [sharing, setSharing] = useState<string | null>(null);
  const [workspacePermission, setWorkspacePermission] = useState<SharePermission>("write");

  // The tenant is itself a principal, so granting to it covers every current
  // AND future workspace member — nothing runs at member-join time.
  const workspacePrincipalId = tenant?.tenant_id ?? null;

  async function handleShare(principalId: string, permissions: SharePermission[] = ["read"]) {
    if (!cogniInstance) return;
    setSharing(principalId);
    try {
      for (const permission of permissions) {
        await cogniInstance.fetch(`/v1/permissions/datasets/${principalId}?permission_name=${permission}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify([datasetId]),
        });
      }
      trackEvent({ pageName, eventName: "dataset_shared", additionalProperties: { dataset_id: datasetId, agent_id: principalId, permission: permissions.join("+") } });
      setSharedWith((prev) => new Set([...prev, principalId]));
    } catch (err) {
      console.error("Share failed:", err);
    } finally {
      setSharing(null);
    }
  }

  function handleShareWithWorkspace() {
    if (!workspacePrincipalId) return;
    // "Can edit" grants read alongside write — the ACL stores them as separate
    // permissions, and write alone would let members cognify but not list/query.
    handleShare(workspacePrincipalId, workspacePermission === "write" ? ["read", "write"] : ["read"]);
  }

  return (
    <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.3)", backdropFilter: "blur(4px)", WebkitBackdropFilter: "blur(4px)", zIndex: 100, display: "flex", alignItems: "center", justifyContent: "center" }} onClick={onClose}>
      <div onClick={(e) => e.stopPropagation()} style={{ background: "rgba(15,15,15,0.92)", backdropFilter: "blur(16px)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 12, padding: 24, width: 480, maxHeight: "70vh", overflow: "auto", display: "flex", flexDirection: "column", gap: 16, boxShadow: "0 20px 60px rgba(0,0,0,0.6)" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <h2 style={{ fontSize: 18, fontWeight: 700, color: "#EDECEA", margin: 0 }}>Share brain</h2>
          <button onClick={onClose} className="cursor-pointer" style={{ background: "none", border: "none", color: "rgba(237,236,234,0.5)", fontSize: 18 }}>&#10005;</button>
        </div>
        <p style={{ fontSize: 13, color: "rgba(237,236,234,0.55)", margin: 0 }}>Share <strong>{datasetName}</strong> with your whole workspace, or grant read access to individual agents and users.</p>

        {workspacePrincipalId && (
          <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "10px 12px", borderRadius: 8, border: "1px solid rgba(101,16,244,0.45)", background: "rgba(101,16,244,0.08)" }}>
            <div style={{ width: 32, height: 32, borderRadius: 8, background: "#6510F4", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none"><path d="M17 21v-2a4 4 0 00-4-4H5a4 4 0 00-4 4v2M9 11a4 4 0 100-8 4 4 0 000 8zM23 21v-2a4 4 0 00-3-3.87M16 3.13a4 4 0 010 7.75" stroke="#fff" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" /></svg>
            </div>
            <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 1, minWidth: 0 }}>
              <span style={{ fontSize: 14, fontWeight: 500, color: "#EDECEA" }}>Everyone in workspace</span>
              <span style={{ fontSize: 12, color: "rgba(237,236,234,0.4)" }}>All current and future members</span>
            </div>
            {sharedWith.has(workspacePrincipalId) ? (
              <span style={{ fontSize: 12, color: "#22C55E", fontWeight: 500, display: "flex", alignItems: "center", gap: 4 }}>
                <svg width="14" height="14" viewBox="0 0 16 16" fill="none"><path d="M3.5 8.5L6.5 11.5L12.5 4.5" stroke="#22C55E" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" /></svg>
                Shared
              </span>
            ) : (
              <>
                <select
                  value={workspacePermission}
                  onChange={(e) => setWorkspacePermission(e.target.value as SharePermission)}
                  className="cursor-pointer"
                  style={{ background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.12)", borderRadius: 6, padding: "5px 8px", fontSize: 12, color: "#EDECEA", fontFamily: "inherit" }}
                >
                  <option value="write">Can edit</option>
                  <option value="read">Can view</option>
                </select>
                <button
                  onClick={handleShareWithWorkspace}
                  disabled={sharing === workspacePrincipalId}
                  className="cursor-pointer hover:bg-[#5A0ED6]"
                  style={{ display: "flex", alignItems: "center", gap: 6, background: "#6510F4", color: "#fff", border: "none", borderRadius: 6, padding: "5px 14px", fontSize: 12, fontWeight: 500 }}
                >
                  {sharing === workspacePrincipalId && <Loader size={12} color="#fff" />}
                  {sharing === workspacePrincipalId ? "Sharing..." : "Share"}
                </button>
              </>
            )}
          </div>
        )}

        {agents.length === 0 ? (
          <span style={{ fontSize: 13, color: "rgba(237,236,234,0.35)", padding: "16px 0" }}>No agents or users found.</span>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
            {agents.map((a) => {
              const isShared = sharedWith.has(a.id);
              const isSharing = sharing === a.id;
              const displayName = a.is_agent ? a.agent_type : a.email;
              const sub = a.is_agent ? a.agent_short_id : (a.email === "default_user@example.com" ? "Owner" : "User");
              return (
                <div key={a.id} style={{ display: "flex", alignItems: "center", gap: 10, padding: "10px 12px", borderRadius: 8, border: "1px solid rgba(255,255,255,0.08)" }}>
                  <div style={{ width: 32, height: 32, borderRadius: 8, background: a.is_agent ? "#6510F4" : "#3B82F6", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                    <span style={{ fontSize: 11, fontWeight: 700, color: "#fff" }}>{displayName.slice(0, 2).toUpperCase()}</span>
                  </div>
                  <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 1, minWidth: 0 }}>
                    <span style={{ fontSize: 14, fontWeight: 500, color: "#EDECEA" }}>{displayName}</span>
                    <span style={{ fontSize: 12, color: "rgba(237,236,234,0.4)" }}>{sub}</span>
                  </div>
                  {isShared ? (
                    <span style={{ fontSize: 12, color: "#22C55E", fontWeight: 500, display: "flex", alignItems: "center", gap: 4 }}>
                      <svg width="14" height="14" viewBox="0 0 16 16" fill="none"><path d="M3.5 8.5L6.5 11.5L12.5 4.5" stroke="#22C55E" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" /></svg>
                      Shared
                    </span>
                  ) : (
                    <button
                      onClick={() => handleShare(a.id)}
                      disabled={isSharing}
                      className="cursor-pointer hover:bg-[#5A0ED6]"
                      style={{ display: "flex", alignItems: "center", gap: 6, background: "#6510F4", color: "#fff", border: "none", borderRadius: 6, padding: "5px 14px", fontSize: 12, fontWeight: 500 }}
                    >
                      {isSharing && <Loader size={12} color="#fff" />}
                      {isSharing ? "Sharing..." : "Share"}
                    </button>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
