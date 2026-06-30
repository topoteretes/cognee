"use client";

import { useState } from "react";
import { useCogniInstance } from "@/modules/tenant/TenantProvider";
import { useFilter } from "@/ui/layout/FilterContext";
import { trackEvent } from "@/modules/analytics";

interface ShareDatasetModalProps {
  datasetId: string;
  datasetName: string;
  onClose: () => void;
  /** Page name used for analytics events. */
  pageName?: string;
}

/**
 * Modal for granting read access to a brain (dataset) to other agents and
 * users in the tenant. Shared between the Brains list and the dataset detail
 * page. Sharing is optimistic — the backend has no "list existing shares" or
 * "revoke" endpoint yet, so the shared set is tracked only for this session.
 */
export default function ShareDatasetModal({ datasetId, datasetName, onClose, pageName = "Brains" }: ShareDatasetModalProps) {
  const { cogniInstance } = useCogniInstance();
  const { agents } = useFilter();
  const [sharedWith, setSharedWith] = useState<Set<string>>(new Set());
  const [sharing, setSharing] = useState<string | null>(null);

  async function handleShare(principalId: string) {
    if (!cogniInstance) return;
    setSharing(principalId);
    try {
      await cogniInstance.fetch(`/v1/permissions/datasets/${principalId}?permission_name=read`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify([datasetId]),
      });
      trackEvent({ pageName, eventName: "dataset_shared", additionalProperties: { dataset_id: datasetId, agent_id: principalId } });
      setSharedWith((prev) => new Set([...prev, principalId]));
    } catch (err) {
      console.error("Share failed:", err);
    } finally {
      setSharing(null);
    }
  }

  return (
    <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.3)", backdropFilter: "blur(4px)", WebkitBackdropFilter: "blur(4px)", zIndex: 100, display: "flex", alignItems: "center", justifyContent: "center" }} onClick={onClose}>
      <div onClick={(e) => e.stopPropagation()} style={{ background: "rgba(15,15,15,0.92)", backdropFilter: "blur(16px)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 12, padding: 24, width: 480, maxHeight: "70vh", overflow: "auto", display: "flex", flexDirection: "column", gap: 16, boxShadow: "0 20px 60px rgba(0,0,0,0.6)" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <h2 style={{ fontSize: 18, fontWeight: 700, color: "#EDECEA", margin: 0 }}>Share brain</h2>
          <button onClick={onClose} className="cursor-pointer" style={{ background: "none", border: "none", color: "rgba(237,236,234,0.5)", fontSize: 18 }}>&#10005;</button>
        </div>
        <p style={{ fontSize: 13, color: "rgba(237,236,234,0.55)", margin: 0 }}>Grant read access to <strong>{datasetName}</strong> for agents and users.</p>

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
                      style={{ background: "#6510F4", color: "#fff", border: "none", borderRadius: 6, padding: "5px 14px", fontSize: 12, fontWeight: 500 }}
                    >
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
