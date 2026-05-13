"use client";

import { useState, useEffect } from "react";
import { useCogniInstance, useTenant } from "@/modules/tenant/TenantProvider";
import getApiKeys from "@/modules/apiKeys/getApiKeys";
import createApiKey from "@/modules/apiKeys/createAPIKey";
import deleteApiKey from "@/modules/apiKeys/deleteAPIKey";

interface ApiKey {
  id: string;
  name: string;
  label: string;
  key: string;
  isNew?: boolean;
}

function CopyIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#A1A1AA" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
      <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
    </svg>
  );
}

function CheckIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
      <path d="M3.5 8.5L6.5 11.5L12.5 4.5" stroke="#22C55E" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

const localApiUrl = process.env.NEXT_PUBLIC_LOCAL_API_URL || "http://localhost:8000";

export default function ApiKeysPage() {
  const { cogniInstance, serviceUrl, isInitializing } = useCogniInstance();
  const { tenant } = useTenant();
  const [keys, setKeys] = useState<ApiKey[]>([]);
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [copiedField, setCopiedField] = useState<string | null>(null);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [newName, setNewName] = useState("");
  const [userId, setUserId] = useState<string | null>(null);

  // Use tenant ID from context (set during provisioning), not from /users/me
  const tenantId = tenant?.tenant_id || null;
  const isDev = serviceUrl?.includes("dev-aws") || serviceUrl?.includes("dev.cloud");
  const apiDocsUrl = serviceUrl ? `${serviceUrl}/docs` : null;

  useEffect(() => {
    if (isInitializing) return;
    loadKeys();
    // Fetch user ID (graceful) — only if tenant instance is available
    if (cogniInstance) {
      cogniInstance.fetch("/v1/users/me").then((r) => r.ok ? r.json() : null).then((data) => {
        if (data) setUserId(data.id || null);
      }).catch(() => {});
    }
  }, [cogniInstance, isInitializing]);

  async function loadKeys() {
    try {
      const data = await getApiKeys();
      setKeys(Array.isArray(data) ? data.map((k) => ({ id: k.id, name: k.name || k.label || "", label: k.label, key: k.api_key, isNew: false })) : []);
    } catch {
      setKeys([]);
    } finally {
      setLoading(false);
    }
  }

  async function handleCreate() {
    if (!newName.trim()) return;
    setCreating(true);
    try {
      const newKey = await createApiKey({ name: newName.trim(), noRedirectOnAuth: true });
      setNewName("");
      setShowCreateModal(false);
      // Reload the list, then mark the new key so the full value is shown once
      const data = await getApiKeys();
      const mapped = Array.isArray(data) ? data.map((k) => {
        const mapped_key = { id: k.id, name: k.name || k.label || "", label: k.label, key: k.api_key, isNew: false };
        // Match the newly created key and show its full value
        if (k.api_key === newKey || (newKey && k.api_key.startsWith(newKey.slice(0, 8)))) {
          mapped_key.key = newKey;
          mapped_key.isNew = true;
        }
        return mapped_key;
      }) : [];
      setKeys(mapped);
    } catch (err) {
      console.error("Failed to create key:", err);
    } finally {
      setCreating(false);
    }
  }

  async function handleRevoke(id: string) {
    try {
      await deleteApiKey(id);
      setKeys((prev) => prev.filter((k) => k.id !== id));
    } catch (err) {
      console.error("Failed to revoke key:", err);
    }
  }

  function handleCopy(id: string, key: string) {
    navigator.clipboard.writeText(key).catch(() => {});
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 1500);
  }

  if (loading || isInitializing) {
    return (
      <div style={{ padding: 32, display: "flex", alignItems: "center", justifyContent: "center", height: "100%" }}>
        <span style={{ fontSize: 14, color: "#71717A" }}>Loading API keys...</span>
      </div>
    );
  }

  return (
    <div style={{ padding: 32, display: "flex", flexDirection: "column", gap: 24, fontFamily: '"Inter", system-ui, sans-serif' }}>
      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          <h1 style={{ fontSize: 24, fontWeight: 600, color: "#18181B", margin: 0 }}>API Keys</h1>
          <span style={{ fontSize: 14, color: "#71717A" }}>Manage keys for programmatic access to the Cognee API.</span>
        </div>
        <button
          onClick={() => setShowCreateModal(true)}
          className="cursor-pointer hover:bg-cognee-purple-hover"
          style={{ background: "#6510F4", color: "#fff", border: "none", borderRadius: 6, padding: "8px 16px", fontSize: 13, fontWeight: 500, display: "flex", alignItems: "center", gap: 6 }}
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="12" y1="5" x2="12" y2="19" /><line x1="5" y1="12" x2="19" y2="12" /></svg>
          Create new key
        </button>
      </div>

      {/* Create modal */}
      {showCreateModal && (
        <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.3)", zIndex: 100, display: "flex", alignItems: "center", justifyContent: "center" }} onClick={() => setShowCreateModal(false)}>
          <div onClick={(e) => e.stopPropagation()} style={{ background: "#fff", borderRadius: 12, padding: 24, width: 420, display: "flex", flexDirection: "column", gap: 16, boxShadow: "0 16px 48px rgba(0,0,0,0.12)" }}>
            <h2 style={{ fontSize: 18, fontWeight: 600, color: "#18181B", margin: 0 }}>Create API key</h2>
            <p style={{ fontSize: 13, color: "#71717A", margin: 0 }}>Give your key a name to identify it later.</p>
            <input
              autoFocus
              type="text"
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter") handleCreate(); }}
              placeholder="e.g. Production, CI/CD, Local Dev..."
              style={{ width: "100%", height: 40, border: "1px solid #E4E4E7", borderRadius: 8, paddingInline: 14, fontSize: 14, color: "#18181B", fontFamily: "inherit", outline: "none", boxSizing: "border-box" }}
              onFocus={(e) => { e.target.style.borderColor = "#6510F4"; e.target.style.boxShadow = "0 0 0 3px rgba(101,16,244,0.1)"; }}
              onBlur={(e) => { e.target.style.borderColor = "#E4E4E7"; e.target.style.boxShadow = "none"; }}
            />
            <div style={{ display: "flex", gap: 8, justifyContent: "flex-end" }}>
              <button onClick={() => { setShowCreateModal(false); setNewName(""); }} className="cursor-pointer" style={{ background: "#fff", border: "1px solid #E4E4E7", borderRadius: 8, padding: "8px 16px", fontSize: 13, fontWeight: 500, color: "#3F3F46", fontFamily: "inherit" }}>Cancel</button>
              <button onClick={handleCreate} disabled={creating || !newName.trim()} className="cursor-pointer" style={{ background: newName.trim() ? "#6510F4" : "#E4E4E7", border: "none", borderRadius: 8, padding: "8px 16px", fontSize: 13, fontWeight: 500, color: newName.trim() ? "#fff" : "#A1A1AA", fontFamily: "inherit" }}>
                {creating ? "Creating..." : "Create"}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Connection details */}
      <div style={{ background: "#fff", border: "1px solid #E4E4E7", borderRadius: 12, padding: "20px 24px", display: "flex", flexDirection: "column", gap: 16 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#6510F4" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M18 8h1a4 4 0 010 8h-1" /><path d="M6 8H5a4 4 0 000 8h1" /><line x1="8" y1="12" x2="16" y2="12" /></svg>
          <span style={{ fontSize: 14, fontWeight: 600, color: "#18181B" }}>Connection Details</span>
          <span style={{ fontSize: 12, color: "#A1A1AA" }}>Use these with Claude, MCP, or any API client</span>
        </div>

        <div style={{ display: "flex", gap: 16 }}>
          {/* API Base URL */}
          <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 6 }}>
            <span style={{ fontSize: 12, fontWeight: 500, color: "#71717A" }}>API Base URL</span>
            <div style={{ display: "flex", alignItems: "center", gap: 8, background: "#F4F4F5", borderRadius: 8, padding: "10px 14px" }}>
              <span style={{ fontSize: 13, color: "#18181B", fontFamily: '"Fira Code", monospace', flex: 1, wordBreak: "break-all" }}>{serviceUrl || localApiUrl}</span>
              <CopyBtn id="url" text={serviceUrl || localApiUrl} copiedField={copiedField} setCopiedField={setCopiedField} />
            </div>
          </div>

          {/* Tenant ID */}
          <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 6 }}>
            <span style={{ fontSize: 12, fontWeight: 500, color: "#71717A" }}>Tenant ID</span>
            <div style={{ display: "flex", alignItems: "center", gap: 8, background: "#F4F4F5", borderRadius: 8, padding: "10px 14px" }}>
              {tenantId ? (
                <>
                  <span style={{ fontSize: 13, color: "#18181B", fontFamily: '"Fira Code", monospace', flex: 1, wordBreak: "break-all" }}>{tenantId}</span>
                  <CopyBtn id="tenant" text={tenantId} copiedField={copiedField} setCopiedField={setCopiedField} />
                </>
              ) : (
                <span style={{ fontSize: 13, color: "#A1A1AA", fontFamily: '"Fira Code", monospace', fontStyle: "italic" }}>Not assigned (local mode)</span>
              )}
            </div>
          </div>

          {/* User ID */}
          <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 6 }}>
            <span style={{ fontSize: 12, fontWeight: 500, color: "#71717A" }}>User ID</span>
            <div style={{ display: "flex", alignItems: "center", gap: 8, background: "#F4F4F5", borderRadius: 8, padding: "10px 14px" }}>
              {userId ? (
                <>
                  <span style={{ fontSize: 13, color: "#18181B", fontFamily: '"Fira Code", monospace', flex: 1, wordBreak: "break-all" }}>{userId}</span>
                  <CopyBtn id="user" text={userId} copiedField={copiedField} setCopiedField={setCopiedField} />
                </>
              ) : (
                <span style={{ fontSize: 13, color: "#A1A1AA", fontFamily: '"Fira Code", monospace' }}>...</span>
              )}
            </div>
          </div>
        </div>

        {/* Auth header hint */}
        <div style={{ display: "flex", alignItems: "center", gap: 8, background: "#18181B", borderRadius: 8, padding: "10px 16px" }}>
          <span style={{ fontSize: 12, color: "#A1A1AA", fontFamily: '"Fira Code", monospace', flex: 1 }}>X-Api-Key: {"<your-api-key>"}{tenantId ? `  •  X-Tenant-Id: ${tenantId}` : ""}</span>
          <CopyBtn id="header" text={`X-Api-Key: <your-api-key>${tenantId ? `\nX-Tenant-Id: ${tenantId}` : ""}`} copiedField={copiedField} setCopiedField={setCopiedField} light />
        </div>
      </div>

      {/* Documentation links */}
      <div style={{ display: "flex", gap: 12 }}>
        <a
          href={isDev ? "https://api.dev-aws.cognee.ai/docs" : "https://api.aws.cognee.ai/docs"}
          target="_blank"
          rel="noopener noreferrer"
          style={{ flex: 1, display: "flex", alignItems: "center", gap: 10, background: "#fff", border: "1px solid #E4E4E7", borderRadius: 10, padding: "14px 18px", textDecoration: "none", transition: "border-color 150ms" }}
          className="hover:border-cognee-purple"
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#6510F4" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0 }}><path d="M2 3h6a4 4 0 014 4v14a3 3 0 00-3-3H2z" /><path d="M22 3h-6a4 4 0 00-4 4v14a3 3 0 013-3h7z" /></svg>
          <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
            <span style={{ fontSize: 13, fontWeight: 500, color: "#18181B" }}>API Reference</span>
            <span style={{ fontSize: 12, color: "#A1A1AA" }}>Interactive Swagger docs for the shared API</span>
          </div>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#A1A1AA" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ marginLeft: "auto", flexShrink: 0 }}><path d="M18 13v6a2 2 0 01-2 2H5a2 2 0 01-2-2V8a2 2 0 012-2h6" /><polyline points="15 3 21 3 21 9" /><line x1="10" y1="14" x2="21" y2="3" /></svg>
        </a>
        {apiDocsUrl && (
          <a
            href={apiDocsUrl}
            target="_blank"
            rel="noopener noreferrer"
            style={{ flex: 1, display: "flex", alignItems: "center", gap: 10, background: "#fff", border: "1px solid #E4E4E7", borderRadius: 10, padding: "14px 18px", textDecoration: "none", transition: "border-color 150ms" }}
            className="hover:border-cognee-purple"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#6510F4" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0 }}><rect x="2" y="3" width="20" height="14" rx="2" ry="2" /><line x1="8" y1="21" x2="16" y2="21" /><line x1="12" y1="17" x2="12" y2="21" /></svg>
            <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
              <span style={{ fontSize: 13, fontWeight: 500, color: "#18181B" }}>API Tenant Reference</span>
              <span style={{ fontSize: 12, color: "#A1A1AA" }}>Swagger docs for your tenant instance</span>
            </div>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#A1A1AA" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ marginLeft: "auto", flexShrink: 0 }}><path d="M18 13v6a2 2 0 01-2 2H5a2 2 0 01-2-2V8a2 2 0 012-2h6" /><polyline points="15 3 21 3 21 9" /><line x1="10" y1="14" x2="21" y2="3" /></svg>
          </a>
        )}
      </div>

      {/* Info banner */}
      <div style={{ display: "flex", gap: 10, background: "#F0EDFF", border: "1px solid #DDD6FE", borderRadius: 8, padding: "14px 16px", alignItems: "flex-start" }}>
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#6510F4" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0, marginTop: 1 }}>
          <circle cx="12" cy="12" r="10" /><line x1="12" y1="16" x2="12" y2="12" /><line x1="12" y1="8" x2="12.01" y2="8" />
        </svg>
        <span style={{ fontSize: 13, color: "#52525B", lineHeight: "20px" }}>
          API keys grant full access to your account. Keep them secret — do not share keys in client-side code or public repositories. Use environment variables instead.
        </span>
      </div>

      {/* Table */}
      <div style={{ background: "#fff", border: "1px solid #E4E4E7", borderRadius: 12, overflow: "hidden" }}>
        {/* Header */}
        <div style={{ display: "flex", alignItems: "center", padding: "12px 24px", borderBottom: "1px solid #E4E4E7" }}>
          <span style={{ width: 200, fontSize: 13, fontWeight: 500, color: "#71717A" }}>Name</span>
          <span style={{ flex: 1, fontSize: 13, fontWeight: 500, color: "#71717A" }}>Key</span>
          <span style={{ width: 80 }} />
        </div>

        {/* Rows */}
        {keys.map((k) => (
          <div
            key={k.id}
            className="hover:bg-cognee-hover"
            style={{ display: "flex", alignItems: "center", padding: "14px 24px", borderBottom: "1px solid #F4F4F5", transition: "background 150ms" }}
          >
            <div style={{ width: 200, display: "flex", flexDirection: "column", gap: 2 }}>
              <span style={{ fontSize: 14, fontWeight: 500, color: "#18181B" }}>{k.name || "Unnamed"}</span>
            </div>
            <div style={{ flex: 1, display: "flex", alignItems: "center", gap: 8 }}>
              {k.isNew ? (
                /* Show full key once for newly created keys */
                <div style={{ display: "flex", alignItems: "center", gap: 8, background: "#F0FDF4", border: "1px solid #BBF7D0", borderRadius: 6, padding: "6px 12px" }}>
                  <span style={{ fontSize: 12, color: "#166534", fontFamily: '"Fira Code", monospace', wordBreak: "break-all" }}>{k.key}</span>
                  <button onClick={() => handleCopy(k.id, k.key)} className="cursor-pointer" style={{ background: "none", border: "none", padding: 2, display: "flex", flexShrink: 0 }} title="Copy key">
                    {copiedId === k.id ? <CheckIcon /> : <CopyIcon />}
                  </button>
                </div>
              ) : (
                <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                  <span style={{ fontSize: 13, color: "#52525B", fontFamily: '"Fira Code", monospace' }}>{k.label || k.key}</span>
                  <button onClick={() => handleCopy(k.id, k.key)} className="cursor-pointer" style={{ background: "none", border: "none", padding: 2, display: "flex" }} title="Copy key">
                    {copiedId === k.id ? <CheckIcon /> : <CopyIcon />}
                  </button>
                </div>
              )}
            </div>
            <div style={{ width: 80, display: "flex", justifyContent: "flex-end" }}>
              <button
                onClick={() => handleRevoke(k.id)}
                className="cursor-pointer hover:underline"
                style={{ background: "none", border: "none", fontSize: 13, color: "#EF4444", fontFamily: "inherit" }}
              >
                Revoke
              </button>
            </div>
          </div>
        ))}

        {keys.length === 0 && (
          <div style={{ padding: "48px 24px", textAlign: "center", display: "flex", flexDirection: "column", alignItems: "center", gap: 12 }}>
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#A1A1AA" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 2l-2 2m-7.61 7.61a5.5 5.5 0 1 1-7.778 7.778 5.5 5.5 0 0 1 7.777-7.777zm0 0L15.5 7.5m0 0l3 3L22 7l-3-3m-3.5 3.5L19 4" />
            </svg>
            <span style={{ fontSize: 14, color: "#71717A" }}>No API keys yet</span>
            <span style={{ fontSize: 13, color: "#A1A1AA" }}>Create one to connect agents or use the API programmatically.</span>
            <button
              onClick={() => setShowCreateModal(true)}
              className="cursor-pointer hover:bg-cognee-purple-hover"
              style={{ background: "#6510F4", color: "#fff", border: "none", borderRadius: 8, padding: "8px 20px", fontSize: 13, fontWeight: 500, marginTop: 4 }}
            >
              Create your first key
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

function CopyBtn({ id, text, copiedField, setCopiedField, light }: { id: string; text: string; copiedField: string | null; setCopiedField: (v: string | null) => void; light?: boolean }) {
  const isCopied = copiedField === id;
  return (
    <button
      onClick={(e) => { e.stopPropagation(); navigator.clipboard.writeText(text); setCopiedField(id); setTimeout(() => setCopiedField(null), 1500); }}
      className="cursor-pointer hover:opacity-80 rounded p-1 active:scale-90 transition-all"
      style={{ background: "none", border: "none", flexShrink: 0, display: "flex" }}
      title="Copy"
    >
      {isCopied ? (
        <svg width="14" height="14" viewBox="0 0 16 16" fill="none"><path d="M3.5 8.5L6.5 11.5L12.5 4.5" stroke="#22C55E" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" /></svg>
      ) : (
        <svg width="14" height="14" viewBox="0 0 16 16" fill="none"><rect x="5" y="5" width="8" height="8" rx="1.5" stroke={light ? "#71717A" : "#A1A1AA"} strokeWidth="1.5" /><path d="M11 3H4.5A1.5 1.5 0 003 4.5V11" stroke={light ? "#71717A" : "#A1A1AA"} strokeWidth="1.5" strokeLinecap="round" /></svg>
      )}
    </button>
  );
}
