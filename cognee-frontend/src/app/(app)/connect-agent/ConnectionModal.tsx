"use client";

import { useState, useEffect } from "react";
import { useCogniInstance } from "@/modules/tenant/TenantProvider";
import getOrCreateApiKey from "@/modules/apiKeys/getOrCreateApiKey";
import { TERMINAL_EXPORT, fillTemplate } from "./prompts";

const localApiUrl = process.env.NEXT_PUBLIC_LOCAL_API_URL || "http://localhost:8000";

function CopyRow({ label, text, loading }: { label: string; text: string; loading?: boolean }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    if (loading) return;
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <button
      onClick={handleCopy}
      className="cursor-pointer"
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        width: "100%",
        background: "#18181B",
        borderRadius: 8,
        padding: "12px 16px",
        border: "none",
        fontFamily: "inherit",
        cursor: loading ? "wait" : "pointer",
      }}
    >
      <span style={{ fontSize: 13, color: "#A1A1AA", fontFamily: '"Fira Code", monospace' }}>
        {loading ? "Loading..." : label}
      </span>
      {copied ? (
        <svg width="14" height="14" viewBox="0 0 16 16" fill="none"><path d="M3.5 8.5L6.5 11.5L12.5 4.5" stroke="#22C55E" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" /></svg>
      ) : (
        <svg width="14" height="14" viewBox="0 0 16 16" fill="none"><rect x="5" y="5" width="8" height="8" rx="1.5" stroke="#71717A" strokeWidth="1.5" /><path d="M11 3H4.5A1.5 1.5 0 003 4.5V11" stroke="#71717A" strokeWidth="1.5" strokeLinecap="round" /></svg>
      )}
    </button>
  );
}

export interface ModalStep {
  title: string;
  description?: string;
  copyLabel: string;
  /** Template string — {{BASE_URL}} and {{API_KEY}} will be replaced */
  copyText: string;
  previewLabel?: string;
}

interface ConnectionModalProps {
  title: string;
  logoSrc: string;
  steps: ModalStep[];
  onClose: () => void;
}

export default function ConnectionModal({ title, logoSrc, steps, onClose }: ConnectionModalProps) {
  const { serviceUrl } = useCogniInstance();
  const [apiKey, setApiKey] = useState<string>("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getOrCreateApiKey()
      .then(setApiKey)
      .catch(() => setApiKey("your-api-key"))
      .finally(() => setLoading(false));
  }, []);

  const baseUrl = serviceUrl || localApiUrl;

  return (
    <div
      style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.3)", zIndex: 100, display: "flex", alignItems: "center", justifyContent: "center" }}
      onClick={onClose}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{ background: "#fff", borderRadius: 12, width: 520, maxHeight: "80vh", overflow: "auto", boxShadow: "0 16px 48px rgba(0,0,0,0.12)", fontFamily: '"Inter", system-ui, sans-serif' }}
      >
        {/* Header */}
        <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "16px 20px", borderBottom: "1px solid #E4E4E7" }}>
          <img src={logoSrc} alt="" style={{ height: 22, width: "auto" }} />
          <span style={{ fontSize: 14, fontWeight: 600, color: "#18181B" }}>{title}</span>
          <button onClick={onClose} className="cursor-pointer" style={{ marginLeft: "auto", background: "none", border: "none", color: "#A1A1AA", fontSize: 16, padding: 2, fontFamily: "inherit" }}>&#10005;</button>
        </div>

        {/* Steps */}
        <div style={{ padding: "16px 20px", display: "flex", flexDirection: "column", gap: 16 }}>
          {steps.map((step, i) => {
            const text = fillTemplate(step.copyText, baseUrl, apiKey || "your-api-key");
            return (
              <div key={i}>
                {i > 0 && <div style={{ height: 1, background: "#E4E4E7", marginBottom: 16 }} />}

                <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10 }}>
                  <div style={{ width: 24, height: 24, borderRadius: "50%", background: "#F0EDFF", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                    <span style={{ color: "#6510F4", fontSize: 12, fontWeight: 600 }}>{i + 1}</span>
                  </div>
                  <span style={{ fontSize: 13, fontWeight: 500, color: "#18181B" }}>{step.title}</span>
                </div>

                {step.description && (
                  <p style={{ margin: "0 0 10px", fontSize: 12, color: "#71717A", lineHeight: 1.5 }}>{step.description}</p>
                )}

                <CopyRow label={step.copyLabel} text={text} loading={loading} />

                {step.previewLabel && (
                  <details style={{ fontSize: 12, color: "#71717A", marginTop: 8 }}>
                    <summary className="cursor-pointer" style={{ fontSize: 12, color: "#6510F4", fontWeight: 500 }}>{step.previewLabel}</summary>
                    <pre style={{
                      marginTop: 8, padding: 12, background: "#FAFAFA", border: "1px solid #E4E4E7",
                      borderRadius: 8, fontSize: 11, lineHeight: 1.5, color: "#3F3F46", overflow: "auto",
                      maxHeight: 200, whiteSpace: "pre-wrap", wordBreak: "break-all", fontFamily: '"Fira Code", monospace',
                    }}>
                      {loading ? "Loading..." : text}
                    </pre>
                  </details>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
