"use client";

import type { ReactElement } from "react";
import { Loader } from "@mantine/core";
import ModalShell from "@/ui/elements/ModalShell";

export default function CreatePromptModal({
  inferringPrompt,
  modelName,
  onGenerate,
  onBlank,
  onCancel,
}: {
  inferringPrompt: boolean;
  // Name of the currently-selected graph model, or null when none is selected.
  modelName: string | null;
  onGenerate: () => void;
  onBlank: () => void;
  onCancel: () => void;
}): ReactElement {
  return (
    <ModalShell width={440} onClose={() => { if (!inferringPrompt) onCancel(); }}>
      <h2 style={{ fontSize: 18, fontWeight: 700, color: "#EDECEA", margin: 0 }}>Create Custom Prompt</h2>
      {inferringPrompt ? (
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 12, padding: "24px 0" }}>
          <Loader size={24} color="#6510F4" />
          <span style={{ fontSize: 14, color: "#6510F4", fontWeight: 500 }}>Generating prompt from &ldquo;{modelName ?? "graph model"}&rdquo;...</span>
          <span style={{ fontSize: 12, color: "rgba(237,236,234,0.55)" }}>This may take a moment</span>
        </div>
      ) : (
        <>
          <p style={{ fontSize: 13, color: "rgba(237,236,234,0.55)", margin: 0, lineHeight: "20px" }}>
            A custom prompt guides how Cognee extracts entities and relationships from your data.
          </p>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            <button
              onClick={onGenerate}
              disabled={!modelName}
              className="cursor-pointer hover:bg-white/10"
              style={{ display: "flex", alignItems: "center", gap: 12, background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 8, padding: "14px 16px", textAlign: "left", fontFamily: "inherit", opacity: !modelName ? 0.5 : 1 }}
            >
              <div style={{ width: 36, height: 36, background: "rgba(188,155,255,0.20)", border: "1px solid rgba(188,155,255,0.35)", borderRadius: 8, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#6510F4" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10" /><path d="M9.09 9a3 3 0 015.83 1c0 2-3 3-3 3" /><line x1="12" y1="17" x2="12.01" y2="17" /></svg>
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
                <span style={{ fontSize: 14, fontWeight: 500, color: "#EDECEA" }}>Generate from graph model</span>
                <span style={{ fontSize: 12, color: "rgba(237,236,234,0.55)" }}>
                  {modelName ? `Using "${modelName}"` : "Select a graph model first"}
                </span>
              </div>
            </button>
            <button
              onClick={onBlank}
              className="cursor-pointer hover:bg-white/10"
              style={{ display: "flex", alignItems: "center", gap: 12, background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 8, padding: "14px 16px", textAlign: "left", fontFamily: "inherit" }}
            >
              <div style={{ width: 36, height: 36, background: "rgba(255,255,255,0.06)", borderRadius: 8, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="rgba(237,236,234,0.55)" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><line x1="12" y1="5" x2="12" y2="19" /><line x1="5" y1="12" x2="19" y2="12" /></svg>
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
                <span style={{ fontSize: 14, fontWeight: 500, color: "#EDECEA" }}>Start blank</span>
                <span style={{ fontSize: 12, color: "rgba(237,236,234,0.55)" }}>Write your own extraction prompt</span>
              </div>
            </button>
          </div>
          <button
            onClick={onCancel}
            className="cursor-pointer"
            style={{ background: "none", border: "none", fontSize: 13, color: "rgba(237,236,234,0.55)", fontFamily: "inherit", padding: "4px 0" }}
          >
            Cancel
          </button>
        </>
      )}
    </ModalShell>
  );
}
