"use client";

import { useState, useRef } from "react";
import { trackEvent } from "@/modules/analytics";
import { StepDots, SkipLink } from "./Shared";

export function Step1({ onNext, files, setFiles }: {
  onNext: () => void;
  files: File[];
  setFiles: React.Dispatch<React.SetStateAction<File[]>>;
}) {
  const [isDragging, setIsDragging] = useState(false);
  const [showPaste, setShowPaste] = useState(false);
  const [pasteText, setPasteText] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Files are only collected here; the actual upload + processing happens
  // in Step2 as a single remember call.
  const handleFiles = (newFiles: FileList | File[]) => {
    const fileArray = Array.from(newFiles);
    setFiles((prev) => [...prev, ...fileArray]);
    trackEvent({ pageName: "Onboarding", eventName: "onboarding_files_added", additionalProperties: { file_count: String(fileArray.length), step: "1" } });
  };

  const removeFile = (index: number) => setFiles((prev) => prev.filter((_, i) => i !== index));

  const handlePasteSubmit = () => {
    if (!pasteText.trim()) return;
    trackEvent({ pageName: "Onboarding", eventName: "onboarding_text_pasted", additionalProperties: { text_length: String(pasteText.length), step: "1" } });
    const blob = new Blob([pasteText], { type: "text/plain" });
    const file = new File([blob], "pasted-text.txt", { type: "text/plain" });
    setFiles((prev) => [...prev, file]);
    setPasteText("");
    setShowPaste(false);
  };

  return (
    <div style={{
      minHeight: "100vh",
      backgroundColor: "#000000",
      backgroundImage: "linear-gradient(rgba(244,244,244,0.10) 1px, transparent 1px), linear-gradient(90deg, rgba(244,244,244,0.10) 1px, transparent 1px)",
      backgroundSize: "33px 33px",
      display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
      boxSizing: "border-box",
    }}>
      <div style={{ alignItems: "center", display: "flex", flexDirection: "column", paddingBottom: 40, paddingLeft: "clamp(16px, 5vw, 80px)", paddingRight: "clamp(16px, 5vw, 80px)", paddingTop: 48, width: "100%", boxSizing: "border-box" }}>

        {/* Header */}
        <div style={{ alignItems: "center", display: "flex", flexDirection: "column", gap: 8, paddingBottom: 40 }}>
          <div style={{ background: "rgba(188,155,255,0.20)", borderRadius: 100, border: "1px solid rgba(188,155,255,0.35)", padding: "5px 12px" }}>
            <div style={{ color: "#EDECEA", fontSize: 13, lineHeight: "16px" }}>Step 1 of 3</div>
          </div>
          <div style={{ color: "#EDECEA", fontSize: 28, fontWeight: 300, lineHeight: "34px", paddingTop: 8, letterSpacing: "-0.02em", fontFamily: '"TWKLausanne", sans-serif' }}>Connect your data</div>
          <div style={{ color: "rgba(237,236,234,0.65)", fontSize: 15, lineHeight: "22px", textAlign: "center", maxWidth: 440 }}>Choose how to get your data into Cognee. You can always add or change sources later.</div>
        </div>

        {/* Card: Add new data — solid #2a2a2e to match the rest of onboarding */}
        <div style={{ maxWidth: 880, width: "100%", background: "#2a2a2e", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 16, boxShadow: "0 20px 60px rgba(0,0,0,0.5)", display: "flex", flexDirection: "column", gap: 20, padding: "48px 64px" }}>
          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            <div style={{ color: "#EDECEA", fontSize: 17, lineHeight: "22px" }}>Add new data</div>
            <div style={{ color: "rgba(237,236,234,0.65)", fontSize: 13, lineHeight: "16px" }}>Upload files or paste content directly</div>
          </div>

          {/* Hidden file input */}
          <input ref={fileInputRef} type="file" multiple accept=".pdf,.csv,.txt,.md,.json,.docx" className="hidden" onChange={(e) => { if (e.target.files) handleFiles(e.target.files); e.target.value = ""; }} />

          {/* Drop zone */}
          <div
            onClick={() => fileInputRef.current?.click()}
            onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={(e) => { e.preventDefault(); setIsDragging(false); if (e.dataTransfer.files.length) handleFiles(e.dataTransfer.files); }}
            style={{ alignItems: "center", background: isDragging ? "rgba(188,155,255,0.20)" : "rgba(255,255,255,0.04)", border: `2px dashed ${isDragging ? "#BC9BFF" : "rgba(255,255,255,0.18)"}`, borderRadius: 12, cursor: "pointer", display: "flex", flexDirection: "column", flexShrink: 0, gap: 8, height: 200, justifyContent: "center", paddingBlock: 40, paddingInline: 20, transition: "background 200ms, border-color 200ms" }}
          >
            <div style={{ alignItems: "center", background: "rgba(255,255,255,0.1)", borderRadius: 10, display: "flex", flexShrink: 0, height: 40, justifyContent: "center", width: 40 }}>
              <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#BC9BFF" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><polyline points="17 8 12 3 7 8" /><line x1="12" y1="3" x2="12" y2="15" /></svg>
            </div>
            <div style={{ color: "#BC9BFF", fontSize: 14, lineHeight: "18px" }}>Drop files here or browse</div>
            <div style={{ color: "rgba(237,236,234,0.4)", fontSize: 12, lineHeight: "16px" }}>PDF, CSV, TXT, Markdown, JSON</div>
          </div>

          {/* File list */}
          {files.length > 0 && (
            <div className="flex flex-col gap-2">
              {files.map((f, i) => (
                <div key={`f-${i}`} className="flex items-center justify-between" style={{ background: "rgba(255,255,255,0.07)", borderRadius: 8, padding: "8px 12px" }}>
                  <div className="flex items-center gap-2">
                    <span style={{ fontSize: 13, color: "#EDECEA" }}>{f.name}</span>
                    <span style={{ fontSize: 11, color: "rgba(237,236,234,0.4)" }}>({(f.size / 1024).toFixed(0)} KB)</span>
                  </div>
                  <button onClick={() => removeFile(i)} className="cursor-pointer bg-transparent border-none p-1" style={{ color: "rgba(237,236,234,0.4)", fontSize: 14 }}>&#10005;</button>
                </div>
              ))}
            </div>
          )}

          {files.length > 0 && (
            <div style={{ fontSize: 13, color: "#22C55E" }}>{files.length} file{files.length !== 1 ? "s" : ""} ready to process</div>
          )}

          {/* Paste text button / area */}
          {!showPaste ? (
            <div onClick={() => setShowPaste(true)} style={{ alignItems: "center", border: "1px solid rgba(255,255,255,0.12)", borderRadius: 10, cursor: "pointer", display: "flex", gap: 8, paddingBlock: 12, paddingInline: 16 }}>
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#BC9BFF" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="8" y="2" width="8" height="4" rx="1" ry="1" /><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2" /><line x1="12" y1="11" x2="12" y2="17" /><line x1="9" y1="14" x2="15" y2="14" /></svg>
              <div style={{ color: "rgba(237,236,234,0.7)", flexShrink: 0, fontSize: 13, lineHeight: "16px" }}>Paste text</div>
            </div>
          ) : (
            <div className="flex flex-col gap-2">
              <textarea
                autoFocus
                value={pasteText}
                onChange={(e) => setPasteText(e.target.value)}
                placeholder="Paste your text content here..."
                style={{ background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.15)", borderRadius: 10, color: "#EDECEA", fontSize: 13, minHeight: 80, padding: 12, resize: "vertical", outline: "none" }}
              />
              <div className="flex gap-2">
                <button onClick={handlePasteSubmit} className="cursor-pointer" style={{ background: "#BC9BFF", border: "none", borderRadius: 8, color: "#1e1e1c", fontSize: 13, padding: "6px 16px" }}>Add text</button>
                <button onClick={() => { setShowPaste(false); setPasteText(""); }} className="cursor-pointer" style={{ background: "none", border: "1px solid rgba(255,255,255,0.12)", borderRadius: 8, color: "rgba(237,236,234,0.6)", fontSize: 13, padding: "6px 16px" }}>Cancel</button>
              </div>
            </div>
          )}
        </div>

        {/* Continue button when files selected */}
        {files.length > 0 && (
          <div style={{ paddingTop: 24 }}>
            <button onClick={() => { trackEvent({ pageName: "Onboarding", eventName: "onboarding_step_completed", additionalProperties: { step: "1", file_count: String(files.length) } }); onNext(); }} className="cursor-pointer" style={{ background: "#BC9BFF", border: "none", borderRadius: 8, padding: "11px 32px", fontSize: 14, fontWeight: 500, color: "#1e1e1c", letterSpacing: "-0.01em" }}>
              Continue with {files.length} file{files.length !== 1 ? "s" : ""} →
            </button>
          </div>
        )}

        <div style={{ marginTop: 24 }}>
          <StepDots current={1} total={3} />
        </div>
        <SkipLink />
      </div>
    </div>
  );
}
