"use client";

import { useState, useRef, useEffect } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { useCogniInstance } from "@/modules/tenant/TenantProvider";
import ServeOnboarding from "./ServeOnboarding";
import { LocalCogneeStep, AgentStep, DatabaseStep } from "./ConnectionSteps";
import addData from "@/modules/ingestion/addData";
import createDataset from "@/modules/datasets/createDataset";
import cognifyDataset from "@/modules/datasets/cognifyDataset";
import searchDataset from "@/modules/datasets/searchDataset";

// ── Icons ──

function UploadIcon() {
  return (
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
      <path d="M12 16V4M8 8l4-4 4 4" stroke="#6510F4" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M4 18h16" stroke="#6510F4" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  );
}

// ── Shared ──

function StepBadge({ step, total = 4 }: { step: number; total?: number }) {
  return (
    <div style={{ background: "#F0EDFF", borderRadius: 100, border: "1px solid #DDD6FE", padding: "5px 12px" }}>
      <span style={{ color: "#000", fontSize: 13, fontWeight: 500 }}>Step {step} of {total}</span>
    </div>
  );
}

function StepDots({ current, total = 4 }: { current: number; total?: number }) {
  return (
    <div className="flex items-center gap-2">
      {Array.from({ length: total }).map((_, i) => (
        <div key={i} style={{ width: 24, height: 4, borderRadius: 2, background: i + 1 === current ? "#6510F4" : "#DDD6FE" }} />
      ))}
    </div>
  );
}

function SkipLink() {
  const router = useRouter();
  return (
    <button onClick={() => { sessionStorage.setItem("cognee-onboarding-skipped", "1"); router.push("/dashboard"); }} className="cursor-pointer" style={{ background: "none", border: "none", color: "#9CA3AF", fontSize: 13, paddingTop: 32, paddingBottom: 24 }}>
      Skip onboarding and go to dashboard
    </button>
  );
}

// ── Step 1: Connect your data (Paper design) ──

function SourceRow({ icon, title, subtitle, onClick }: { icon: React.ReactNode; title: string; subtitle: string; onClick?: () => void }) {
  return (
    <div onClick={onClick} style={{ alignItems: "center", borderBottomColor: "#F3F4F6", borderBottomStyle: "solid", borderBottomWidth: 1, boxSizing: "border-box", display: "flex", gap: 14, paddingBlock: 16, cursor: onClick ? "pointer" : "default" }}>
      <div style={{ alignItems: "center", backgroundColor: "#F5F3FF", borderRadius: 8, display: "flex", flexShrink: 0, height: 36, justifyContent: "center", width: 36 }}>{icon}</div>
      <div style={{ display: "flex", flexBasis: "0%", flexDirection: "column", flexGrow: 1, flexShrink: 1 }}>
        <div style={{ color: "#111111", fontFamily: "system-ui, sans-serif", fontSize: 14, lineHeight: "18px" }}>{title}</div>
        <div style={{ color: "#9CA3AF", fontFamily: "system-ui, sans-serif", fontSize: 12, lineHeight: "16px" }}>{subtitle}</div>
      </div>
      <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#C4B5FD" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0 }}><polyline points="9 18 15 12 9 6" /></svg>
    </div>
  );
}

function Step1({ onNext, files, setFiles, cogniInstance, datasetId, setDatasetId }: {
  onNext: () => void;
  files: File[];
  setFiles: React.Dispatch<React.SetStateAction<File[]>>;
  cogniInstance: NonNullable<ReturnType<typeof useCogniInstance>["cogniInstance"]>;
  datasetId: string | null;
  setDatasetId: (id: string) => void;
}) {
  const router = useRouter();
  const [isDragging, setIsDragging] = useState(false);
  const [showPaste, setShowPaste] = useState(false);
  const [pasteText, setPasteText] = useState("");
  const [connectionView, setConnectionView] = useState<"local" | "agent" | "database" | "cloud" | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const datasetIdRef = useRef(datasetId);
  datasetIdRef.current = datasetId;

  const uploadFiles = async (newFiles: File[]) => {
    setIsUploading(true);
    setUploadError(null);
    try {
      let dsId = datasetIdRef.current;
      if (!dsId) {
        const ds = await createDataset({ name: "default_dataset" }, cogniInstance);
        dsId = ds.id as string;
        datasetIdRef.current = dsId;
        setDatasetId(dsId);
      }
      await addData({ id: dsId, name: "default_dataset" }, newFiles, cogniInstance);
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setIsUploading(false);
    }
  };

  const handleFiles = (newFiles: FileList | File[]) => {
    const fileArray = Array.from(newFiles);
    setFiles((prev) => [...prev, ...fileArray]);
    uploadFiles(fileArray);
  };

  const removeFile = (index: number) => setFiles((prev) => prev.filter((_, i) => i !== index));

  const handlePasteSubmit = () => {
    if (!pasteText.trim()) return;
    const blob = new Blob([pasteText], { type: "text/plain" });
    const file = new File([blob], "pasted-text.txt", { type: "text/plain" });
    setFiles((prev) => [...prev, file]);
    uploadFiles([file]);
    setPasteText("");
    setShowPaste(false);
  };

  // Show connection sub-step if selected
  if (connectionView === "local") return <LocalCogneeStep onBack={() => setConnectionView(null)} onSkip={() => { sessionStorage.setItem("cognee-onboarding-skipped", "1"); router.push("/dashboard"); }} />;
  if (connectionView === "agent") return <AgentStep onBack={() => setConnectionView(null)} onSkip={() => { sessionStorage.setItem("cognee-onboarding-skipped", "1"); router.push("/dashboard"); }} />;
  if (connectionView === "database") return <DatabaseStep onBack={() => setConnectionView(null)} onSkip={() => { sessionStorage.setItem("cognee-onboarding-skipped", "1"); router.push("/dashboard"); }} />;

  return (
    <div style={{ alignItems: "center", display: "flex", flexDirection: "column", justifyContent: "center", width: "100%", fontFamily: "system-ui, sans-serif" }}>
      <div style={{ alignItems: "center", display: "flex", flexDirection: "column", paddingBottom: 40, paddingLeft: 80, paddingRight: 80, paddingTop: 60, width: "100%" }}>

        {/* Header */}
        <div style={{ alignItems: "center", display: "flex", flexDirection: "column", gap: 8, paddingBottom: 40 }}>
          <div style={{ backgroundColor: "#FFFFFF", borderColor: "#E5E7EB", borderRadius: 100, borderStyle: "solid", borderWidth: 1, paddingBlock: 6, paddingInline: 16 }}>
            <div style={{ color: "#6B7280", fontSize: 13, lineHeight: "16px" }}>Step 1 of 4</div>
          </div>
          <div style={{ color: "#111111", fontSize: 28, lineHeight: "34px", paddingTop: 8 }}>Connect your data</div>
          <div style={{ color: "#6B7280", fontSize: 15, lineHeight: "18px" }}>Choose how to get your data into Cognee. You can always add or change sources later.</div>
        </div>

        {/* Two-column cards */}
        <div style={{ display: "flex", gap: 32, maxWidth: 880, width: "100%" }}>

          {/* Left card: Add new data */}
          <div style={{ backgroundColor: "#FFFFFF", borderColor: "#E5E7EB", borderRadius: 16, borderStyle: "solid", borderWidth: 1, display: "flex", flexBasis: "0%", flexDirection: "column", flexGrow: 1, flexShrink: 1, gap: 20, paddingBlock: 32, paddingInline: 32 }}>
            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              <div style={{ color: "#111111", fontSize: 17, lineHeight: "22px" }}>Add new data</div>
              <div style={{ color: "#9CA3AF", fontSize: 13, lineHeight: "16px" }}>Upload files or paste content directly</div>
            </div>

            {/* Hidden file input */}
            <input ref={fileInputRef} type="file" multiple accept=".pdf,.csv,.txt,.md,.json,.docx" className="hidden" onChange={(e) => { if (e.target.files) handleFiles(e.target.files); e.target.value = ""; }} />

            {/* Drop zone */}
            <div
              onClick={() => fileInputRef.current?.click()}
              onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={(e) => { e.preventDefault(); setIsDragging(false); if (e.dataTransfer.files.length) handleFiles(e.dataTransfer.files); }}
              style={{ alignItems: "center", backgroundColor: isDragging ? "#EDE9FE" : "#F5F3FF", borderColor: isDragging ? "#7C3AED" : "#D4D0F8", borderRadius: 12, borderStyle: "dashed", borderWidth: 2, cursor: "pointer", display: "flex", flexDirection: "column", flexShrink: 0, gap: 8, height: 200, justifyContent: "center", paddingBlock: 40, paddingInline: 20 }}
            >
              <div style={{ alignItems: "center", backgroundColor: "#FFFFFF", borderRadius: 10, display: "flex", flexShrink: 0, height: 40, justifyContent: "center", width: 40 }}>
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#7C3AED" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><polyline points="17 8 12 3 7 8" /><line x1="12" y1="3" x2="12" y2="15" /></svg>
              </div>
              <div style={{ color: "#7C3AED", fontSize: 14, lineHeight: "18px" }}>Drop files here or browse</div>
              <div style={{ color: "#9CA3AF", fontSize: 12, lineHeight: "16px" }}>PDF, CSV, TXT, Markdown, JSON</div>
            </div>

            {/* File list */}
            {files.length > 0 && (
              <div className="flex flex-col gap-2">
                {files.map((f, i) => (
                  <div key={`f-${i}`} className="flex items-center justify-between" style={{ background: "#F4F4F5", borderRadius: 8, padding: "8px 12px" }}>
                    <div className="flex items-center gap-2">
                      <span style={{ fontSize: 13, color: "#18181B" }}>{f.name}</span>
                      <span style={{ fontSize: 11, color: "#A1A1AA" }}>({(f.size / 1024).toFixed(0)} KB)</span>
                    </div>
                    <button onClick={() => removeFile(i)} className="cursor-pointer bg-transparent border-none p-1" style={{ color: "#A1A1AA", fontSize: 14 }}>&#10005;</button>
                  </div>
                ))}
              </div>
            )}

            {/* Upload status */}
            {isUploading && (
              <div className="flex items-center gap-2" style={{ fontSize: 13, color: "#6510F4" }}>
                <div style={{ width: 14, height: 14, borderRadius: "50%", border: "2px solid #E4E4E7", borderTopColor: "#6510F4", animation: "spin 1s linear infinite" }} />
                Uploading...
              </div>
            )}
            {uploadError && (
              <div style={{ background: "#FEF2F2", border: "1px solid #FECACA", borderRadius: 8, padding: "8px 12px", fontSize: 13, color: "#991B1B" }}>
                {uploadError}
              </div>
            )}
            {!isUploading && !uploadError && files.length > 0 && (
              <div style={{ fontSize: 13, color: "#22C55E" }}>Uploaded to default_dataset</div>
            )}

            {/* Paste text button / area */}
            {!showPaste ? (
              <div onClick={() => setShowPaste(true)} style={{ alignItems: "center", borderColor: "#E5E7EB", borderRadius: 10, borderStyle: "solid", borderWidth: 1, cursor: "pointer", display: "flex", gap: 8, paddingBlock: 12, paddingInline: 16 }}>
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#7C3AED" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="8" y="2" width="8" height="4" rx="1" ry="1" /><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2" /><line x1="12" y1="11" x2="12" y2="17" /><line x1="9" y1="14" x2="15" y2="14" /></svg>
                <div style={{ color: "#374151", flexShrink: 0, fontSize: 13, lineHeight: "16px" }}>Paste text</div>
              </div>
            ) : (
              <div className="flex flex-col gap-2">
                <textarea
                  autoFocus
                  value={pasteText}
                  onChange={(e) => setPasteText(e.target.value)}
                  placeholder="Paste your text content here..."
                  style={{ borderColor: "#D4D0F8", borderRadius: 10, borderStyle: "solid", borderWidth: 1, fontSize: 13, minHeight: 80, padding: 12, resize: "vertical", outline: "none" }}
                />
                <div className="flex gap-2">
                  <button onClick={handlePasteSubmit} className="cursor-pointer" style={{ background: "#7C3AED", border: "none", borderRadius: 8, color: "#fff", fontSize: 13, padding: "6px 16px" }}>Add text</button>
                  <button onClick={() => { setShowPaste(false); setPasteText(""); }} className="cursor-pointer" style={{ background: "none", border: "1px solid #E5E7EB", borderRadius: 8, color: "#6B7280", fontSize: 13, padding: "6px 16px" }}>Cancel</button>
                </div>
              </div>
            )}
          </div>

          {/* Right card: Connect a source */}
          <div style={{ backgroundColor: "#FFFFFF", borderColor: "#E5E7EB", borderRadius: 16, borderStyle: "solid", borderWidth: 1, display: "flex", flexBasis: "0%", flexDirection: "column", flexGrow: 1, flexShrink: 1, gap: 20, paddingBlock: 32, paddingInline: 32 }}>
            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              <div style={{ color: "#111111", fontSize: 17, lineHeight: "22px" }}>Connect a source</div>
              <div style={{ color: "#9CA3AF", fontSize: 13, lineHeight: "16px" }}>Link an existing system to sync data</div>
            </div>
            <div style={{ display: "flex", flexDirection: "column" }}>
              <SourceRow
                icon={<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#7C3AED" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="2" y="2" width="20" height="8" rx="2" ry="2" /><rect x="2" y="14" width="20" height="8" rx="2" ry="2" /><line x1="6" y1="6" x2="6.01" y2="6" /><line x1="6" y1="18" x2="6.01" y2="18" /></svg>}
                title="Local Cognee"
                subtitle="Sync your local instance"
                onClick={() => setConnectionView("local")}
              />
              <SourceRow
                icon={<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#7C3AED" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="11" width="18" height="10" rx="2" /><circle cx="12" cy="5" r="2" /><path d="M12 7v4" /><line x1="8" y1="16" x2="8" y2="16" /><line x1="16" y1="16" x2="16" y2="16" /></svg>}
                title="Agent"
                subtitle="OpenAI, OpenClaw, and more"
                onClick={() => setConnectionView("agent")}
              />
              <SourceRow
                icon={<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#7C3AED" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3" /><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3" /><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5" /></svg>}
                title="Database"
                subtitle="Postgres, MySQL, and more"
                onClick={() => setConnectionView("database")}
              />
              <SourceRow
                icon={<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#7C3AED" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M18 10h-1.26A8 8 0 1 0 9 20h9a5 5 0 0 0 0-10z" /></svg>}
                title="Cloud storage"
                subtitle="S3, GCS, Azure Blob, and more"
                onClick={() => router.push("/connections")}
              />
            </div>
          </div>
        </div>

        {/* Continue button when files selected */}
        {files.length > 0 && (
          <div style={{ paddingTop: 24 }}>
            <button onClick={onNext} className="cursor-pointer" style={{ background: "#6510F4", border: "none", borderRadius: 8, padding: "10px 32px", fontSize: 14, fontWeight: 500, color: "#fff" }}>
              Continue with {files.length} file{files.length !== 1 ? "s" : ""}
            </button>
          </div>
        )}

        <SkipLink />
      </div>
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}

// ── Step 2: Build memory (upload + cognify) ──

interface ProcessingStep {
  label: string;
  progress: number;
  status: "pending" | "active" | "done" | "error";
}

function Step2({ files, datasetId, onNext, cogniInstance }: {
  files: File[];
  datasetId: string | null;
  onNext: (dsId: string) => void;
  cogniInstance: NonNullable<ReturnType<typeof useCogniInstance>["cogniInstance"]>;
}) {
  const [steps, setSteps] = useState<ProcessingStep[]>([
    { label: "Uploading files", progress: 0, status: "active" },
    { label: "Extracting entities", progress: 0, status: "pending" },
    { label: "Building knowledge graph", progress: 0, status: "pending" },
  ]);
  const [error, setError] = useState<string | null>(null);
  const [dsId, setDsId] = useState<string | null>(datasetId);
  const started = useRef(false);

  const allDone = steps.every((s) => s.status === "done");

  useEffect(() => {
    if (started.current) return;
    started.current = true;
    runPipeline();
  }, []);

  async function runPipeline() {
    try {
      let currentDsId = dsId;

      if (!currentDsId) {
        const ds = await createDataset({ name: "default_dataset" }, cogniInstance);
        currentDsId = ds.id;
        setDsId(currentDsId);
      }

      // Files already uploaded in Step 1 — animate quickly as confirmation
      updateStep(0, { status: "active", progress: 30 });
      await new Promise((r) => setTimeout(r, 400));
      updateStep(0, { progress: 70 });
      await new Promise((r) => setTimeout(r, 400));
      updateStep(0, { progress: 100, status: "done" });

      // Step 2: Cognify (extracting + building)
      updateStep(1, { status: "active", progress: 20 });

      // Simulate progress while cognify runs
      const progressInterval = setInterval(() => {
        setSteps((prev) => prev.map((s, i) => {
          if (i === 1 && s.status === "active" && s.progress < 90) return { ...s, progress: s.progress + 5 };
          if (i === 2 && s.status === "active" && s.progress < 90) return { ...s, progress: s.progress + 3 };
          return s;
        }));
      }, 500);

      await cognifyDataset({ id: currentDsId!, name: "default_dataset", data: [], status: "ready" }, cogniInstance);

      clearInterval(progressInterval);
      updateStep(1, { progress: 100, status: "done" });
      updateStep(2, { progress: 100, status: "done" });

    } catch (err) {
      setError(err instanceof Error ? err.message : "Processing failed");
      setSteps((prev) => prev.map((s) => s.status === "active" ? { ...s, status: "error" } : s));
    }
  }

  function updateStep(index: number, update: Partial<ProcessingStep>) {
    setSteps((prev) => prev.map((s, i) => i === index ? { ...s, ...update } : s));
  }

  return (
    <div className="flex flex-col items-center justify-center gap-6 flex-1" style={{ padding: 48, fontFamily: '"Inter", system-ui, sans-serif' }}>
      <StepBadge step={2} />
      <h1 style={{ fontSize: 28, fontWeight: 600, color: "#18181B", margin: 0 }}>Building your memory</h1>
      <p style={{ fontSize: 15, color: "#71717A", margin: 0, textAlign: "center", maxWidth: 480, lineHeight: "22px" }}>
        Cognee is extracting entities, building relationships, and generating embeddings.
      </p>

      <div style={{ width: 480, background: "#fff", border: "1px solid #E4E4E7", borderRadius: 12, padding: 24 }}>
        <div className="flex flex-col gap-5">
          {steps.map((step, i) => (
            <div key={i} className="flex items-center gap-3">
              {step.status === "done" ? (
                <div className="flex items-center justify-center flex-shrink-0 rounded-full" style={{ width: 32, height: 32, background: "#22C55E" }}>
                  <svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M3.5 8.5L6.5 11.5L12.5 4.5" stroke="#fff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" /></svg>
                </div>
              ) : step.status === "error" ? (
                <div className="flex items-center justify-center flex-shrink-0 rounded-full" style={{ width: 32, height: 32, background: "#EF4444" }}>
                  <span style={{ color: "#fff", fontSize: 13, fontWeight: 700 }}>!</span>
                </div>
              ) : step.status === "active" ? (
                <div className="flex items-center justify-center flex-shrink-0 rounded-full" style={{ width: 32, height: 32, background: "#6510F4" }}>
                  <span style={{ color: "#fff", fontSize: 13, fontWeight: 700 }}>{i + 1}</span>
                </div>
              ) : (
                <div className="flex items-center justify-center flex-shrink-0 rounded-full" style={{ width: 32, height: 32, border: "1.5px solid #D4D4D8" }}>
                  <span style={{ color: "#A1A1AA", fontSize: 13, fontWeight: 700 }}>{i + 1}</span>
                </div>
              )}
              <div className="flex-1 flex flex-col gap-1">
                <div className="flex justify-between">
                  <span style={{ fontSize: 14, fontWeight: 500, color: step.status === "pending" ? "#A1A1AA" : step.status === "error" ? "#EF4444" : "#18181B" }}>{step.label}</span>
                  {step.status === "done" && <span style={{ fontSize: 12, fontWeight: 500, color: "#22C55E" }}>Done</span>}
                  {step.status === "active" && <span style={{ fontSize: 12, fontWeight: 500, color: "#6510F4" }}>{step.progress}%</span>}
                  {step.status === "error" && <span style={{ fontSize: 12, fontWeight: 500, color: "#EF4444" }}>Failed</span>}
                </div>
                <div style={{ height: 4, borderRadius: 2, background: "#E4E4E7" }}>
                  <div style={{ height: 4, borderRadius: 2, background: step.status === "done" ? "#22C55E" : step.status === "active" ? "#6510F4" : step.status === "error" ? "#EF4444" : "transparent", width: `${step.progress}%`, transition: "width 0.3s" }} />
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {error && (
        <div style={{ background: "#FEF2F2", border: "1px solid #FECACA", borderRadius: 8, padding: "10px 16px", fontSize: 13, color: "#991B1B", maxWidth: 480 }}>
          {error}
        </div>
      )}

      <StepDots current={2} />

      {allDone && (
        <button onClick={() => onNext(dsId!)} className="cursor-pointer" style={{ background: "#6510F4", border: "none", borderRadius: 8, padding: "10px 24px", fontSize: 14, fontWeight: 500, color: "#fff" }}>
          Continue
        </button>
      )}

      <SkipLink />
    </div>
  );
}

// ── Step 3: Search ──

function Step3({ datasetId, onNext, cogniInstance }: {
  datasetId: string;
  onNext: () => void;
  cogniInstance: NonNullable<ReturnType<typeof useCogniInstance>["cogniInstance"]>;
}) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<string[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const suggestions = ["What entities are in my dataset?", "Summarize key findings", "Show relationships between concepts"];

  const handleSearch = async (q: string) => {
    if (!q.trim()) return;
    setQuery(q);
    setIsSearching(true);
    setResults([]);
    try {
      const data = await searchDataset(cogniInstance, { query: q, searchType: "GRAPH_COMPLETION", datasetIds: [datasetId] });
      const texts: string[] = [];
      if (Array.isArray(data)) {
        for (const item of data) {
          if (Array.isArray(item.search_result)) {
            texts.push(...item.search_result);
          }
        }
      }
      setResults(texts.length > 0 ? texts : ["No results found for this query."]);
    } catch {
      setResults(["Search failed. Try again."]);
    } finally {
      setIsSearching(false);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center gap-6 flex-1" style={{ padding: 48, fontFamily: '"Inter", system-ui, sans-serif' }}>
      <StepBadge step={3} />
      <h1 style={{ fontSize: 28, fontWeight: 600, color: "#18181B", margin: 0 }}>Search your data</h1>
      <p style={{ fontSize: 15, color: "#71717A", margin: 0, textAlign: "center", maxWidth: 480, lineHeight: "22px" }}>
        Ask questions in natural language and Cognee will search your knowledge graph.
      </p>

      <div className="flex flex-col gap-4" style={{ width: 560 }}>
        <div className="flex items-center gap-[10px] bg-white" style={{ border: `2px solid ${query ? "#6510F4" : "#E4E4E7"}`, borderRadius: 10, padding: "10px 16px", transition: "border-color 0.2s" }}>
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#A1A1AA" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><circle cx="11" cy="11" r="8" /><line x1="21" y1="21" x2="16.65" y2="16.65" /></svg>
          <input
            ref={inputRef}
            type="text" value={query} onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter") handleSearch(query); }}
            placeholder="Ask a question about your data..."
            style={{ flex: 1, border: "none", outline: "none", fontSize: 14, color: "#18181B", fontFamily: "inherit", background: "transparent" }}
          />
          {query && (
            <button onClick={() => handleSearch(query)} className="cursor-pointer" style={{ background: "#6510F4", border: "none", borderRadius: 6, padding: "4px 12px", fontSize: 12, fontWeight: 500, color: "#fff" }}>Search</button>
          )}
        </div>

        {!results.length && !isSearching && (
          <div className="flex flex-wrap gap-2 justify-center">
            {suggestions.map((s) => (
              <div key={s} onClick={() => { setQuery(s); handleSearch(s); }} className="cursor-pointer hover:bg-[#F0EDFF] transition-colors" style={{ background: "#FAFAF9", border: "1px solid #DDD6FE", borderRadius: 100, padding: "6px 14px", fontSize: 13, color: "#000" }}>
                {s}
              </div>
            ))}
          </div>
        )}

        {isSearching && (
          <div className="flex items-center justify-center gap-3 py-6">
            <div style={{ width: 16, height: 16, borderRadius: "50%", border: "2px solid #E4E4E7", borderTopColor: "#6510F4", animation: "spin 1s linear infinite" }} />
            <span style={{ fontSize: 13, color: "#71717A" }}>Searching knowledge graph...</span>
          </div>
        )}

        {results.length > 0 && !isSearching && (
          <div className="flex flex-col gap-3">
            {results.map((r, i) => (
              <div key={i} className="flex items-start gap-3" style={{ background: "#fff", border: "1px solid #E4E4E7", borderRadius: 10, padding: "14px 16px" }}>
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none" className="flex-shrink-0 mt-0.5"><path d="M3.5 8.5L6.5 11.5L12.5 4.5" stroke="#6510F4" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" /></svg>
                <span style={{ fontSize: 13, color: "#18181B", lineHeight: "18px" }}>{r}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      <StepDots current={3} />

      {results.length > 0 && (
        <button onClick={onNext} className="cursor-pointer" style={{ background: "#6510F4", border: "none", borderRadius: 8, padding: "10px 24px", fontSize: 14, fontWeight: 500, color: "#fff" }}>Continue</button>
      )}

      <SkipLink />
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}

// ── Step 4: Done ──

function Step4() {
  const router = useRouter();
  return (
    <div className="flex flex-col items-center justify-center gap-6 flex-1" style={{ padding: 48, fontFamily: '"Inter", system-ui, sans-serif' }}>
      <StepBadge step={4} />
      <h1 style={{ fontSize: 28, fontWeight: 600, color: "#18181B", margin: 0 }}>You&apos;re all set!</h1>
      <p style={{ fontSize: 15, color: "#71717A", margin: 0, textAlign: "center", maxWidth: 480, lineHeight: "22px" }}>
        Your knowledge graph is built and searchable. Explore your data, add more documents, or connect agents.
      </p>

      <div className="flex gap-3">
        <button onClick={() => router.push("/dashboard")} className="cursor-pointer" style={{ background: "#6510F4", color: "#fff", borderRadius: 8, padding: "10px 24px", fontSize: 14, fontWeight: 500, border: "none" }}>
          Go to dashboard
        </button>
        <button onClick={() => router.push("/datasets")} className="cursor-pointer bg-white" style={{ border: "1px solid #E4E4E7", borderRadius: 8, padding: "10px 24px", fontSize: 14, fontWeight: 500, color: "#000" }}>
          View datasets
        </button>
      </div>

      <StepDots current={4} />
    </div>
  );
}

// ── Main ──

export default function OnboardingPage() {
  const { cogniInstance, isInitializing } = useCogniInstance();
  const searchParams = useSearchParams();
  const [step, setStep] = useState(1);
  const [files, setFiles] = useState<File[]>([]);
  const [datasetId, setDatasetId] = useState<string | null>(null);

  const isServeMode = searchParams.get("source") === "serve";

  if (isInitializing || !cogniInstance) {
    return (
      <div className="flex items-center justify-center h-screen" style={{ fontFamily: '"Inter", system-ui, sans-serif' }}>
        <span style={{ fontSize: 14, color: "#71717A" }}>Connecting...</span>
      </div>
    );
  }

  if (isServeMode) {
    return <ServeOnboarding />;
  }

  return (
    <div className="flex flex-col h-full overflow-auto" style={{ background: "#FAFAF9" }}>
      {step === 1 && <Step1 files={files} setFiles={setFiles} onNext={() => setStep(2)} cogniInstance={cogniInstance} datasetId={datasetId} setDatasetId={setDatasetId} />}
      {step === 2 && <Step2 files={files} datasetId={datasetId} onNext={(id) => { setDatasetId(id); setStep(3); }} cogniInstance={cogniInstance} />}
      {step === 3 && datasetId && <Step3 datasetId={datasetId} onNext={() => setStep(4)} cogniInstance={cogniInstance} />}
      {step === 4 && <Step4 />}
    </div>
  );
}
