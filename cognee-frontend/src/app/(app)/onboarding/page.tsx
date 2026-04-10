"use client";

import { useState, useRef, useEffect } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { useCogniInstance } from "@/modules/tenant/TenantProvider";
import ServeOnboarding from "./ServeOnboarding";
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
    <button onClick={() => router.push("/dashboard")} className="cursor-pointer" style={{ background: "none", border: "none", color: "#71717A", fontSize: 13 }}>
      Skip onboarding and go to dashboard
    </button>
  );
}

// ── Step 1: Upload data ──

function Step1({ onNext, files, setFiles }: {
  onNext: () => void;
  files: File[];
  setFiles: React.Dispatch<React.SetStateAction<File[]>>;
}) {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFiles = (newFiles: FileList | File[]) => {
    setFiles((prev) => [...prev, ...Array.from(newFiles)]);
  };

  const removeFile = (index: number) => setFiles((prev) => prev.filter((_, i) => i !== index));

  return (
    <div
      className="flex flex-col items-center gap-8 flex-1"
      style={{ paddingTop: 48, paddingBottom: 48, paddingInline: 80, fontFamily: '"Inter", system-ui, sans-serif' }}
    >
      <div className="flex flex-col items-center gap-2">
        <StepBadge step={1} />
        <h1 style={{ fontSize: 28, fontWeight: 600, color: "#18181B", margin: 0 }}>Connect your data</h1>
        <p style={{ fontSize: 15, color: "#71717A", margin: 0, textAlign: "center", lineHeight: "22px" }}>
          Cognee will extract entities, build a knowledge graph, and make your data searchable.
        </p>
      </div>

      {/* Drop zone */}
      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept=".pdf,.csv,.txt,.md,.json,.docx"
        className="hidden"
        onChange={(e) => { if (e.target.files) handleFiles(e.target.files); e.target.value = ""; }}
      />
      <div
        className="flex flex-col items-center gap-3 cursor-pointer transition-colors"
        style={{
          width: 560,
          background: isDragging ? "#E8E2FD" : "#F0EDFF",
          border: `2px dashed ${isDragging ? "#6510F4" : "#C4B5FD"}`,
          borderRadius: 16,
          padding: "36px 32px",
        }}
        onClick={() => fileInputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={(e) => { e.preventDefault(); setIsDragging(false); if (e.dataTransfer.files.length) handleFiles(e.dataTransfer.files); }}
      >
        <div className="flex items-center justify-center" style={{ width: 48, height: 48, background: "#fff", borderRadius: 12 }}>
          <UploadIcon />
        </div>
        <span style={{ fontSize: 15, fontWeight: 500, color: "#6510F4" }}>Drop files here or browse</span>
        <span style={{ fontSize: 13, color: "#71717A" }}>PDF, CSV, TXT, Markdown, JSON — up to 50 MB</span>
      </div>

      {/* File list */}
      {files.length > 0 && (
        <div style={{ width: 560 }} className="flex flex-col gap-2">
          {files.map((f, i) => (
            <div key={`f-${i}`} className="flex items-center justify-between" style={{ background: "#F4F4F5", borderRadius: 8, padding: "8px 12px" }}>
              <div className="flex items-center gap-2">
                <svg width="14" height="14" viewBox="0 0 16 16" fill="none"><rect x="3" y="1.5" width="10" height="13" rx="1.5" stroke="#71717A" strokeWidth="1.2" /><path d="M6 5h4M6 8h4M6 11h2" stroke="#71717A" strokeWidth="1.2" strokeLinecap="round" /></svg>
                <span style={{ fontSize: 13, color: "#18181B" }}>{f.name}</span>
                <span style={{ fontSize: 11, color: "#A1A1AA" }}>({(f.size / 1024).toFixed(0)} KB)</span>
              </div>
              <button onClick={() => removeFile(i)} className="cursor-pointer bg-transparent border-none p-1 hover:bg-black/5 rounded" style={{ color: "#A1A1AA", fontSize: 14 }}>&#10005;</button>
            </div>
          ))}
        </div>
      )}

      {files.length > 0 && (
        <button onClick={onNext} className="cursor-pointer" style={{ background: "#6510F4", border: "none", borderRadius: 8, padding: "10px 32px", fontSize: 14, fontWeight: 500, color: "#fff" }}>
          Continue with {files.length} file{files.length !== 1 ? "s" : ""}
        </button>
      )}

      <SkipLink />
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
      // Step 1: Upload
      updateStep(0, { status: "active", progress: 30 });
      let currentDsId = dsId;

      if (!currentDsId) {
        const ds = await createDataset({ name: "default_dataset" }, cogniInstance);
        currentDsId = ds.id;
        setDsId(currentDsId);
      }

      updateStep(0, { progress: 60 });
      await addData({ name: "default_dataset" }, files, cogniInstance);
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
      {step === 1 && <Step1 files={files} setFiles={setFiles} onNext={() => setStep(2)} />}
      {step === 2 && <Step2 files={files} datasetId={datasetId} onNext={(id) => { setDatasetId(id); setStep(3); }} cogniInstance={cogniInstance} />}
      {step === 3 && datasetId && <Step3 datasetId={datasetId} onNext={() => setStep(4)} cogniInstance={cogniInstance} />}
      {step === 4 && <Step4 />}
    </div>
  );
}
