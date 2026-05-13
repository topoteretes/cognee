"use client";

import { useState, useRef } from "react";
import { useRouter } from "next/navigation";
import { useCogniInstance } from "@/modules/tenant/TenantProvider";
import searchDataset from "@/modules/datasets/searchDataset";

function StepBadge({ step, total = 4 }: { step: number; total?: number }) {
  return (
    <div style={{ background: "#ECFDF5", borderRadius: 100, border: "1px solid #A7F3D0", padding: "5px 12px" }}>
      <span style={{ color: "#065F46", fontSize: 13, fontWeight: 500 }}>Step {step} of {total}</span>
    </div>
  );
}

function StepDots({ current, total = 4 }: { current: number; total?: number }) {
  return (
    <div className="flex items-center gap-2">
      {Array.from({ length: total }).map((_, i) => (
        <div key={i} style={{ width: 24, height: 4, borderRadius: 2, background: i + 1 === current ? "#059669" : "#A7F3D0" }} />
      ))}
    </div>
  );
}

function CodeBlock({ code }: { code: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <div style={{ position: "relative", background: "#18181B", borderRadius: 8, padding: "16px 20px", width: "100%", maxWidth: 520 }}>
      <pre style={{ margin: 0, fontSize: 13, lineHeight: 1.6, color: "#E4E4E7", overflowX: "auto", whiteSpace: "pre-wrap" }}>
        <code>{code}</code>
      </pre>
      <button
        onClick={() => { navigator.clipboard.writeText(code); setCopied(true); setTimeout(() => setCopied(false), 2000); }}
        className="cursor-pointer"
        style={{ position: "absolute", top: 8, right: 8, background: "#27272A", border: "1px solid #3F3F46", borderRadius: 4, padding: "4px 8px", fontSize: 11, color: "#A1A1AA" }}
      >
        {copied ? "Copied!" : "Copy"}
      </button>
    </div>
  );
}

// Step 1: Verify connection
function ServeStep1({ onNext }: { onNext: () => void }) {
  return (
    <div className="flex flex-col items-center gap-8 flex-1" style={{ paddingTop: 48, paddingBottom: 48, paddingInline: 80, fontFamily: '"Inter", system-ui, sans-serif' }}>
      <div className="flex flex-col items-center gap-2">
        <StepBadge step={1} />
        <h1 style={{ fontSize: 28, fontWeight: 600, color: "#18181B", margin: 0 }}>Connected to Cognee Cloud</h1>
        <p style={{ fontSize: 15, color: "#71717A", margin: 0, textAlign: "center", lineHeight: "22px", maxWidth: 480 }}>
          Your local SDK is connected to your Cognee Cloud instance via <code style={{ background: "#F4F4F5", padding: "1px 6px", borderRadius: 4, fontSize: 13 }}>cognee.serve()</code>.
          You can now use both the SDK and this UI to manage your knowledge graph.
        </p>
      </div>

      <div style={{ background: "#ECFDF5", border: "1px solid #A7F3D0", borderRadius: 12, padding: "20px 24px", display: "flex", gap: 12, alignItems: "center", maxWidth: 480, width: "100%" }}>
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none"><path d="M9 12l2 2 4-4" stroke="#059669" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/><circle cx="12" cy="12" r="10" stroke="#059669" strokeWidth="2"/></svg>
        <div>
          <div style={{ fontSize: 14, fontWeight: 600, color: "#065F46" }}>Connection active</div>
          <div style={{ fontSize: 13, color: "#059669" }}>All V2 operations route to your cloud instance</div>
        </div>
      </div>

      <button onClick={onNext} className="cursor-pointer" style={{ background: "#059669", border: "none", borderRadius: 8, padding: "10px 24px", fontSize: 14, fontWeight: 500, color: "#fff" }}>
        Continue
      </button>
      <StepDots current={1} />
    </div>
  );
}

// Step 2: SDK quickstart
function ServeStep2({ onNext }: { onNext: () => void }) {
  return (
    <div className="flex flex-col items-center gap-6 flex-1" style={{ paddingTop: 48, paddingBottom: 48, paddingInline: 80, fontFamily: '"Inter", system-ui, sans-serif' }}>
      <div className="flex flex-col items-center gap-2">
        <StepBadge step={2} />
        <h1 style={{ fontSize: 28, fontWeight: 600, color: "#18181B", margin: 0 }}>SDK quickstart</h1>
        <p style={{ fontSize: 15, color: "#71717A", margin: 0, textAlign: "center", lineHeight: "22px" }}>
          Use these commands from your Python environment to interact with your cloud graph.
        </p>
      </div>

      <div className="flex flex-col gap-4 w-full items-center">
        <div style={{ width: "100%", maxWidth: 520 }}>
          <div style={{ fontSize: 12, fontWeight: 600, color: "#71717A", marginBottom: 6, textTransform: "uppercase", letterSpacing: "0.05em" }}>Store data</div>
          <CodeBlock code={`import cognee\n\nawait cognee.serve()  # Already connected\n\nawait cognee.remember(\n    "Einstein developed general relativity in 1915.",\n    dataset_name="scientists"\n)`} />
        </div>

        <div style={{ width: "100%", maxWidth: 520 }}>
          <div style={{ fontSize: 12, fontWeight: 600, color: "#71717A", marginBottom: 6, textTransform: "uppercase", letterSpacing: "0.05em" }}>Query knowledge</div>
          <CodeBlock code={`results = await cognee.recall(\n    "What did Einstein develop?",\n    datasets=["scientists"]\n)\nprint(results)`} />
        </div>

        <div style={{ width: "100%", maxWidth: 520 }}>
          <div style={{ fontSize: 12, fontWeight: 600, color: "#71717A", marginBottom: 6, textTransform: "uppercase", letterSpacing: "0.05em" }}>Visualize</div>
          <CodeBlock code={`await cognee.visualize("graph.html")`} />
        </div>
      </div>

      <button onClick={onNext} className="cursor-pointer" style={{ background: "#059669", border: "none", borderRadius: 8, padding: "10px 24px", fontSize: 14, fontWeight: 500, color: "#fff" }}>
        Continue
      </button>
      <StepDots current={2} />
    </div>
  );
}

// Step 3: Test from UI
function ServeStep3({ onNext, cogniInstance }: {
  onNext: () => void;
  cogniInstance: NonNullable<ReturnType<typeof useCogniInstance>["cogniInstance"]>;
}) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<string[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const suggestions = ["What entities are in my dataset?", "Show relationships between concepts", "Summarize key findings"];

  const handleSearch = async (q: string) => {
    if (!q.trim()) return;
    setQuery(q);
    setIsSearching(true);
    setResults([]);
    try {
      const data = await searchDataset(cogniInstance, { query: q, searchType: "GRAPH_COMPLETION" });
      const texts: string[] = [];
      if (Array.isArray(data)) {
        for (const item of data) {
          if (Array.isArray(item.search_result)) {
            texts.push(...item.search_result);
          }
        }
      }
      setResults(texts.length > 0 ? texts : ["No results yet. Try adding data via the SDK first."]);
    } catch {
      setResults(["Search failed. Make sure you've added data via cognee.remember()."]);
    } finally {
      setIsSearching(false);
    }
  };

  return (
    <div className="flex flex-col items-center gap-6 flex-1" style={{ paddingTop: 48, paddingBottom: 48, paddingInline: 80, fontFamily: '"Inter", system-ui, sans-serif' }}>
      <div className="flex flex-col items-center gap-2">
        <StepBadge step={3} />
        <h1 style={{ fontSize: 28, fontWeight: 600, color: "#18181B", margin: 0 }}>Test from the UI</h1>
        <p style={{ fontSize: 15, color: "#71717A", margin: 0, textAlign: "center", lineHeight: "22px" }}>
          Try querying data you added via the SDK, or upload more from the dashboard later.
        </p>
      </div>

      <div style={{ display: "flex", gap: 8, width: "100%", maxWidth: 520 }}>
        <input
          ref={inputRef}
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter") handleSearch(query); }}
          placeholder="Ask a question about your data..."
          style={{ flex: 1, padding: "10px 14px", border: "1px solid #E4E4E7", borderRadius: 8, fontSize: 14, outline: "none" }}
        />
        <button onClick={() => handleSearch(query)} disabled={isSearching} className="cursor-pointer" style={{ background: "#059669", border: "none", borderRadius: 8, padding: "10px 20px", fontSize: 14, fontWeight: 500, color: "#fff", opacity: isSearching ? 0.6 : 1 }}>
          {isSearching ? "..." : "Search"}
        </button>
      </div>

      <div className="flex gap-2 flex-wrap justify-center">
        {suggestions.map((s) => (
          <button key={s} onClick={() => handleSearch(s)} className="cursor-pointer" style={{ background: "#F4F4F5", border: "1px solid #E4E4E7", borderRadius: 6, padding: "6px 12px", fontSize: 12, color: "#52525B" }}>
            {s}
          </button>
        ))}
      </div>

      {results.length > 0 && (
        <div style={{ background: "#fff", border: "1px solid #E4E4E7", borderRadius: 8, padding: 16, width: "100%", maxWidth: 520, maxHeight: 200, overflowY: "auto" }}>
          {results.map((r, i) => (
            <p key={i} style={{ margin: i > 0 ? "8px 0 0" : 0, fontSize: 14, color: "#3F3F46", lineHeight: 1.5 }}>{r}</p>
          ))}
        </div>
      )}

      <StepDots current={3} />

      <button onClick={onNext} className="cursor-pointer" style={{ background: "#059669", border: "none", borderRadius: 8, padding: "10px 24px", fontSize: 14, fontWeight: 500, color: "#fff" }}>
        Continue
      </button>
    </div>
  );
}

// Step 4: Done
function ServeStep4() {
  const router = useRouter();
  return (
    <div className="flex flex-col items-center justify-center gap-6 flex-1" style={{ padding: 48, fontFamily: '"Inter", system-ui, sans-serif' }}>
      <StepBadge step={4} />
      <h1 style={{ fontSize: 28, fontWeight: 600, color: "#18181B", margin: 0 }}>You&apos;re all set!</h1>
      <p style={{ fontSize: 15, color: "#71717A", margin: 0, textAlign: "center", maxWidth: 480, lineHeight: "22px" }}>
        Your SDK is connected and your knowledge graph is ready. Use the UI to explore, or keep working from your Python environment.
      </p>

      <div className="flex gap-3">
        <button onClick={() => router.push("/dashboard")} className="cursor-pointer" style={{ background: "#059669", color: "#fff", borderRadius: 8, padding: "10px 24px", fontSize: 14, fontWeight: 500, border: "none" }}>
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

export default function ServeOnboarding() {
  const { cogniInstance, isInitializing } = useCogniInstance();
  const [step, setStep] = useState(1);

  if (isInitializing || !cogniInstance) {
    return (
      <div className="flex items-center justify-center h-screen" style={{ fontFamily: '"Inter", system-ui, sans-serif' }}>
        <span style={{ fontSize: 14, color: "#71717A" }}>Connecting...</span>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full overflow-auto" style={{ background: "#FAFAF9" }}>
      {step === 1 && <ServeStep1 onNext={() => setStep(2)} />}
      {step === 2 && <ServeStep2 onNext={() => setStep(3)} />}
      {step === 3 && <ServeStep3 onNext={() => setStep(4)} cogniInstance={cogniInstance} />}
      {step === 4 && <ServeStep4 />}
    </div>
  );
}
