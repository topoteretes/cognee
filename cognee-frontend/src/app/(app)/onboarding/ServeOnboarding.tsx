"use client";

import { useState, useRef } from "react";
import { useRouter } from "next/navigation";
import { useCogniInstance } from "@/modules/tenant/TenantProvider";
import searchDataset from "@/modules/datasets/searchDataset";
import markOnboardingComplete from "@/modules/users/markOnboardingComplete";
import { markOnboardingCompleteLocally } from "@/utils/onboardingFlag";
import { trackEvent, TrackPageView } from "@/modules/analytics";

const darkPage: React.CSSProperties = {
  backgroundColor: "#000000",
  backgroundImage: "linear-gradient(rgba(244,244,244,0.10) 1px, transparent 1px), linear-gradient(90deg, rgba(244,244,244,0.10) 1px, transparent 1px)",
  backgroundSize: "33px 33px",
};

function StepBadge({ step, total = 4 }: { step: number; total?: number }) {
  return (
    <div style={{ background: "rgba(188,155,255,0.20)", borderRadius: 100, border: "1px solid rgba(188,155,255,0.35)", padding: "5px 12px" }}>
      <span style={{ color: "#EDECEA", fontSize: 13, fontWeight: 500 }}>Step {step} of {total}</span>
    </div>
  );
}

function StepDots({ current, total = 4 }: { current: number; total?: number }) {
  return (
    <div className="flex items-center gap-2">
      {Array.from({ length: total }).map((_, i) => (
        <div key={i} style={{ width: 24, height: 4, borderRadius: 2, background: i + 1 === current ? "#BC9BFF" : "rgba(255,255,255,0.2)" }} />
      ))}
    </div>
  );
}

function CodeBlock({ code }: { code: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <div style={{ position: "relative", background: "rgba(255,255,255,0.08)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 8, padding: "16px 20px", width: "100%", maxWidth: 520 }}>
      <pre style={{ margin: 0, fontSize: 13, color: "#E4E4E7", overflowX: "auto", whiteSpace: "pre-wrap" }}>
        <code>{code}</code>
      </pre>
      <button
        onClick={() => { navigator.clipboard.writeText(code); setCopied(true); setTimeout(() => setCopied(false), 2000); trackEvent({ pageName: "Onboarding Serve", eventName: "serve_code_copied", additionalProperties: { snippet: code.slice(0, 30) } }); }}
        className="cursor-pointer"
        style={{ position: "absolute", top: 8, right: 8, background: "#27272A", border: "1px solid #3F3F46", borderRadius: 4, padding: "4px 8px", fontSize: 11, color: "rgba(237,236,234,0.65)" }}
      >
        {copied ? "Copied!" : "Copy"}
      </button>
    </div>
  );
}

// Step 1: Verify connection
function ServeStep1({ onNext }: { onNext: () => void }) {
  return (
    <div className="flex flex-col items-center gap-8 flex-1" style={{ paddingTop: 48, paddingBottom: 48, paddingInline: 80 }}>
      <div className="flex flex-col items-center gap-2">
        <StepBadge step={1} />
        <h1 style={{ fontSize: 28, fontWeight: 300, color: "#EDECEA", margin: 0, fontFamily: '"TWKLausanne", sans-serif', letterSpacing: "-0.02em" }}>Connected to Cognee Cloud</h1>
        <p style={{ fontSize: 15, color: "rgba(237,236,234,0.55)", margin: 0, textAlign: "center", lineHeight: "22px", maxWidth: 480 }}>
          Your local SDK is connected to your Cognee Cloud instance via <code style={{ background: "rgba(255,255,255,0.1)", color: "#EDECEA", padding: "1px 6px", borderRadius: 4, fontSize: 13 }}>cognee.serve()</code>.
          You can now use both the SDK and this UI to manage your knowledge graph.
        </p>
      </div>

      <div style={{ background: "rgba(34,197,94,0.12)", border: "1px solid rgba(34,197,94,0.35)", borderRadius: 12, padding: "20px 24px", display: "flex", gap: 12, alignItems: "center", maxWidth: 480, width: "100%" }}>
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none"><path d="M9 12l2 2 4-4" stroke="#22C55E" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/><circle cx="12" cy="12" r="10" stroke="#22C55E" strokeWidth="2"/></svg>
        <div>
          <div style={{ fontSize: 14, fontWeight: 700, color: "#22C55E" }}>Connection active</div>
          <div style={{ fontSize: 13, color: "rgba(34,197,94,0.8)" }}>All V2 operations route to your cloud instance</div>
        </div>
      </div>

      <button onClick={() => { trackEvent({ pageName: "Onboarding Serve", eventName: "serve_step_completed", additionalProperties: { step: "1" } }); onNext(); }} className="cursor-pointer" style={{ background: "#BC9BFF", border: "none", borderRadius: 8, padding: "10px 24px", fontSize: 14, fontWeight: 500, color: "#1e1e1c" }}>
        Continue
      </button>
      <StepDots current={1} />
    </div>
  );
}

// Step 2: SDK quickstart
function ServeStep2({ onNext }: { onNext: () => void }) {
  return (
    <div className="flex flex-col items-center gap-6 flex-1" style={{ paddingTop: 48, paddingBottom: 48, paddingInline: 80 }}>
      <div className="flex flex-col items-center gap-2">
        <StepBadge step={2} />
        <h1 style={{ fontSize: 28, fontWeight: 300, color: "#EDECEA", margin: 0, fontFamily: '"TWKLausanne", sans-serif', letterSpacing: "-0.02em" }}>SDK quickstart</h1>
        <p style={{ fontSize: 15, color: "rgba(237,236,234,0.55)", margin: 0, textAlign: "center", lineHeight: "22px" }}>
          Use these commands from your Python environment to interact with your cloud graph.
        </p>
      </div>

      <div className="flex flex-col gap-4 w-full items-center">
        <div style={{ width: "100%", maxWidth: 520 }}>
          <div style={{ fontSize: 12, fontWeight: 700, color: "rgba(237,236,234,0.45)", marginBottom: 6, textTransform: "uppercase", letterSpacing: "0.05em" }}>Store data</div>
          <CodeBlock code={`import cognee\n\nawait cognee.serve()  # Already connected\n\nawait cognee.remember(\n    "Einstein developed general relativity in 1915.",\n    dataset_name="scientists"\n)`} />
        </div>

        <div style={{ width: "100%", maxWidth: 520 }}>
          <div style={{ fontSize: 12, fontWeight: 700, color: "rgba(237,236,234,0.45)", marginBottom: 6, textTransform: "uppercase", letterSpacing: "0.05em" }}>Query knowledge</div>
          <CodeBlock code={`results = await cognee.recall(\n    "What did Einstein develop?",\n    datasets=["scientists"]\n)\nprint(results)`} />
        </div>

        <div style={{ width: "100%", maxWidth: 520 }}>
          <div style={{ fontSize: 12, fontWeight: 700, color: "rgba(237,236,234,0.45)", marginBottom: 6, textTransform: "uppercase", letterSpacing: "0.05em" }}>Visualize</div>
          <CodeBlock code={`await cognee.visualize("graph.html")`} />
        </div>
      </div>

      <button onClick={() => { trackEvent({ pageName: "Onboarding Serve", eventName: "serve_step_completed", additionalProperties: { step: "2" } }); onNext(); }} className="cursor-pointer" style={{ background: "#BC9BFF", border: "none", borderRadius: 8, padding: "10px 24px", fontSize: 14, fontWeight: 500, color: "#1e1e1c" }}>
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
    trackEvent({ pageName: "Onboarding Serve", eventName: "serve_search_executed", additionalProperties: { query_length: String(q.length) } });
    setQuery(q);
    setIsSearching(true);
    setResults([]);
    try {
      const data = await searchDataset(cogniInstance, { query: q, searchType: "HYBRID_COMPLETION" });
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
    <div className="flex flex-col items-center gap-6 flex-1" style={{ paddingTop: 48, paddingBottom: 48, paddingInline: 80 }}>
      <div className="flex flex-col items-center gap-2">
        <StepBadge step={3} />
        <h1 style={{ fontSize: 28, fontWeight: 300, color: "#EDECEA", margin: 0, fontFamily: '"TWKLausanne", sans-serif', letterSpacing: "-0.02em" }}>Test from the UI</h1>
        <p style={{ fontSize: 15, color: "rgba(237,236,234,0.55)", margin: 0, textAlign: "center", lineHeight: "22px" }}>
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
          style={{ flex: 1, padding: "10px 14px", border: "1px solid rgba(255,255,255,0.12)", background: "rgba(255,255,255,0.06)", color: "#EDECEA", borderRadius: 8, fontSize: 14, outline: "none" }}
        />
        <button onClick={() => handleSearch(query)} disabled={isSearching} className="cursor-pointer" style={{ background: "#BC9BFF", border: "none", borderRadius: 8, padding: "10px 20px", fontSize: 14, fontWeight: 500, color: "#1e1e1c", opacity: isSearching ? 0.6 : 1 }}>
          {isSearching ? "..." : "Search"}
        </button>
      </div>

      <div className="flex gap-2 flex-wrap justify-center">
        {suggestions.map((s) => (
          <button key={s} onClick={() => { trackEvent({ pageName: "Onboarding Serve", eventName: "serve_suggestion_clicked", additionalProperties: { suggestion: s } }); handleSearch(s); }} className="cursor-pointer hover:bg-white/10" style={{ background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.12)", borderRadius: 6, padding: "6px 12px", fontSize: 12, color: "rgba(237,236,234,0.7)" }}>
            {s}
          </button>
        ))}
      </div>

      {results.length > 0 && (
        <div style={{ background: "rgba(255,255,255,0.08)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 8, padding: 16, width: "100%", maxWidth: 520, maxHeight: 200, overflowY: "auto" }}>
          {results.map((r, i) => (
            <p key={i} style={{ margin: i > 0 ? "8px 0 0" : 0, fontSize: 14, color: "rgba(237,236,234,0.8)", lineHeight: 1.5 }}>{r}</p>
          ))}
        </div>
      )}

      <StepDots current={3} />

      <button onClick={() => { trackEvent({ pageName: "Onboarding Serve", eventName: "serve_step_completed", additionalProperties: { step: "3" } }); onNext(); }} className="cursor-pointer" style={{ background: "#BC9BFF", border: "none", borderRadius: 8, padding: "10px 24px", fontSize: 14, fontWeight: 500, color: "#1e1e1c" }}>
        Continue
      </button>
    </div>
  );
}

// Step 4: Done
function ServeStep4() {
  const router = useRouter();

  const finish = (destination: string) => {
    markOnboardingCompleteLocally();
    sessionStorage.setItem("cognee-onboarding-skipped", "1");
    markOnboardingComplete().catch(() => {}); // persist to Auth0 — fire-and-forget
    trackEvent({ pageName: "Onboarding Serve", eventName: "serve_onboarding_completed", additionalProperties: { destination } });
    router.push(`/${destination}`);
  };

  return (
    <div className="flex flex-col items-center justify-center gap-6 flex-1" style={{ padding: 48 }}>
      <StepBadge step={4} />
      <h1 style={{ fontSize: 28, fontWeight: 300, color: "#EDECEA", margin: 0, fontFamily: '"TWKLausanne", sans-serif', letterSpacing: "-0.02em" }}>You&apos;re all set!</h1>
      <p style={{ fontSize: 15, color: "rgba(237,236,234,0.55)", margin: 0, textAlign: "center", maxWidth: 480, lineHeight: "22px" }}>
        Your SDK is connected and your knowledge graph is ready. Use the UI to explore, or keep working from your Python environment.
      </p>

      <div className="flex gap-3">
        <button onClick={() => finish("datasets")} className="cursor-pointer" style={{ background: "#BC9BFF", color: "#1e1e1c", borderRadius: 8, padding: "10px 24px", fontSize: 14, fontWeight: 500, border: "none" }}>
          View datasets
        </button>
      </div>

      <StepDots current={4} />
    </div>
  );
}

export default function ServeOnboarding() {
  const { cogniInstance, isInitializing } = useCogniInstance();
  const router = useRouter();
  const [step, setStep] = useState(1);

  function skipToDashboard() {
    try {
      markOnboardingCompleteLocally();
      sessionStorage.setItem("cognee-onboarding-skipped", "1");
    } catch { /* ignore */ }
    markOnboardingComplete().catch(() => {});
    router.push("/dashboard");
  }

  if (isInitializing || !cogniInstance) {
    return (
      <>
        <TrackPageView page="Onboarding Serve" />
        <div className="flex flex-col items-center justify-center h-screen gap-4" style={darkPage}>
          <span style={{ fontSize: 14, color: "rgba(237,236,234,0.65)" }}>Still preparing your memory…</span>
          <button
            onClick={skipToDashboard}
            className="cursor-pointer"
            style={{ background: "none", border: "1px solid rgba(255,255,255,0.2)", borderRadius: 8, padding: "8px 18px", fontSize: 13, fontWeight: 500, color: "rgba(237,236,234,0.8)" }}
          >
            Skip to dashboard
          </button>
        </div>
      </>
    );
  }

  return (
    <div className="flex flex-col h-full overflow-auto" style={darkPage}>
      {step === 1 && <ServeStep1 onNext={() => setStep(2)} />}
      {step === 2 && <ServeStep2 onNext={() => setStep(3)} />}
      {step === 3 && <ServeStep3 onNext={() => setStep(4)} cogniInstance={cogniInstance} />}
      {step === 4 && <ServeStep4 />}
    </div>
  );
}
