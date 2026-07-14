"use client";

import { useRouter } from "next/navigation";
import { useCogniInstance } from "@/modules/tenant/TenantProvider";
import { useUser } from "@/modules/users/UserContext";
import { trackEvent } from "@/modules/analytics";
import { AgentActivityTerminal, type OnboardingDemoEntry } from "@/ui/elements/AgentActivityTerminal";
import { completeOnboardingAndNavigate } from "../completeOnboardingAndNavigate";
import { StepBadge, StepDots } from "./Shared";

// Mirror of the "no answer" detection used in the terminal so the parent
// can store a non-boilerplate top result. Kept narrow on purpose.
export function extractFirstAnswer(data: unknown): string | null {
  if (!Array.isArray(data)) return null;
  const noAnswerPatterns: RegExp[] = [
    /no (relevant |specific |such )?(information|data|context|knowledge|results?|answer)/i,
    /\b(does not|doesn'?t|do not|don'?t)\b[^.]{0,60}\b(contain|include|provide|mention|have|appear)/i,
    /unable to (find|answer|provide|determine|locate)/i,
    /there is no (information|mention|data|reference)/i,
  ];
  for (const row of data as Array<{ text?: unknown; answer?: unknown; search_result?: unknown }>) {
    let text: string | null = null;
    if (typeof row.text === "string") text = row.text;
    else if (typeof row.answer === "string") text = row.answer;
    else if (typeof row.search_result === "string") text = row.search_result;
    else if (Array.isArray(row.search_result) && row.search_result.every((x) => typeof x === "string")) text = (row.search_result as string[]).join("\n\n");
    if (!text || !text.trim()) continue;
    const t = text.trim();
    if (t.length < 300 && noAnswerPatterns.some((p) => p.test(t))) continue;
    return t.slice(0, 220);
  }
  return null;
}

export function Step3({ datasetId, cogniInstance, demoEntries }: {
  datasetId: string;
  cogniInstance: NonNullable<ReturnType<typeof useCogniInstance>["cogniInstance"]>;
  demoEntries: OnboardingDemoEntry[] | null;
}) {
  const router = useRouter();
  const { markOnboardingComplete } = useUser();
  const dataset = { id: datasetId, name: "default_dataset" };

  return (
    <div style={{
      minHeight: "100vh",
      backgroundColor: "#000000",
      position: "relative",
      display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
      padding: "56px 24px", boxSizing: "border-box",
      overflow: "hidden",
    }}>
      <div style={{
        position: "absolute", inset: -20, zIndex: 0,
        backgroundImage: "linear-gradient(rgba(244,244,244,0.10) 1px, transparent 1px), linear-gradient(90deg, rgba(244,244,244,0.10) 1px, transparent 1px)",
        backgroundSize: "33px 33px",
        pointerEvents: "none",
      }} />
      <div style={{
        position: "relative", zIndex: 1,
        padding: "48px 64px",
        display: "flex", flexDirection: "column", alignItems: "center", gap: 24,
        background: "#2a2a2e",
        border: "1px solid rgba(255,255,255,0.1)",
        borderRadius: 16,
        boxShadow: "0 20px 60px rgba(0,0,0,0.5)",
        maxWidth: 860, width: "100%", boxSizing: "border-box",
      }}>
      <StepBadge step={3} total={3} />
      <h1 style={{ fontSize: 28, fontWeight: 300, color: "#EDECEA", margin: 0, fontFamily: '"TWKLausanne", sans-serif', letterSpacing: "-0.02em" }}>Ask cognee anything</h1>
      <p style={{ fontSize: 15, color: "rgba(237,236,234,0.65)", margin: 0, textAlign: "center", lineHeight: "22px" }}>
        Your memory is ready. Ask anything about your data below.
      </p>

      <div style={{ width: "100%", maxWidth: 780 }}>
        <AgentActivityTerminal
          sessions={[]}
          runs={[]}
          agents={[]}
          datasets={[dataset]}
          selectedDataset={dataset}
          cogniInstance={cogniInstance}
          dataLoading={false}
          range="24h"
          onNavigate={router.push}
          variant="onboarding"
          onboardingDemo={demoEntries}
        />
      </div>

      <StepDots current={3} total={3} />

      <button
        onClick={() => {
          trackEvent({ pageName: "Onboarding", eventName: "onboarding_completed", additionalProperties: { destination: "dashboard" } });
          completeOnboardingAndNavigate(markOnboardingComplete, () => router.push("/dashboard"));
        }}
        className="cursor-pointer"
        style={{ background: "#BC9BFF", border: "none", borderRadius: 8, padding: "11px 32px", fontSize: 14, fontWeight: 500, color: "#1e1e1c", letterSpacing: "-0.01em" }}
      >
        Connect my agent now →
      </button>
      </div>
    </div>
  );
}
