"use client";

import { useRouter } from "next/navigation";
import { useUser } from "@/modules/users/UserContext";
import { trackEvent } from "@/modules/analytics";
import { completeOnboardingAndNavigate } from "../completeOnboardingAndNavigate";

export function StepBadge({ step, total = 4 }: { step: number; total?: number }) {
  return (
    <div style={{ background: "rgba(188,155,255,0.20)", borderRadius: 100, border: "1px solid rgba(188,155,255,0.35)", padding: "5px 12px" }}>
      <span style={{ color: "#EDECEA", fontSize: 13, fontWeight: 500 }}>Step {step} of {total}</span>
    </div>
  );
}

export function StepDots({ current, total = 4 }: { current: number; total?: number }) {
  return (
    <div className="flex items-center gap-2">
      {Array.from({ length: total }).map((_, i) => (
        <div key={i} style={{ width: 24, height: 4, borderRadius: 2, background: i + 1 === current ? "#BC9BFF" : "rgba(255,255,255,0.2)" }} />
      ))}
    </div>
  );
}

export function SkipLink({ label = "Skip onboarding and go to dashboard", compact = false }: { label?: string; compact?: boolean } = {}) {
  const router = useRouter();
  const { markOnboardingComplete } = useUser();
  return (
    <button
      onClick={() => {
        trackEvent({ pageName: "Onboarding", eventName: "onboarding_skipped" });
        completeOnboardingAndNavigate(markOnboardingComplete, () => router.push("/dashboard"));
      }}
      className="cursor-pointer"
      style={{ background: "none", border: "none", color: "rgba(237,236,234,0.65)", fontSize: 13, paddingTop: compact ? 12 : 32, paddingBottom: compact ? 0 : 24 }}
    >
      {label}
    </button>
  );
}
