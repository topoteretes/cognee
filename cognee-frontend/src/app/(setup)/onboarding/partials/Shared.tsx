"use client";

import { useRouter } from "next/navigation";
import { useUser } from "@/modules/users/UserContext";
import { completeOnboardingAndNavigate } from "../completeOnboardingAndNavigate";
import { useOnboardingTrackEvent } from "../useOnboardingTrackEvent";

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

// Back / Set up later / Continue, matching the agent-connect flow's footer so
// both onboarding paths steer the same way.
//
// Renders as the card's own bottom strip, like the agent flow's. That card puts
// padding on each section, so its footer can simply sit last; these cards put
// the padding on the card itself, so the footer cancels it with negative
// margins to reach the edges instead. Pass the card's padding if it differs
// from the 48/64 the onboarding cards share.
export function OnboardingFooter({ onBack, onContinue, continueDisabled = false, cardPaddingX = 64, cardPaddingBottom = 48 }: {
  onBack: () => void;
  onContinue: () => void;
  continueDisabled?: boolean;
  cardPaddingX?: number;
  cardPaddingBottom?: number;
}) {
  const router = useRouter();
  const { markOnboardingComplete } = useUser();
  const track = useOnboardingTrackEvent();

  function setUpLater(): void {
    track({ pageName: "Onboarding", eventName: "onboarding_skipped" });
    completeOnboardingAndNavigate(markOnboardingComplete, () => router.push("/dashboard"));
  }

  return (
    <div style={{
      display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12,
      alignSelf: "stretch", boxSizing: "border-box",
      marginTop: 8,
      marginLeft: -cardPaddingX, marginRight: -cardPaddingX, marginBottom: -cardPaddingBottom,
      padding: `14px ${cardPaddingX}px`,
      borderTop: "1px solid rgba(255,255,255,0.07)",
    }}>
      <button
        onClick={onBack}
        className="cursor-pointer"
        style={{ display: "inline-flex", alignItems: "center", gap: 7, background: "transparent", border: "1px solid rgba(255,255,255,0.12)", padding: "9px 16px", fontSize: 13, fontWeight: 500, color: "rgba(237,236,234,0.65)", fontFamily: "inherit" }}
      >
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="19" y1="12" x2="5" y2="12" /><polyline points="12 19 5 12 12 5" /></svg>
        Back
      </button>
      <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
        <button
          onClick={setUpLater}
          className="cursor-pointer"
          style={{ background: "none", border: "none", color: "rgba(237,236,234,0.5)", fontSize: 13, fontFamily: "inherit", padding: "9px 6px" }}
        >
          Set up later
        </button>
        {/* Unlike the agent flow, disabling Continue here is safe: these steps
            genuinely cannot proceed without data, and "Set up later" is always
            an escape. */}
        <button
          onClick={onContinue}
          disabled={continueDisabled}
          className={continueDisabled ? undefined : "cursor-pointer"}
          style={{
            display: "inline-flex", alignItems: "center", gap: 7,
            background: continueDisabled ? "rgba(255,255,255,0.07)" : "#BC9BFF",
            border: "none", padding: "9px 20px", fontSize: 13, fontWeight: 500,
            color: continueDisabled ? "rgba(237,236,234,0.35)" : "#1e1e1c",
            fontFamily: "inherit", cursor: continueDisabled ? "not-allowed" : "pointer",
          }}
        >
          Continue
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="5" y1="12" x2="19" y2="12" /><polyline points="12 5 19 12 12 19" /></svg>
        </button>
      </div>
    </div>
  );
}

export function SkipLink({ label = "Skip onboarding and go to dashboard", compact = false }: { label?: string; compact?: boolean } = {}) {
  const router = useRouter();
  const { markOnboardingComplete } = useUser();
  const track = useOnboardingTrackEvent();
  return (
    <button
      onClick={() => {
        track({ pageName: "Onboarding", eventName: "onboarding_skipped" });
        completeOnboardingAndNavigate(markOnboardingComplete, () => router.push("/dashboard"));
      }}
      className="cursor-pointer"
      style={{ background: "none", border: "none", color: "rgba(237,236,234,0.65)", fontSize: 13, paddingTop: compact ? 12 : 32, paddingBottom: compact ? 0 : 24 }}
    >
      {label}
    </button>
  );
}
