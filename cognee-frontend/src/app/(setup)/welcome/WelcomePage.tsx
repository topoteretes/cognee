"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { notifications } from "@mantine/notifications";
import { useTenant } from "@/modules/tenant/TenantProvider";
import { useUser } from "@/modules/users/UserContext";
import { trackEvent } from "@/modules/analytics";
import TetrisBackground from "@/ui/elements/Auth/TetrisBackground";

export default function WelcomePage() {
  const router = useRouter();
  const { releaseLoader } = useTenant();
  const { markWelcomeSeen } = useUser();
  const [saving, setSaving] = useState(false);
  const [primaryHover, setPrimaryHover] = useState(false);
  const [secondaryHover, setSecondaryHover] = useState(false);
  const [showTutorial, setShowTutorial] = useState(false);

  useEffect(() => {
    releaseLoader();
  }, [releaseLoader]);

  useEffect(() => {
    if (!showTutorial) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") setShowTutorial(false);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [showTutorial]);

  async function handleLetsGo() {
    setSaving(true);
    trackEvent({ pageName: "Onboarding", eventName: "onboarding_welcome_cta" });
    // Navigate before persisting: markWelcomeSeen's optimistic cache update
    // flips isSeenWelcome to true while pathname is still "/welcome", which
    // races UserProvider's "already seen welcome" effect — that effect calls
    // router.replace("/dashboard") before this function's own router.push
    // below fires, so every user landed on the dashboard instead of
    // onboarding. Navigating first removes the race entirely. Onboarding now
    // goes straight to the preparing screen (its own former welcome step was
    // this same content, moved here to remove the duplicate).
    router.push("/onboarding");
    try {
      await markWelcomeSeen();
    } catch {
      notifications.show({ color: "red", message: "Something went wrong, please try again." });
    }
  }

  return (
    <div
      style={{
        minHeight: "100vh",
        position: "relative",
        backgroundColor: "#000000",
        backgroundImage:
          "linear-gradient(rgba(244,244,244,0.10) 1px, transparent 1px)," +
          "linear-gradient(90deg, rgba(244,244,244,0.10) 1px, transparent 1px)",
        backgroundSize: "33px 33px",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        padding: "56px 24px",
        boxSizing: "border-box",
        overflow: "hidden",
      }}
    >
      <TetrisBackground />

      <div
        style={{
          position: "relative",
          zIndex: 3,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: 24,
          background: "#2a2a2e",
          border: "1px solid rgba(255,255,255,0.1)",
          borderRadius: 16,
          padding: "48px 64px",
          maxWidth: 540,
          width: "100%",
          boxShadow: "0 20px 60px rgba(0,0,0,0.5)",
        }}
      >
        <h1
          style={{
            fontSize: 34,
            fontWeight: 300,
            color: "#EDECEA",
            margin: 0,
            textAlign: "center",
            letterSpacing: "-0.02em",
            lineHeight: 1.15,
            fontFamily: '"TWKLausanne", sans-serif',
          }}
        >
          Welcome to Cognee Cloud
        </h1>

        <p
          style={{
            fontSize: 15,
            color: "rgba(237,236,234,0.65)",
            margin: "0 0 12px",
            textAlign: "center",
            lineHeight: "24px",
            maxWidth: 400,
          }}
        >
          Let&apos;s take a minute to set up your account.
        </p>

        <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap", justifyContent: "center" }}>
          <button
            onClick={handleLetsGo}
            disabled={saving}
            onMouseEnter={() => setPrimaryHover(true)}
            onMouseLeave={() => setPrimaryHover(false)}
            style={{
              background: saving ? "rgba(188,155,255,0.5)" : primaryHover ? "#A87CFF" : "#BC9BFF",
              border: "none",
              borderRadius: 8,
              padding: "11px 32px",
              fontSize: 14,
              fontWeight: 500,
              color: "#1e1e1c",
              cursor: saving ? "not-allowed" : "pointer",
              letterSpacing: "-0.01em",
              transition: "background 150ms ease",
            }}
          >
            {saving ? "Saving…" : "Get started →"}
          </button>
          <button
            onClick={() => setShowTutorial(true)}
            onMouseEnter={() => setSecondaryHover(true)}
            onMouseLeave={() => setSecondaryHover(false)}
            style={{
              display: "inline-flex", alignItems: "center", gap: 7,
              background: secondaryHover ? "rgba(188,155,255,0.10)" : "transparent",
              border: `1px solid ${secondaryHover ? "#A87CFF" : "#BC9BFF"}`,
              borderRadius: 8,
              padding: "11px 20px", fontSize: 14, fontWeight: 500,
              color: secondaryHover ? "#A87CFF" : "#BC9BFF",
              cursor: "pointer",
              fontFamily: "inherit",
              transition: "background 150ms ease, border-color 150ms ease, color 150ms ease",
            }}
          >
            <svg width="13" height="13" viewBox="0 0 24 24" fill="#BC9BFF"><polygon points="5 3 19 12 5 21 5 3" /></svg>
            Watch Quick Tutorial
          </button>
        </div>
      </div>

      {/* Tutorial lightbox — video window over a dimmed backdrop */}
      {showTutorial && (
        <div
          onClick={() => setShowTutorial(false)}
          style={{
            position: "fixed",
            inset: 0,
            zIndex: 100,
            background: "rgba(0,0,0,0.82)",
            backdropFilter: "blur(6px)",
            WebkitBackdropFilter: "blur(6px)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            padding: 24,
          }}
        >
          <div
            onClick={(e) => e.stopPropagation()}
            style={{ position: "relative", width: "100%", maxWidth: 810 }}
          >
            <button
              onClick={() => setShowTutorial(false)}
              aria-label="Close tutorial"
              style={{
                position: "absolute",
                top: -36,
                right: 0,
                background: "none",
                border: "none",
                color: "rgba(237,236,234,0.7)",
                fontSize: 22,
                lineHeight: 1,
                cursor: "pointer",
                padding: 4,
              }}
            >
              ✕
            </button>
            <video
              controls
              autoPlay
              playsInline
              poster="/videos/full-demo-poster.jpg"
              style={{
                width: "100%",
                aspectRatio: "1280 / 962",
                background: "rgba(255,255,255,0.05)",
                border: "1px solid rgba(255,255,255,0.1)",
                borderRadius: 12,
                boxShadow: "0 20px 60px rgba(0,0,0,0.5)",
                display: "block",
              }}
            >
              <source src="/videos/full-demo.mp4" type="video/mp4" />
            </video>
          </div>
        </div>
      )}
    </div>
  );
}
