"use client";

import { useEffect, useState } from "react";

// Past this the wake has outrun the measured cold start (~35-60s) and the
// reassuring copy stops being honest.
const TAKING_LONGER_MS = 75_000;

function formatElapsed(ms: number): string {
  const total = Math.floor(ms / 1000);
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, "0")}`;
}

/**
 * Shown while a scaled-to-zero workspace is being resumed through the control
 * plane. Distinct from WorkspaceProvisioning on purpose: this workspace already
 * exists and its data is intact, so "being set up" is both wrong and mildly
 * alarming, and the user mainly wants to know nothing was lost.
 *
 * Deliberately no progress bar. The cold start is measured at ~35-60s but
 * varies, so a filling bar would be inventing a number; elapsed time is the
 * only thing we actually know.
 */
export default function WorkspaceWaking({ workspaceName, startedAt }: { workspaceName?: string; startedAt?: number | null }): React.ReactElement {
  const [elapsedMs, setElapsedMs] = useState(0);

  useEffect(() => {
    // The wake's own start time, not this component's. Two routes render this
    // screen from different places in the tree (the shell for pod-dependent
    // pages, OverviewPage for the dashboard), so clicking between them remounts
    // it — and a reload does too. Counting from mount restarted the one number
    // on the screen the user is actually watching, and reset the 75s copy with
    // it. The context sources this from the durable marker, so it survives both.
    const start = startedAt ?? Date.now();
    setElapsedMs(Date.now() - start);
    const id = setInterval(() => setElapsedMs(Date.now() - start), 1000);
    return () => clearInterval(id);
  }, [startedAt]);

  const takingLonger = elapsedMs > TAKING_LONGER_MS;

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        height: "100%",
        minHeight: 320,
        gap: 4,
        padding: 48,
        textAlign: "center",
      }}
    >
      <style>{`
        @keyframes wswake-spin { to { transform: rotate(360deg); } }
        @media (prefers-reduced-motion: reduce) { .wswake-spinner { animation: none !important; } }
      `}</style>
      <div
        className="wswake-spinner"
        style={{
          width: 36,
          height: 36,
          borderRadius: "50%",
          border: "3px solid rgba(237,236,234,0.12)",
          borderTopColor: "#BC9BFF",
          animation: "wswake-spin 0.8s linear infinite",
          marginBottom: 20,
        }}
      />

      {/* The live region is the message, not the whole panel: a region that
          wraps the spinner's <style> and the ticking clock announces noise. */}
      <div role="status" aria-live="polite">
        <h2 style={{ margin: 0, fontSize: 18, fontWeight: 700, color: "#EDECEA", letterSpacing: "-0.01em", maxWidth: 360, overflowWrap: "anywhere" }}>
          {workspaceName ? `Waking ${workspaceName}` : "Waking your workspace"}
        </h2>
        <p style={{ margin: "8px 0 0", fontSize: 13, color: "rgba(237,236,234,0.65)", maxWidth: 360, lineHeight: "20px" }}>
          {takingLonger
            ? "Still waking up. This one is taking longer than usual, but your data is safe."
            : "It went to sleep after a quiet spell. Starting it back up usually takes under a minute, and nothing was lost."}
        </p>
      </div>

      <div
        aria-hidden="true"
        style={{
          marginTop: 18,
          fontSize: 11,
          letterSpacing: "0.08em",
          color: "rgba(237,236,234,0.4)",
          fontVariantNumeric: "tabular-nums",
        }}
      >
        RESTORING · {formatElapsed(elapsedMs)}
      </div>
    </div>
  );
}
