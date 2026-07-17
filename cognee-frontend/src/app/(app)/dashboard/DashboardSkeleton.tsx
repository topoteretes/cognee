"use client";

import { useState, useEffect } from "react";
import SkeletonBar from "@/ui/elements/SkeletonBar";

const cardStyle: React.CSSProperties = {
  border: "1px solid rgba(255,255,255,0.1)",
  borderRadius: 12,
  background: "rgba(255,255,255,0.03)",
  padding: 16,
  display: "flex",
  flexDirection: "column",
  gap: 12,
  minHeight: 120,
};

function StatCard() {
  return (
    <div
      style={{
        border: "1px solid rgba(255,255,255,0.1)",
        borderRadius: 10,
        background: "rgba(255,255,255,0.03)",
        padding: "14px 16px",
        display: "flex",
        flexDirection: "column",
        gap: 10,
        flex: "1 1 0",
        minWidth: 120,
      }}
    >
      <SkeletonBar width={64} height={10} />
      <SkeletonBar width={36} height={18} />
    </div>
  );
}

function ConnectionCard() {
  return (
    <div style={cardStyle}>
      <SkeletonBar width={40} height={40} />
      <SkeletonBar width="70%" height={12} />
      <SkeletonBar width="90%" height={10} />
    </div>
  );
}

export default function DashboardSkeleton() {
  // Escalates the message after 30s so a genuinely slow (but not yet
  // podUnreachable) pod doesn't sit silently under the same "usually under a
  // minute" text well past when that stops being true.
  const [elapsedMs, setElapsedMs] = useState(0);
  useEffect(() => {
    const start = Date.now();
    const id = setInterval(() => setElapsedMs(Date.now() - start), 1000);
    return () => clearInterval(id);
  }, []);
  const takingLonger = elapsedMs > 30_000;

  return (
    <div style={{ minHeight: "100%", flexShrink: 0 }}>
      <div style={{ padding: "clamp(16px, 3vw, 32px)", display: "flex", flexDirection: "column", gap: 40 }}>

        {/* Provisioning note */}
        <p style={{ margin: 0, fontSize: 13, color: "rgba(237,236,234,0.5)" }}>
          {takingLonger
            ? "Still going — this can take a couple of minutes on a brand-new account."
            : "Your workspace is being set up — usually takes under a minute."}
        </p>

        {/* Greeting */}
        <SkeletonBar width={260} height={26} />

        {/* KPI strip */}
        <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
          {Array.from({ length: 6 }).map((_, i) => (
            <StatCard key={i} />
          ))}
        </div>

        {/* Get started */}
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          <div>
            <div style={{ fontSize: 18, fontWeight: 700, color: "#EDECEA", lineHeight: "24px" }}>Get started</div>
            <div style={{ fontSize: 13, color: "rgba(237,236,234,0.65)", marginTop: 3 }}>Connect your AI agents to give them persistent memory</div>
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))", gap: 12 }}>
            {Array.from({ length: 4 }).map((_, i) => (
              <ConnectionCard key={i} />
            ))}
          </div>
        </div>

        {/* Memory Activity */}
        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          <div>
            <h2 style={{ margin: 0, fontSize: 18, fontWeight: 700, color: "#EDECEA", letterSpacing: "-0.01em", lineHeight: "24px" }}>Memory Activity</h2>
            <p style={{ margin: "3px 0 0", fontSize: 13, color: "rgba(237,236,234,0.55)" }}>A live log of every search against your memory — by your agents and by you.</p>
          </div>
          <div
            style={{
              border: "1px solid rgba(255,255,255,0.1)",
              borderRadius: 12,
              background: "rgba(255,255,255,0.03)",
              height: 220,
              padding: 16,
              display: "flex",
              flexDirection: "column",
              gap: 12,
            }}
          >
            {Array.from({ length: 5 }).map((_, i) => (
              <SkeletonBar key={i} width="100%" height={14} />
            ))}
          </div>
        </div>

      </div>
    </div>
  );
}
