"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { AgentStep, DatabaseStep, LocalCogneeStep } from "../onboarding/ConnectionSteps";

type ConnectionType = "agent" | "local" | "database" | null;

function AgentIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#6510F4" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="11" width="18" height="10" rx="2" /><circle cx="12" cy="5" r="2" /><path d="M12 7v4" /><line x1="8" y1="16" x2="8" y2="16" /><line x1="16" y1="16" x2="16" y2="16" />
    </svg>
  );
}

function ServerIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#6510F4" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <rect x="2" y="2" width="20" height="8" rx="2" ry="2" /><rect x="2" y="14" width="20" height="8" rx="2" ry="2" /><line x1="6" y1="6" x2="6.01" y2="6" /><line x1="6" y1="18" x2="6.01" y2="18" />
    </svg>
  );
}

function DatabaseIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#6510F4" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <ellipse cx="12" cy="5" rx="9" ry="3" /><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3" /><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5" />
    </svg>
  );
}

function ConnectionCard({ icon, title, subtitle, onClick }: { icon: React.ReactNode; title: string; subtitle: string; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="cursor-pointer hover:border-cognee-purple"
      style={{ display: "flex", alignItems: "center", gap: 16, background: "#fff", border: "1px solid #E4E4E7", borderRadius: 12, padding: "20px 24px", width: "100%", textAlign: "left", transition: "border-color 150ms", fontFamily: "inherit" }}
    >
      <div style={{ width: 44, height: 44, background: "#F0EDFF", borderRadius: 10, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
        {icon}
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 2, flex: 1 }}>
        <span style={{ fontSize: 15, fontWeight: 500, color: "#18181B" }}>{title}</span>
        <span style={{ fontSize: 13, color: "#71717A" }}>{subtitle}</span>
      </div>
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#A1A1AA" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0 }}>
        <polyline points="9 18 15 12 9 6" />
      </svg>
    </button>
  );
}

export default function ConnectAgentPage() {
  const router = useRouter();
  const [view, setView] = useState<ConnectionType>(null);

  const handleBack = () => setView(null);
  const handleSkip = () => router.push("/dashboard");

  if (view === "agent") return <AgentStep onBack={handleBack} onSkip={handleSkip} />;
  if (view === "local") return <LocalCogneeStep onBack={handleBack} onSkip={handleSkip} />;
  if (view === "database") return <DatabaseStep onBack={handleBack} onSkip={handleSkip} />;

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 32, padding: "48px 32px", fontFamily: '"Inter", system-ui, sans-serif' }}>
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 8, maxWidth: 480, textAlign: "center" }}>
        <h1 style={{ fontSize: 26, fontWeight: 600, color: "#18181B", margin: 0 }}>Connect to Cognee</h1>
        <p style={{ fontSize: 14, color: "#71717A", margin: 0, lineHeight: "22px" }}>
          Choose how you want to connect. You can link an agent framework, sync a local Cognee instance, or ingest data from a database.
        </p>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 12, width: "100%", maxWidth: 520 }}>
        <ConnectionCard
          icon={<AgentIcon />}
          title="Connect an Agent"
          subtitle="OpenClaw, CrewAI, LangGraph, AutoGen, or custom"
          onClick={() => setView("agent")}
        />
        <ConnectionCard
          icon={<ServerIcon />}
          title="Sync Local Cognee"
          subtitle="Connect your local instance to Cognee Cloud"
          onClick={() => setView("local")}
        />
        <ConnectionCard
          icon={<DatabaseIcon />}
          title="Ingest from Any Source"
          subtitle="Databases, Slack, Notion, GitHub, CSV, REST APIs"
          onClick={() => setView("database")}
        />
      </div>
    </div>
  );
}
