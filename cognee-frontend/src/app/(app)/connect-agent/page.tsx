"use client";

import { useState, useRef, useEffect } from "react";
import { useRouter } from "next/navigation";
import { AgentStep, DatabaseStep, LocalCogneeStep } from "../onboarding/ConnectionSteps";
import { trackEvent, TrackPageView } from "@/modules/analytics";
import QuickstartCards from "@/ui/elements/QuickstartCards";

const TABS = [
  {
    key: "agent",
    title: "Connect an Agent",
    subtitle: "OpenClaw, CrewAI, LangGraph, AutoGen, or custom",
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <rect x="3" y="11" width="18" height="10" rx="2" /><circle cx="12" cy="5" r="2" /><path d="M12 7v4" /><line x1="8" y1="16" x2="8" y2="16" /><line x1="16" y1="16" x2="16" y2="16" />
      </svg>
    ),
  },
  {
    key: "local",
    title: "Sync Local Cognee",
    subtitle: "Connect your local instance to Cognee Cloud",
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <rect x="2" y="2" width="20" height="8" rx="2" ry="2" /><rect x="2" y="14" width="20" height="8" rx="2" ry="2" /><line x1="6" y1="6" x2="6.01" y2="6" /><line x1="6" y1="18" x2="6.01" y2="18" />
      </svg>
    ),
  },
  {
    key: "database",
    title: "Ingest from Any Source",
    subtitle: "Databases, Slack, Notion, GitHub, CSV, REST APIs",
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <ellipse cx="12" cy="5" rx="9" ry="3" /><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3" /><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5" />
      </svg>
    ),
  },
];

export default function ConnectAgentPage() {
  const router = useRouter();
  const [activeTab, setActiveTab] = useState<string | null>(null);
  const contentRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (activeTab && contentRef.current) {
      contentRef.current.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }, [activeTab]);

  const handleTabClick = (key: string) => {
    setActiveTab(activeTab === key ? null : key);
    trackEvent({ pageName: "Connect Agent", eventName: "connection_tab_clicked", additionalProperties: { tab: key } });
  };

  const noop = () => {};
  const handleSkip = (type: string) => {
    trackEvent({ pageName: "Connect Agent", eventName: "connection_type_selected", additionalProperties: { connection_type: type } });
    router.push("/dashboard");
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 24, padding: 32, fontFamily: '"Inter", system-ui, sans-serif' }}>
      <TrackPageView page="Connect Agent" />

      {/* Header */}
      <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        <h1 style={{ fontSize: 20, fontWeight: 300, color: "#18181B", margin: 0, fontFamily: '"TWK Lausanne", system-ui, sans-serif' }}>Connect to Cognee</h1>
        <p style={{ fontSize: 14, color: "#71717A", margin: 0 }}>
          Choose how you want to connect. You can link an agent framework, sync a local Cognee instance, or ingest data from a database.
        </p>
      </div>

      {/* Quickstart prompt cards */}
      <QuickstartCards />

      {/* Tab headers — horizontal row */}
      <div style={{ display: "flex", gap: 12 }}>
        {TABS.map((tab) => {
          const isActive = activeTab === tab.key;
          return (
            <button
              key={tab.key}
              onClick={() => handleTabClick(tab.key)}
              className="cursor-pointer"
              style={{
                flex: 1,
                display: "flex",
                alignItems: "center",
                gap: 12,
                padding: "14px 16px",
                borderRadius: 10,
                border: `1px solid ${isActive ? "#6510F4" : "#E4E4E7"}`,
                background: isActive ? "#F9F8FF" : "#fff",
                cursor: "pointer",
                fontFamily: "inherit",
                transition: "all 150ms",
              }}
            >
              <div style={{ color: isActive ? "#6510F4" : "#71717A", flexShrink: 0, transition: "color 150ms" }}>
                {tab.icon}
              </div>
              <div style={{ textAlign: "left" }}>
                <span style={{ display: "block", fontSize: 14, fontWeight: 500, color: isActive ? "#6510F4" : "#18181B", lineHeight: 1.3, fontFamily: '"TWK Lausanne", system-ui, sans-serif' }}>
                  {tab.title}
                </span>
                <span style={{ display: "block", fontSize: 12, color: "#71717A", lineHeight: 1.4, marginTop: 2 }}>
                  {tab.subtitle}
                </span>
              </div>
              <svg
                width="16" height="16" viewBox="0 0 16 16" fill="none"
                style={{ marginLeft: "auto", flexShrink: 0, transform: isActive ? "rotate(180deg)" : "rotate(0deg)", transition: "transform 200ms", color: isActive ? "#6510F4" : "#A1A1AA" }}
              >
                <path d="M4 6l4 4 4-4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </button>
          );
        })}
      </div>

      {/* Tab content — full width below */}
      {activeTab && (
        <div
          ref={contentRef}
          style={{
            border: "1px solid #E4E4E7",
            borderRadius: 12,
            background: "#fff",
            overflow: "hidden",
            animation: "slideDown 200ms ease",
          }}
        >
          <style>{`
            @keyframes slideDown {
              from { opacity: 0; max-height: 0; }
              to { opacity: 1; max-height: 1000px; }
            }
          `}</style>

          {activeTab === "agent" && (
            <AgentStep onBack={noop} onSkip={() => handleSkip("agent")} standalone />
          )}
          {activeTab === "local" && (
            <LocalCogneeStep onBack={noop} onSkip={() => handleSkip("local")} standalone />
          )}
          {activeTab === "database" && (
            <DatabaseStep onBack={noop} onSkip={() => handleSkip("database")} standalone />
          )}
        </div>
      )}
    </div>
  );
}
