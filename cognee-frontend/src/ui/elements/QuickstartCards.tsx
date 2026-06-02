"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { CLAUDE_PROMPT, CODEX_PROMPT, OPENCLAW_PROMPT, MCP_SERVER_COMMAND, MCP_CLIENT_CONFIG, SKILLS_CONTENT } from "@/app/(app)/connect-agent/prompts";
import { useFilter } from "@/ui/layout/FilterContext";

function CompanyDataIcon() {
  return (
    <svg height="110" viewBox="0 0 80 100" fill="none" xmlns="http://www.w3.org/2000/svg">
      {/* Back sheet */}
      <rect x="16" y="6" width="54" height="70" rx="6" fill="#D4D4D8" stroke="#71717A" strokeWidth="3.5"/>
      {/* Middle sheet */}
      <rect x="8" y="14" width="54" height="70" rx="6" fill="#E4E4E7" stroke="#71717A" strokeWidth="3.5"/>
      {/* Front sheet */}
      <rect x="2" y="22" width="54" height="70" rx="6" fill="#F4F4F5" stroke="#52525B" strokeWidth="3.5"/>
      {/* Folded corner */}
      <path d="M38 22v16h18" stroke="#52525B" strokeWidth="3.5" strokeLinecap="round" strokeLinejoin="round"/>
      {/* Text lines */}
      <line x1="12" y1="52" x2="46" y2="52" stroke="#52525B" strokeWidth="3" strokeLinecap="round"/>
      <line x1="12" y1="63" x2="46" y2="63" stroke="#52525B" strokeWidth="3" strokeLinecap="round"/>
      <line x1="12" y1="74" x2="30" y2="74" stroke="#52525B" strokeWidth="3" strokeLinecap="round"/>
    </svg>
  );
}

const CARDS: {
  key: string;
  name: string;
  connected: boolean;
  description: string;
  steps: string[];
  prompt: string;
  logo: React.ReactNode;
  logoRight: number;
  noPanel?: boolean;
  href?: string;
}[] = [
  {
    key: "company-data",
    name: "Company brain",
    connected: false,
    description: "Your uploaded documents are processed into a knowledge graph and made available for retrieval. Upload files in Datasets to activate.",
    steps: [],
    prompt: "",
    logo: <CompanyDataIcon />,
    logoRight: -12,
    noPanel: true,
    href: "/datasets",
  },
  {
    key: "agents",
    name: "Claude & Codex",
    connected: false,
    description: "Give Claude and Codex persistent memory across sessions by pasting the prompt below into your CLAUDE.md, AGENTS.md, or project system instructions.",
    steps: [
      "Set your credentials in the terminal (export COGNEE_BASE_URL and COGNEE_API_KEY)",
      "Paste the prompt below into your CLAUDE.md or AGENTS.md file",
      "Claude and Codex will use Cognee to store and retrieve knowledge across sessions",
    ],
    prompt: `# For Claude (CLAUDE.md / project system instructions)\n${CLAUDE_PROMPT}\n\n---\n\n# For Codex (AGENTS.md / system instructions)\n${CODEX_PROMPT}`,
    logo: (
      <div style={{ display: "flex", alignItems: "flex-end" }}>
        <img src="/visuals/logos/claude.svg" alt="Claude" style={{ height: 88, width: "auto", marginRight: -20, position: "relative", zIndex: 1 }} />
        <img src="/visuals/logos/codex.svg" alt="Codex" style={{ height: 110, width: "auto" }} />
      </div>
    ),
    logoRight: -36,
  },
  {
    key: "openclaw",
    name: "OpenClaw",
    connected: false,
    description: "Connect Cognee to OpenClaw by adding instructions to your AGENTS.md or agent configuration.",
    steps: [
      "Set your credentials in ~/.openclaw/.env or export in your terminal",
      "Paste the prompt below into your AGENTS.md file",
      "OpenClaw will use Cognee for persistent memory across sessions",
    ],
    prompt: OPENCLAW_PROMPT,
    logo: <img src="/visuals/logos/openclaw.svg" alt="OpenClaw" style={{ height: 110, width: "auto" }} />,
    logoRight: -36,
  },
  {
    key: "skills",
    name: "Skills",
    connected: false,
    description: "Use the Cognee Python SDK to add memory skills directly to your agent code.",
    steps: [
      "Set your credentials in the terminal (export COGNEE_BASE_URL and COGNEE_API_KEY)",
      "Save the content below as .claude/skills/cognee-cloud/SKILL.md in your project",
      "Claude Code will use it to connect to Cognee Cloud for persistent memory",
    ],
    prompt: SKILLS_CONTENT,
    logo: <img src="/visuals/logos/skills.svg" alt="Skills" style={{ height: 110, width: "auto" }} />,
    logoRight: -12,
  },
  {
    key: "mcp",
    name: "MCP",
    connected: false,
    description: "Expose Cognee as an MCP server so any MCP-compatible client can use it as a memory tool.",
    steps: [
      "Set your credentials in the terminal (export COGNEE_BASE_URL and COGNEE_API_KEY)",
      "Install cognee-mcp and start the server (see setup commands below)",
      "Add the client config to your Claude Desktop, Cursor, or VS Code MCP settings",
    ],
    prompt: `# Setup commands\n${MCP_SERVER_COMMAND}\n\n# Client config (claude_desktop_config.json)\n${MCP_CLIENT_CONFIG}`,
    logo: <img src="/visuals/logos/mcp.svg" alt="MCP" style={{ height: 110, width: "auto" }} />,
    logoRight: -12,
  },
];

export default function QuickstartCards() {
  const router = useRouter();
  const { datasets } = useFilter();
  const hasDocuments = datasets.length > 0;

  const [activeKey, setActiveKey] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  const activeCard = CARDS.find((c) => c.key === activeKey) ?? null;

  function handleCardClick(key: string) {
    const card = CARDS.find((c) => c.key === key);
    if (card?.href) { router.push(card.href); return; }
    if (card?.noPanel) return;
    setActiveKey((prev) => (prev === key ? null : key));
    setCopied(false);
  }

  function isConnected(card: typeof CARDS[number]) {
    if (card.key === "company-data") return hasDocuments;
    return card.connected;
  }

  function handleCopy(text: string) {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  }

  return (
    <>
      <style>{`
        .qs-logo { transition: transform 300ms ease, filter 300ms ease; filter: brightness(0) invert(1) brightness(0.847); }
        .qs-card:not(.qs-connected):hover .qs-logo { transform: scale(1.2); filter: none; }
        .qs-connected .qs-logo { filter: none; }
      `}</style>

      {/* Cards row */}
      <div style={{ display: "flex", gap: 16 }}>
        {CARDS.map((card) => {
          const connected = isConnected(card);
          const isActive = activeKey === card.key;
          const bg = "#fff";
          const borderColor = isActive ? "#6510F4" : "#E4E4E7";
          const nameColor = isActive ? "#18181B" : connected ? "#18181B" : "#A1A1AA";

          return (
            <button
              key={card.key}
              onClick={() => handleCardClick(card.key)}
              className={`cursor-pointer qs-card${connected ? " qs-connected" : ""}`}
              style={{
                flex: 1,
                minWidth: 0,
                position: "relative",
                background: bg,
                border: `1px solid ${borderColor}`,
                borderRadius: 12,
                padding: "20px 16px 0 16px",
                height: 160,
                overflow: "hidden",
                cursor: "pointer",
                textAlign: "left",
                display: "flex",
                flexDirection: "column",
                transition: "border-color 150ms",
              }}
            >
              {/* Status badge */}
              <div style={{ position: "absolute", top: 12, right: 12, display: "flex", alignItems: "center", gap: 4, zIndex: 1 }}>
                <span style={{ width: 6, height: 6, borderRadius: "50%", background: connected ? "#22C55E" : "#D4D4D8", flexShrink: 0 }} />
                <span style={{ fontSize: 11, fontWeight: 500, color: connected ? "#16A34A" : "#A1A1AA", fontFamily: '"Inter", system-ui, sans-serif', whiteSpace: "nowrap" }}>
                  {connected ? "active" : "not active"}
                </span>
              </div>

              {/* Name */}
              <span style={{
                fontSize: 16,
                fontWeight: 300,
                color: nameColor,
                lineHeight: 1.25,
                letterSpacing: "-0.01em",
                fontFamily: '"TWK Lausanne", system-ui, sans-serif',
                transition: "color 150ms",
              }}>
                {card.name}
              </span>

              <div className="qs-logo" style={{ position: "absolute", bottom: -18, right: card.logoRight, pointerEvents: "none", filter: isActive ? "none" : undefined, transform: isActive ? "scale(1.2)" : undefined }}>
                {card.logo}
              </div>
            </button>
          );
        })}
      </div>

      {/* Instructions panel — appears on click */}
      <div style={{
        display: "grid",
        gridTemplateRows: activeCard ? "1fr" : "0fr",
        transition: "grid-template-rows 200ms ease",
      }}>
        <div style={{ overflow: "hidden" }}>
          {activeCard && (
            <div
              style={{
                marginTop: 8,
                background: "#fff",
                border: "1px solid #E4E4E7",
                borderRadius: 12,
                padding: "20px 24px",
                display: "flex",
                gap: 32,
                boxShadow: "0 4px 16px rgba(0,0,0,0.06)",
              }}
            >
              {/* Left: steps */}
              <div style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column", gap: 12 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <span style={{ fontSize: 14, fontWeight: 600, color: "#18181B", fontFamily: '"TWK Lausanne", system-ui, sans-serif' }}>
                    Connect to {activeCard.name}
                  </span>
                </div>
                <p style={{ fontSize: 13, color: "#71717A", margin: 0, lineHeight: 1.5 }}>{activeCard.description}</p>
                <ol style={{ margin: 0, paddingLeft: 18, display: "flex", flexDirection: "column", gap: 6 }}>
                  {activeCard.steps.map((step, i) => (
                    <li key={i} style={{ fontSize: 13, color: "#3F3F46", lineHeight: 1.5 }}>{step}</li>
                  ))}
                </ol>
              </div>

              {/* Right: code block */}
              <div style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column", gap: 8 }}>
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                  <span style={{ fontSize: 11, fontWeight: 600, color: "#71717A", textTransform: "uppercase", letterSpacing: "0.08em" }}>Prompt / Config</span>
                  <button
                    onClick={() => handleCopy(activeCard.prompt)}
                    className="cursor-pointer"
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: 5,
                      background: copied ? "#F0EDFF" : "#F4F4F5",
                      border: "none",
                      borderRadius: 6,
                      padding: "4px 10px",
                      fontSize: 11,
                      fontWeight: 500,
                      color: copied ? "#6510F4" : "#52525B",
                      fontFamily: "inherit",
                      transition: "background 150ms, color 150ms",
                    }}
                  >
                    {copied ? (
                      <svg width="11" height="11" viewBox="0 0 16 16" fill="none"><path d="M3.5 8.5L6.5 11.5L12.5 4.5" stroke="#6510F4" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" /></svg>
                    ) : (
                      <svg width="11" height="11" viewBox="0 0 16 16" fill="none"><rect x="5" y="5" width="8" height="8" rx="1.5" stroke="#52525B" strokeWidth="1.5" /><path d="M11 3H4.5A1.5 1.5 0 003 4.5V11" stroke="#52525B" strokeWidth="1.5" strokeLinecap="round" /></svg>
                    )}
                    {copied ? "Copied!" : "Copy"}
                  </button>
                </div>
                <pre style={{
                  margin: 0,
                  background: "#FAFAF9",
                  border: "1px solid #F0F0EF",
                  borderRadius: 8,
                  padding: "12px 14px",
                  fontSize: 11.5,
                  fontFamily: '"Fira Code", "Cascadia Code", "Consolas", monospace',
                  color: "#18181B",
                  overflowX: "auto",
                  lineHeight: 1.6,
                  whiteSpace: "pre-wrap",
                  wordBreak: "break-word",
                  maxHeight: 300,
                }}>
                  {activeCard.prompt}
                </pre>
              </div>
            </div>
          )}
        </div>
      </div>
    </>
  );
}
