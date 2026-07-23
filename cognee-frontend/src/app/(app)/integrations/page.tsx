"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { useCogniInstance } from "@/modules/tenant/TenantProvider";
import { TrackPageView } from "@/modules/analytics";
import {
  OPENCLAW_PROMPT,
  CLAUDE_MARKETPLACE_ADD, CLAUDE_PLUGIN_INSTALL,
  CODEX_HOOKS_ENABLE, CODEX_MARKETPLACE_ADD, CODEX_PLUGIN_INSTALL,
  MCP_STDIO_CONFIG, HERMES_MCP_CONFIG, GENERIC_SKILL_INSTALL, fillTemplate,
  UPLOAD_MEMORY_PROMPT, UPLOAD_SAMPLE_PROMPT, RECALL_SAMPLE_PROMPT,
} from "@/data/prompts";

// VS Code's MCP config uses a "servers" wrapper (not "mcpServers"). The API
// key is requested via an inputs promptString instead of being written into
// .vscode/mcp.json — that file is commonly committed, so a raw key would leak.
const VSCODE_MCP_CONFIG = `{
  "inputs": [
    {
      "type": "promptString",
      "id": "cognee-api-key",
      "description": "Cognee API key",
      "password": true
    }
  ],
  "servers": {
    "cognee": {
      "type": "stdio",
      "command": "uvx",
      "args": ["cognee-mcp"],
      "env": {
        "COGNEE_BASE_URL": "{{BASE_URL}}",
        "COGNEE_API_KEY": "\${input:cognee-api-key}"
      }
    }
  }
}`;

// ── Copy-on-click code block ──────────────────────────────────────────────

function InlineCodeBlock({ code, toCopy, loading }: { code: string; toCopy?: string; loading?: boolean }) {
  const [copied, setCopied] = useState(false);
  function doCopy() {
    if (loading) return;
    navigator.clipboard.writeText(toCopy ?? code);
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  }
  return (
    <div onClick={(e) => { e.stopPropagation(); doCopy(); }} style={{ background: "#18181B", borderRadius: 8, padding: "11px 14px", display: "flex", alignItems: "center", justifyContent: "space-between", gap: 10, cursor: loading ? "wait" : "pointer" }}>
      <pre style={{ margin: 0, fontSize: 12.5, fontFamily: 'ui-monospace, Menlo, Monaco, "Cascadia Mono", "Segoe UI Mono", "Roboto Mono", monospace', color: loading ? "rgba(237,236,234,0.45)" : "rgba(237,236,234,0.85)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis", flex: 1 }}>
        {loading ? "Loading…" : code}
      </pre>
      <button onClick={(e) => { e.stopPropagation(); doCopy(); }} aria-label={copied ? "Copied" : "Copy"} style={{ background: "none", border: "none", cursor: loading ? "wait" : "pointer", flexShrink: 0, padding: 2, borderRadius: 4 }}>
        {copied
          ? <svg width="14" height="14" viewBox="0 0 16 16" fill="none"><path d="M3.5 8.5L6.5 11.5L12.5 4.5" stroke="#22C55E" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" /></svg>
          : <svg width="14" height="14" viewBox="0 0 16 16" fill="none"><rect x="5" y="5" width="8" height="8" rx="1.5" stroke="#6B7280" strokeWidth="1.5" /><path d="M11 3H4.5A1.5 1.5 0 003 4.5V11" stroke="#6B7280" strokeWidth="1.5" strokeLinecap="round" /></svg>}
      </button>
    </div>
  );
}

// ── Agent config ──────────────────────────────────────────────────────────

interface StepDef {
  title: string;
  description: string;
  code?: string;
  codeToCopy?: string;
  loading?: boolean;
  /** When set, renders multiple separately-copyable code blocks (commands run one at a time). */
  codeBlocks?: { code: string; codeToCopy?: string; label?: string }[];
}

interface AgentCfg {
  key: string;
  name: string;
  cta: string;
  /** Large logo rendered at the bottom-right of the card (height ≈ 110px) */
  logo: React.ReactNode;
  /** Smaller icon shown in the modal header (24px) */
  icon: React.ReactNode;
  buildSteps: (baseUrl: string, apiKey: string, isInitializing: boolean) => StepDef[];
}

function imgLogo(src: string, alt: string) {
  return <img src={src} alt={alt} style={{ height: 110, width: "auto" }} draggable={false} />;
}
function imgIcon(src: string, alt: string) {
  return <img src={src} alt={alt} style={{ width: 24, height: 24, objectFit: "contain" }} />;
}

function credStep(baseUrl: string, apiKey: string, loading: boolean): StepDef {
  return {
    title: "Set your API credentials",
    description: "Open a terminal and run these commands to configure your Cognee endpoint and key.",
    code: `export COGNEE_BASE_URL="${baseUrl}"`,
    codeToCopy: `export COGNEE_BASE_URL="${baseUrl}"\nexport COGNEE_API_KEY="${apiKey}"`,
    loading,
  };
}
function mcpConfigStep(displayPath: string, app: string, dir: string, file: string, baseUrl: string, apiKey: string, loading: boolean): StepDef {
  return {
    title: `Configure ${app}`,
    description: `Run this command to write the config file (an existing one is backed up to .bak first), then restart ${app}. Cognee runs via uvx — no separate install needed (requires uv). If you already use other MCP servers in ${app}, merge the "cognee" entry into your existing config instead.`,
    code: displayPath,
    codeToCopy: `mkdir -p ${dir} && [ -f ${file} ] && cp ${file} ${file}.bak; cat > ${file} << 'COGNEE_EOF'\n${fillTemplate(MCP_STDIO_CONFIG, baseUrl, apiKey)}\nCOGNEE_EOF`,
    loading,
  };
}

// ── API large logo ────────────────────────────────────────────────────────

function ApiBigLogo() {
  return (
    <svg height="110" viewBox="0 0 90 110" fill="none" xmlns="http://www.w3.org/2000/svg">
      <rect x="5" y="20" width="80" height="50" rx="10" fill="#1a1a2e" stroke="rgba(255,255,255,0.15)" strokeWidth="2"/>
      <path d="M25 35L16 45L25 55" stroke="#BC9BFF" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"/>
      <path d="M65 35L74 45L65 55" stroke="#BC9BFF" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"/>
      <line x1="50" y1="30" x2="40" y2="60" stroke="rgba(255,255,255,0.4)" strokeWidth="2.5" strokeLinecap="round"/>
    </svg>
  );
}
function ApiIcon() {
  return <svg width="24" height="24" viewBox="0 0 24 24" fill="none"><rect x="3" y="6" width="18" height="12" rx="2" stroke="rgba(237,236,234,0.7)" strokeWidth="1.5"/><path d="M7 9L4 12L7 15" stroke="#BC9BFF" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/><path d="M17 9L20 12L17 15" stroke="#BC9BFF" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/><line x1="13" y1="8" x2="11" y2="16" stroke="rgba(237,236,234,0.5)" strokeWidth="1.5" strokeLinecap="round"/></svg>;
}

const AGENT_CARDS: AgentCfg[] = [
  {
    key: "claude-code",
    name: "Claude Code",
    cta: "Connect via plugin",
    logo: imgLogo("/visuals/logos/claude.svg", "Claude Code"),
    icon: imgIcon("/visuals/logos/claude.svg", "Claude Code"),
    buildSteps: (baseUrl, apiKey, loading) => [
      credStep(baseUrl, apiKey, loading),
      {
        title: "Install the Cognee plugin",
        description: "Run these in your terminal one at a time — register the Cognee marketplace, then install the memory plugin.",
        codeBlocks: [
          { code: CLAUDE_MARKETPLACE_ADD },
          { code: CLAUDE_PLUGIN_INSTALL },
        ],
      },
      {
        title: "Upload something to Cognee",
        description: "Pick one and paste it into Claude Code — it stores the content in your Cognee memory so you can recall it next.",
        codeBlocks: [
          { label: "Option A · Your existing memory", code: UPLOAD_MEMORY_PROMPT },
          { label: "Option B · Try it with a sample", code: UPLOAD_SAMPLE_PROMPT },
        ],
      },
      {
        title: "Recall it from Cognee",
        description: "First run /exit to close the session — that syncs it into Cognee Cloud — then reopen Claude Code and ask the question below. Answering from a fresh session proves it's recalling from your cloud memory.",
        codeBlocks: [
          { code: "/exit" },
          { code: RECALL_SAMPLE_PROMPT },
        ],
      },
      {
        title: "You're all set",
        description: "The Cognee plugin hooks into Claude Code's lifecycle — no curl or manual API calls — and captures your session as you work. When a session ends (e.g. /exit), it consolidates that session into your Cognee Cloud knowledge graph, and every new session automatically recalls it back. Sessions are disposable; your memory isn't.",
      },
    ],
  },
  {
    key: "codex",
    name: "Codex",
    cta: "Connect via plugin",
    logo: imgLogo("/visuals/logos/codex.svg", "Codex"),
    icon: imgIcon("/visuals/logos/codex.svg", "Codex"),
    buildSteps: (baseUrl, apiKey, loading) => [
      credStep(baseUrl, apiKey, loading),
      {
        title: "Install the Cognee plugin",
        description: "Run these in your terminal one at a time — enable Codex hooks, register the Cognee marketplace, then install the memory plugin.",
        codeBlocks: [
          { code: CODEX_HOOKS_ENABLE },
          { code: CODEX_MARKETPLACE_ADD },
          { code: CODEX_PLUGIN_INSTALL },
        ],
      },
      {
        title: "Upload something to Cognee",
        description: "Pick one and paste it into Codex — it stores the content in your Cognee memory so you can recall it next.",
        codeBlocks: [
          { label: "Option A · Your existing memory", code: UPLOAD_MEMORY_PROMPT },
          { label: "Option B · Try it with a sample", code: UPLOAD_SAMPLE_PROMPT },
        ],
      },
      {
        title: "Recall it from Cognee",
        description: "First run /exit to close the session — that syncs it into Cognee Cloud — then reopen Codex and ask the question below. Answering from a fresh session proves it's recalling from your cloud memory.",
        codeBlocks: [
          { code: "/exit" },
          { code: RECALL_SAMPLE_PROMPT },
        ],
      },
      {
        title: "You're all set",
        description: "The Cognee plugin hooks into Codex's lifecycle — no curl or manual API calls — and captures your session as you work. When a session ends (e.g. /exit), it consolidates that session into your Cognee Cloud knowledge graph, and every new session automatically recalls it back. Sessions are disposable; your memory isn't.",
      },
    ],
  },
  {
    key: "openclaw",
    name: "OpenClaw",
    cta: "Connect via prompts",
    logo: imgLogo("/visuals/logos/openclaw.svg", "OpenClaw"),
    icon: imgIcon("/visuals/logos/openclaw.svg", "OpenClaw"),
    buildSteps: (baseUrl, apiKey, loading) => [
      credStep(baseUrl, apiKey, loading),
      // OpenClaw only loads AGENTS.md from its workspace directory, not the project root.
      { title: "Create the workspace AGENTS.md", description: "Run this command to add the Cognee memory instructions to OpenClaw's workspace. An existing AGENTS.md is backed up to AGENTS.md.bak — merge it manually afterwards.", code: "~/.openclaw/workspace/AGENTS.md", codeToCopy: `mkdir -p ~/.openclaw/workspace && [ -f ~/.openclaw/workspace/AGENTS.md ] && cp ~/.openclaw/workspace/AGENTS.md ~/.openclaw/workspace/AGENTS.md.bak; cat > ~/.openclaw/workspace/AGENTS.md << 'COGNEE_EOF'\n${OPENCLAW_PROMPT}\nCOGNEE_EOF` },
      { title: "Test the connection", description: `Open OpenClaw and ask: "What do you know from cognee?" — if it responds with knowledge from your brain, you're connected.` },
    ],
  },
  {
    key: "claude-desktop",
    name: "Claude Desktop",
    cta: "Connect via MCP",
    logo: imgLogo("/visuals/logos/claude.svg", "Claude Desktop"),
    icon: imgIcon("/visuals/logos/claude.svg", "Claude Desktop"),
    buildSteps: (baseUrl, apiKey, loading) => [
      mcpConfigStep("claude_desktop_config.json", "Claude Desktop", '"$HOME/Library/Application Support/Claude"', '"$HOME/Library/Application Support/Claude/claude_desktop_config.json"', baseUrl, apiKey, loading),
      { title: "Test the connection", description: "Restart Claude Desktop. Open a new conversation and ask: \"What do you know from cognee?\" — Cognee's memory will respond." },
    ],
  },
  {
    key: "cursor",
    name: "Cursor",
    cta: "Connect via MCP",
    logo: imgLogo("/visuals/logos/cursor.svg", "Cursor"),
    icon: imgIcon("/visuals/logos/cursor.svg", "Cursor"),
    buildSteps: (baseUrl, apiKey, loading) => [
      mcpConfigStep("~/.cursor/mcp.json", "Cursor", "~/.cursor", "~/.cursor/mcp.json", baseUrl, apiKey, loading),
      { title: "Test the connection", description: "Restart Cursor. Open a project and ask the agent: \"What do you know from cognee?\" — you should get a memory-grounded response." },
    ],
  },
  {
    key: "hermes",
    name: "Hermes Agent",
    cta: "Connect via MCP",
    logo: imgLogo("/visuals/logos/hermes.svg", "Hermes Agent"),
    icon: imgIcon("/visuals/logos/hermes.svg", "Hermes Agent"),
    buildSteps: (baseUrl, apiKey, loading) => [
      { title: "Configure Hermes Agent", description: "Hermes reads YAML — add this block under mcp_servers in ~/.hermes/config.yaml and restart the agent. Cognee runs via uvx (requires uv) — no separate install.", code: "~/.hermes/config.yaml", codeToCopy: fillTemplate(HERMES_MCP_CONFIG, baseUrl, apiKey), loading },
      { title: "Test the connection", description: "Ask Hermes: \"What do you know from cognee?\" — it should use the Cognee memory tool and return a response from your brain." },
    ],
  },
  {
    key: "vscode",
    name: "VS Code",
    cta: "Connect via MCP",
    logo: imgLogo("/visuals/logos/vscode.svg", "VS Code"),
    icon: imgIcon("/visuals/logos/vscode.svg", "VS Code"),
    buildSteps: (baseUrl, apiKey, loading) => [
      { title: "Configure VS Code", description: "From your project root, run this command to register the Cognee MCP server for VS Code (Copilot agent mode), then reload the window. Cognee runs via uvx (requires uv) — no separate install. VS Code prompts for your API key on first use (kept out of the committable file); an existing mcp.json is backed up to .bak.", code: ".vscode/mcp.json", codeToCopy: `mkdir -p .vscode && [ -f .vscode/mcp.json ] && cp .vscode/mcp.json .vscode/mcp.json.bak; cat > .vscode/mcp.json << 'COGNEE_EOF'\n${fillTemplate(VSCODE_MCP_CONFIG, baseUrl, apiKey)}\nCOGNEE_EOF`, loading },
      { title: "Test the connection", description: "Open Copilot Chat in agent mode and ask: \"What do you know from cognee?\" — the Cognee memory tools should respond." },
    ],
  },
  {
    key: "gemini-cli",
    name: "Gemini CLI",
    cta: "Connect via MCP",
    logo: imgLogo("/visuals/logos/gemini.svg", "Gemini CLI"),
    icon: imgIcon("/visuals/logos/gemini.svg", "Gemini CLI"),
    buildSteps: (baseUrl, apiKey, loading) => [
      mcpConfigStep("~/.gemini/settings.json", "Gemini CLI", "~/.gemini", "~/.gemini/settings.json", baseUrl, apiKey, loading),
      { title: "Test the connection", description: "Run `gemini` in your terminal and ask: \"What do you know from cognee?\" — Cognee's memory tool will respond." },
    ],
  },
  {
    key: "cline",
    name: "Cline",
    cta: "Connect via MCP",
    logo: imgLogo("/visuals/logos/cline.svg", "Cline"),
    icon: imgIcon("/visuals/logos/cline.svg", "Cline"),
    buildSteps: (baseUrl, apiKey, loading) => [
      { title: "Configure Cline", description: "In VS Code, open the Cline sidebar → MCP Servers → Configure MCP Servers — paste this JSON block and save. Cognee runs via uvx (requires uv) — no separate install.", code: '{ "mcpServers": { "cognee": … } }', codeToCopy: fillTemplate(MCP_STDIO_CONFIG, baseUrl, apiKey), loading },
      { title: "Test the connection", description: "Ask Cline: \"What do you know from cognee?\" — it should use the Cognee memory tool and return a response from your brain." },
    ],
  },
  {
    key: "https-api",
    name: "API / MCP",
    cta: "Connect via API or MCP",
    logo: <ApiBigLogo />,
    icon: <ApiIcon />,
    buildSteps: (baseUrl, apiKey, loading) => [
      credStep(baseUrl, apiKey, loading),
      {
        title: "Query the REST API",
        description: "Send a recall query to your Cognee endpoint from any HTTP client or language.",
        code: `curl -X POST ${baseUrl}/api/v1/recall`,
        codeToCopy: fillTemplate(
          'curl -X POST {{BASE_URL}}/api/v1/recall \\\n  -H "X-Api-Key: {{API_KEY}}" \\\n  -H "Content-Type: application/json" \\\n  -d \'{"query": "What are the main entities?"}\'',
          baseUrl, apiKey,
        ),
        loading,
      },
      {
        title: "Or install the Cognee skill",
        description: "Prefer skills? Run this command from your project root to create the skill file, then point your agent at it (skills directory, instructions file, or system prompt). The skill teaches your agent to call the Cognee API using the credentials from step 1.",
        code: "skills/cognee/SKILL.md",
        codeToCopy: GENERIC_SKILL_INSTALL,
      },
      { title: "Test the connection", description: "Ask your agent: \"What do you know from cognee?\" — Cognee's memory tool should respond with knowledge from your brain." },
    ],
  },
];

// ── Automation platforms ──────────────────────────────────────────────────

const AUTOMATION_CARDS: AgentCfg[] = [
  {
    key: "n8n",
    name: "n8n",
    cta: "Connect via node",
    // Wide icon+wordmark logo: margins compensate the card's overflow offsets so it stays fully visible.
    logo: <img src="/visuals/logos/n8n.svg" alt="n8n" style={{ height: 74, width: "auto", marginBottom: 62, marginRight: 42 }} draggable={false} />,
    icon: imgIcon("/visuals/logos/n8n.svg", "n8n"),
    buildSteps: (baseUrl, apiKey, loading) => [
      { title: "Install the Cognee community node", description: "In n8n, open Settings → Community Nodes → Install and enter the package name.", code: "n8n-nodes-cognee", codeToCopy: "n8n-nodes-cognee" },
      { title: "Create the Cognee API credential", description: "Add a \"Cognee API\" credential, paste your Base URL and API key, and click Test — it verifies the connection against your tenant.", code: baseUrl, codeToCopy: `${baseUrl}\n${apiKey}`, loading },
      { title: "Test the connection", description: "Add a Cognee node to a workflow (or attach it to an AI Agent) and run it — it should answer from your brain." },
    ],
  },
  {
    key: "dify",
    name: "Dify",
    cta: "Connect via plugin",
    // Wide wordmark logo: margins compensate the card's overflow offsets so it stays fully visible.
    logo: <img src="/visuals/logos/dify.svg" alt="Dify" style={{ height: 44, width: "auto", marginBottom: 77, marginRight: 42 }} draggable={false} />,
    icon: imgIcon("/visuals/logos/dify.svg", "Dify"),
    buildSteps: (baseUrl, apiKey, loading) => [
      { title: "Install the Cognee plugin", description: "In your Dify workspace, open the Marketplace, search for \"Cognee\" (by topoteretes), and click Install.", code: "marketplace.dify.ai/plugin/topoteretes/cognee", codeToCopy: "https://marketplace.dify.ai/plugin/topoteretes/cognee" },
      // The Dify plugin needs the /api suffix — without it, validation passes (root /health) but every tool call 404s.
      { title: "Configure the plugin", description: "Open the plugin's authorization settings and enter your Cognee Base URL (including the /api suffix) and API key.", code: `${baseUrl}/api`, codeToCopy: `${baseUrl}/api\n${apiKey}`, loading },
      { title: "Add Cognee tools to your app", description: "In an Agent or Workflow app, add the Cognee tools — create a dataset, ingest text or files, run Cognify, then search your memory." },
      { title: "Test the connection", description: "Run the app and ask: \"What do you know from cognee?\" — the Cognee search tool should answer from your brain." },
    ],
  },
];

// ── Agent section component ───────────────────────────────────────────────

function AgentSection({ cards }: { cards: AgentCfg[] }) {
  const router = useRouter();
  const { serviceUrl, apiKey, isInitializing } = useCogniInstance();
  const [activeKey, setActiveKey] = useState<string | null>(null);
  const [stepIndexMap, setStepIndexMap] = useState<Partial<Record<string, number>>>({});

  const baseUrl = serviceUrl || "https://your-tenant.aws.cognee.ai";
  const resolvedKey = apiKey || "your-api-key";

  const activeCfg = cards.find(c => c.key === activeKey);
  const activeSteps = activeCfg ? activeCfg.buildSteps(baseUrl, resolvedKey, isInitializing) : [];
  const currentStep = activeKey ? (stepIndexMap[activeKey] ?? 0) : 0;

  function openCard(key: string) {
    setActiveKey(key);
    if (stepIndexMap[key] === undefined) setStepIndexMap(s => ({ ...s, [key]: 0 }));
  }

  return (
    <>
      <style>{`
        @keyframes aci-check { 0%{transform:scale(0.4);opacity:0} 100%{transform:scale(1);opacity:1} }
        @keyframes aci-popup { 0%{opacity:0;transform:scale(0.97) translateY(6px)} 100%{opacity:1;transform:scale(1) translateY(0)} }
        .aci-card-logo { transition: transform 300ms ease; }
        .aci-card:hover .aci-card-logo { transform: scale(1.1); }
        .aci-card:hover .aci-cta-chip { background: rgba(101,16,244,0.85) !important; }
        .aci-step-row:hover { background: rgba(255,255,255,0.04); }
        .aci-step-row[data-active="true"]:hover { background: transparent; }
        .int-agent-grid {
          display: grid;
          grid-template-columns: repeat(5, minmax(0, 1fr));
          gap: 16px;
        }
        @media (max-width: 1100px) { .int-agent-grid { grid-template-columns: repeat(4, minmax(0, 1fr)); } }
        @media (max-width: 800px)  { .int-agent-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); } }
        @media (max-width: 560px)  { .int-agent-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
      `}</style>

      <div className="int-agent-grid">
        {cards.map((card) => {
          const isActive = activeKey === card.key;
          return (
            <button
              key={card.key}
              className="aci-card"
              onClick={() => openCard(card.key)}
              style={{
                position: "relative",
                background: isActive ? "rgba(188,155,255,0.20)" : "rgba(255,255,255,0.06)",
                backdropFilter: "blur(12px)",
                border: `1px solid ${isActive ? "rgba(188,155,255,0.35)" : "rgba(255,255,255,0.1)"}`,
                borderRadius: 12,
                padding: "18px 14px 0 14px",
                height: 160,
                overflow: "hidden",
                cursor: "pointer",
                textAlign: "left",
                display: "flex",
                flexDirection: "column",
                transition: "border-color 150ms, background 150ms",
              }}
            >
              {/* Name */}
              <span style={{ fontSize: 15, fontWeight: 300, color: "#EDECEA", lineHeight: 1.25, letterSpacing: "-0.01em", fontFamily: '"TWKLausanne", sans-serif' }}>
                {card.name}
              </span>

              {/* CTA chip */}
              <div style={{ position: "absolute", bottom: 14, left: 14, zIndex: 1 }}>
                <span className="aci-cta-chip" style={{ display: "inline-flex", alignItems: "center", gap: 4, background: "rgba(20,20,22,0.92)", backdropFilter: "blur(12px)", WebkitBackdropFilter: "blur(12px)", border: "1px solid rgba(237,236,234,0.65)", borderRadius: 6, padding: "4px 9px", fontSize: 11, fontWeight: 500, color: "rgba(237,236,234,0.65)", whiteSpace: "nowrap", transition: "background 150ms" }}>
                  {card.cta}
                </span>
              </div>

              {/* Large overflowing logo */}
              <div className="aci-card-logo" style={{ position: "absolute", bottom: -18, right: -28, pointerEvents: "none" }}>
                {card.logo}
              </div>
            </button>
          );
        })}
      </div>

      {/* Popup modal */}
      {activeKey && activeCfg && (
        <div role="dialog" aria-modal="true" style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.35)", backdropFilter: "blur(4px)", WebkitBackdropFilter: "blur(4px)", zIndex: 200, display: "flex", alignItems: "center", justifyContent: "center", padding: 16 }} onClick={() => setActiveKey(null)}>
          <div className="aci-popup" onClick={(e) => e.stopPropagation()} style={{ background: "rgba(15,15,15,0.92)", backdropFilter: "blur(16px)", borderRadius: 14, width: 520, maxWidth: "100%", boxShadow: "0 20px 60px rgba(0,0,0,0.6), 0 0 0 1px rgba(255,255,255,0.1)", overflow: "hidden", animation: "aci-popup 200ms cubic-bezier(0.22,1,0.36,1) forwards" }}>

            {/* Header */}
            <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "16px 20px", borderBottom: "1px solid rgba(255,255,255,0.1)" }}>
              <div style={{ width: 24, height: 24, flexShrink: 0, display: "flex", alignItems: "center", justifyContent: "center" }}>{activeCfg.icon}</div>
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ fontSize: 15, fontWeight: 700, color: "#EDECEA", lineHeight: "20px" }}>Connect {activeCfg.name}</div>
                <div style={{ fontSize: 12, color: "rgba(237,236,234,0.45)", marginTop: 1 }}>Step {currentStep + 1} of {activeSteps.length}</div>
              </div>
              <button onClick={() => setActiveKey(null)} style={{ background: "none", border: "none", color: "rgba(237,236,234,0.65)", cursor: "pointer", padding: 4, borderRadius: 6, lineHeight: 1 }}>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
              </button>
            </div>

            {/* Steps */}
            {activeSteps.map((step, i) => {
              const isStepActive = currentStep === i;
              const isDone = i < currentStep;
              return (
                <div key={i} className="aci-step-row" data-active={isStepActive ? "true" : undefined} onClick={() => setStepIndexMap(p => ({ ...p, [activeKey]: i }))} style={{ borderBottom: i < activeSteps.length - 1 ? "1px solid rgba(255,255,255,0.07)" : "none", cursor: isStepActive ? "default" : "pointer" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 12, padding: isStepActive ? "14px 20px 0" : "14px 20px" }}>
                    <div style={{ width: 24, height: 24, borderRadius: "50%", flexShrink: 0, display: "flex", alignItems: "center", justifyContent: "center", background: isDone ? "#DCFCE7" : isStepActive ? "#6510F4" : "#F4F4F5", transition: "background 200ms" }}>
                      {isDone
                        ? <svg width="10" height="10" viewBox="0 0 16 16" fill="none" style={{ animation: "aci-check 220ms cubic-bezier(0.22,1,0.36,1) forwards" }}><path d="M3 8.5L6.5 12L13 5" stroke="#16A34A" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round"/></svg>
                        : <span style={{ fontSize: 11, fontWeight: 700, color: isStepActive ? "#fff" : "#A1A1AA", lineHeight: 1 }}>{i + 1}</span>}
                    </div>
                    <span style={{ flex: 1, fontSize: 14, fontWeight: isStepActive ? 500 : 400, color: isDone ? "rgba(237,236,234,0.4)" : isStepActive ? "#EDECEA" : "rgba(237,236,234,0.35)" }}>
                      {step.title}
                    </span>
                    {isDone && <span style={{ fontSize: 10, fontWeight: 700, letterSpacing: "0.06em", textTransform: "uppercase", background: "#DCFCE7", color: "#15803D", borderRadius: 100, padding: "2px 8px", flexShrink: 0 }}>Done</span>}
                  </div>
                  <div style={{ display: "grid", gridTemplateRows: isStepActive ? "1fr" : "0fr", opacity: isStepActive ? 1 : 0, transition: "grid-template-rows 260ms ease, opacity 200ms ease" }}>
                    <div style={{ overflow: "hidden" }}>
                      <div onClick={(e) => e.stopPropagation()} style={{ padding: "10px 20px 18px 56px" }}>
                        {step.description && <p style={{ fontSize: 13, color: "rgba(237,236,234,0.55)", margin: "0 0 12px", lineHeight: 1.6 }}>{step.description}</p>}
                        {step.code && <InlineCodeBlock code={step.code} toCopy={step.codeToCopy} loading={step.loading} />}
                        {step.codeBlocks && (
                          <div style={{ display: "flex", flexDirection: "column", gap: step.codeBlocks.some(cb => cb.label) ? 14 : 8 }}>
                            {step.codeBlocks.map((cb, j) => (
                              cb.label ? (
                                <div key={j}>
                                  <div style={{ fontSize: 13, fontWeight: 600, color: "#EDECEA", marginBottom: 6 }}>{cb.label}</div>
                                  <InlineCodeBlock code={cb.code} toCopy={cb.codeToCopy} />
                                </div>
                              ) : (
                                <InlineCodeBlock key={j} code={cb.code} toCopy={cb.codeToCopy} />
                              )
                            ))}
                          </div>
                        )}
                        {i < activeSteps.length - 1
                          ? <p style={{ margin: "10px 0 0", fontSize: 12, color: "#C8C8C8" }}>Click step {i + 2} when ready ↓</p>
                          : <button onClick={(e) => { e.stopPropagation(); setActiveKey(null); router.push("/sessions"); }} style={{ marginTop: 12, display: "inline-flex", alignItems: "center", gap: 5, background: "none", border: "1px solid rgba(255,255,255,0.15)", borderRadius: 8, padding: "7px 14px", fontSize: 13, fontWeight: 500, color: "rgba(237,236,234,0.7)", fontFamily: "inherit", cursor: "pointer" }}>Go to Sessions →</button>}
                      </div>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </>
  );
}

// ── Data source integrations ──────────────────────────────────────────────

const INTEGRATIONS = [
  { name: "Slack",               description: "Turn channels and threads into searchable memory.",   initials: "Sl", color: "#4A154B" },
  { name: "Notion",              description: "Sync pages and databases into your knowledge graph.", initials: "No", color: "#000000" },
  { name: "Google Drive",        description: "Ingest Docs, Sheets, and Slides automatically.",      initials: "GD", color: "#1A73E8" },
  { name: "GitHub",              description: "Index issues, PRs, and repository docs.",             initials: "GH", color: "#181717" },
  { name: "Linear",              description: "Bring issues and project context into memory.",       initials: "Li", color: "#5E6AD2" },
  { name: "Confluence",          description: "Connect spaces and wikis as a data source.",          initials: "Cf", color: "#172B4D" },
  { name: "Jira",                description: "Sync tickets and epics into the graph.",              initials: "Jr", color: "#0052CC" },
  { name: "Google Drive Sheets", description: "Pull structured data from spreadsheets.",            initials: "Sh", color: "#0F9D58" },
];

// ── Page ──────────────────────────────────────────────────────────────────

export default function IntegrationsPage() {
  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", overflowY: "auto" }}>
      <TrackPageView page="Integrations" />

      <div style={{ padding: "24px 32px 40px", display: "flex", flexDirection: "column", gap: 40 }}>

        {/* ── Agents ── */}
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <div>
            <h1 style={{ fontSize: 18, fontWeight: 700, color: "#EDECEA", margin: "0 0 4px", letterSpacing: "-0.01em" }}>Agents</h1>
            <p style={{ fontSize: 14, color: "rgba(237,236,234,0.55)", margin: 0 }}>Connect your AI agents and coding tools to Cognee for persistent memory.</p>
          </div>
          <AgentSection cards={AGENT_CARDS} />
        </div>

        {/* ── Divider ── */}
        <div style={{ height: 1, background: "rgba(255,255,255,0.08)" }} />

        {/* ── Automation platforms ── */}
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <div>
            <h2 style={{ fontSize: 18, fontWeight: 700, color: "#EDECEA", margin: "0 0 4px", letterSpacing: "-0.01em" }}>Automation platforms</h2>
            <p style={{ fontSize: 14, color: "rgba(237,236,234,0.55)", margin: 0 }}>Give your automation workflows access to Cognee memory via MCP.</p>
          </div>
          <AgentSection cards={AUTOMATION_CARDS} />
        </div>

        {/* ── Divider ── */}
        <div style={{ height: 1, background: "rgba(255,255,255,0.08)" }} />

        {/* ── Data sources ── */}
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 4 }}>
              <h2 style={{ fontSize: 18, fontWeight: 700, color: "#EDECEA", margin: 0, letterSpacing: "-0.01em" }}>Data sources</h2>
              <span style={{ background: "rgba(188,155,255,0.20)", color: "#BC9BFF", fontSize: 11, fontWeight: 700, padding: "2px 10px", borderRadius: 999 }}>Coming soon</span>
            </div>
            <p style={{ fontSize: 14, color: "rgba(237,236,234,0.55)", margin: 0 }}>Connect the tools your team already uses as data sources for your brains.</p>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))", gap: 16 }}>
            {INTEGRATIONS.map((it) => (
              <div key={it.name} style={{ background: "rgba(255,255,255,0.06)", backdropFilter: "blur(12px)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 12, padding: 20, display: "flex", flexDirection: "column", gap: 12 }}>
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 12, minWidth: 0 }}>
                    <div style={{ width: 40, height: 40, borderRadius: 10, background: it.color, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                      <span style={{ color: "#fff", fontSize: 14, fontWeight: 700 }}>{it.initials}</span>
                    </div>
                    <span style={{ fontSize: 16, fontWeight: 500, color: "#EDECEA", fontFamily: '"TWKLausanne", sans-serif', overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{it.name}</span>
                  </div>
                  <span style={{ flexShrink: 0, background: "rgba(255,255,255,0.06)", color: "rgba(237,236,234,0.35)", fontSize: 11, fontWeight: 500, padding: "2px 8px", borderRadius: 999 }}>Coming soon</span>
                </div>
                <p style={{ fontSize: 13, color: "rgba(237,236,234,0.55)", margin: 0 }}>{it.description}</p>
                <button disabled style={{ alignSelf: "flex-start", background: "rgba(255,255,255,0.06)", color: "rgba(237,236,234,0.35)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 8, padding: "6px 14px", fontSize: 13, fontWeight: 500, cursor: "not-allowed" }}>Notify me</button>
              </div>
            ))}
          </div>

          <p style={{ fontSize: 13, color: "rgba(237,236,234,0.35)", textAlign: "center", margin: 0 }}>
            More integrations on the way.{" "}
            <a href="mailto:support@cognee.ai?subject=Integration%20request" style={{ color: "#6510F4", textDecoration: "underline" }}>Let us know</a> what to prioritize.
          </p>
        </div>

      </div>
    </div>
  );
}
