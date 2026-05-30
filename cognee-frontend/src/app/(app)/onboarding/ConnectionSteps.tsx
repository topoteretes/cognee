"use client";

import React, { useState } from "react";
import Link from "next/link";
import { trackEvent } from "@/modules/analytics";

// ── Shared components ──

function StepBadge({ step, total = 4 }: { step: number; total?: number }) {
  return (
    <div style={{ backgroundColor: "#F0EDFF", borderRadius: 12, paddingBlock: 4, paddingInline: 12 }}>
      <span style={{ color: "#6510F4", fontFamily: '"Inter", system-ui, sans-serif', fontSize: 14, fontWeight: 500, lineHeight: "20px" }}>
        Step {step} of {total}
      </span>
    </div>
  );
}

function NumberCircle({ n }: { n: number }) {
  return (
    <div style={{ alignItems: "center", backgroundColor: "#F0EDFF", borderRadius: "50%", display: "flex", flexShrink: 0, height: 28, justifyContent: "center", width: 28 }}>
      <span style={{ color: "#6510F4", fontFamily: '"Inter", system-ui, sans-serif', fontSize: 13, fontWeight: 600 }}>{n}</span>
    </div>
  );
}

function Divider() {
  return <div style={{ backgroundColor: "#E4E4E7", flexShrink: 0, height: 1 }} />;
}

function colorizeLine(line: string): React.ReactNode {
  if (!line.trim()) return "\u00A0";
  // Comments
  if (line.trimStart().startsWith("#") || line.trimStart().startsWith("//")) {
    return <span style={{ color: "#6B7280" }}>{line}</span>;
  }
  // Try to highlight keywords, strings, and the rest
  const parts: React.ReactNode[] = [];
  let remaining = line;
  let key = 0;
  const pattern = /(import |from |await |async |def |for |if |return |export |pip install |curl |cognee\.\w+|"[^"]*"|'[^']*')/g;
  let match;
  let lastIndex = 0;
  while ((match = pattern.exec(remaining)) !== null) {
    if (match.index > lastIndex) {
      parts.push(<span key={key++} style={{ color: "#E4E4E7" }}>{remaining.slice(lastIndex, match.index)}</span>);
    }
    const token = match[0];
    if (token.startsWith('"') || token.startsWith("'")) {
      parts.push(<span key={key++} style={{ color: "#A5D6A7" }}>{token}</span>);
    } else if (token.startsWith("cognee.")) {
      parts.push(<span key={key++} style={{ color: "#CE93D8" }}>{token}</span>);
    } else if (["import ", "from ", "await ", "async ", "def ", "for ", "if ", "return ", "export "].includes(token)) {
      parts.push(<span key={key++} style={{ color: "#90CAF9" }}>{token}</span>);
    } else if (token.startsWith("pip ") || token.startsWith("curl ")) {
      parts.push(<span key={key++} style={{ color: "#FFD54F" }}>{token}</span>);
    } else {
      parts.push(<span key={key++} style={{ color: "#E4E4E7" }}>{token}</span>);
    }
    lastIndex = match.index + token.length;
  }
  if (lastIndex < remaining.length) {
    parts.push(<span key={key++} style={{ color: "#E4E4E7" }}>{remaining.slice(lastIndex)}</span>);
  }
  return parts.length > 0 ? parts : <span style={{ color: "#E4E4E7" }}>{line}</span>;
}

function CopyBtn({ text, copied, onCopy, step }: { text: string; copied: boolean; onCopy: () => void; step?: string }) {
  return (
    <button onClick={() => { if (step) { trackEvent({ pageName: `Onboarding - ${step}`, eventName: "onboarding_code_copied", additionalProperties: { step: step.toLowerCase().replace(/ /g, "_"), snippet: text.slice(0, 30) } }); } onCopy(); }} className="cursor-pointer hover:opacity-80" style={{ background: copied ? "#22C55E22" : "#ffffff10", border: "none", borderRadius: 6, padding: "4px 10px", flexShrink: 0, display: "flex", alignItems: "center", gap: 4, transition: "background 150ms" }} title="Copy">
      {copied ? (
        <><svg width="12" height="12" viewBox="0 0 16 16" fill="none"><path d="M3.5 8.5L6.5 11.5L12.5 4.5" stroke="#22C55E" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" /></svg><span style={{ color: "#22C55E", fontSize: 11, fontWeight: 500 }}>Copied</span></>
      ) : (
        <><svg width="12" height="12" viewBox="0 0 16 16" fill="none"><rect x="5" y="5" width="8" height="8" rx="1.5" stroke="#9CA3AF" strokeWidth="1.5" /><path d="M11 3H4.5A1.5 1.5 0 003 4.5V11" stroke="#9CA3AF" strokeWidth="1.5" strokeLinecap="round" /></svg><span style={{ color: "#9CA3AF", fontSize: 11 }}>Copy</span></>
      )}
    </button>
  );
}

function CodeBlock({ children, step }: { children: string; step?: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <div style={{ backgroundColor: "#0F0F10", borderRadius: 10, border: "1px solid #2A2A2E", display: "flex", alignItems: "center", justifyContent: "space-between", paddingBlock: 14, paddingInline: 20, gap: 12, overflow: "hidden" }}>
      <code style={{ color: "#E4E4E7", fontFamily: '"Fira Code", "SF Mono", "Courier New", monospace', fontSize: 13.5, lineHeight: "20px", overflowX: "auto", whiteSpace: "nowrap" }}>{colorizeLine(children)}</code>
      <CopyBtn text={children} copied={copied} step={step} onCopy={() => { navigator.clipboard.writeText(children); setCopied(true); setTimeout(() => setCopied(false), 2000); }} />
    </div>
  );
}

function MultiLineCode({ lines, step }: { lines: string[]; step?: string }) {
  const full = lines.join("\n");
  const [copied, setCopied] = useState(false);
  return (
    <div style={{ backgroundColor: "#0F0F10", borderRadius: 10, border: "1px solid #2A2A2E", display: "flex", flexDirection: "column", paddingBlock: 16, paddingInline: 20, position: "relative", gap: 0, overflow: "hidden" }}>
      <div style={{ position: "absolute", top: 12, right: 12, zIndex: 1 }}>
        <CopyBtn text={full} copied={copied} step={step} onCopy={() => { navigator.clipboard.writeText(full); setCopied(true); setTimeout(() => setCopied(false), 2000); }} />
      </div>
      <div style={{ overflowX: "auto" }}>
        {lines.map((line, i) => (
          <div key={i} style={{ display: "flex", gap: 16, minHeight: 22 }}>
            <span style={{ color: "#4A4A4F", fontFamily: '"Fira Code", "SF Mono", monospace', fontSize: 12, lineHeight: "22px", userSelect: "none", width: 20, textAlign: "right", flexShrink: 0 }}>{i + 1}</span>
            <code style={{ fontFamily: '"Fira Code", "SF Mono", "Courier New", monospace', fontSize: 13.5, lineHeight: "22px", whiteSpace: "pre" }}>{colorizeLine(line)}</code>
          </div>
        ))}
      </div>
    </div>
  );
}

function TokenDisplay({ label, token, step }: { label: string; token: string; step?: string }) {
  const masked = token.length > 8 ? token.slice(0, 6) + "****" + token.slice(-4) : token || "sk-cog-****3f8a";
  const [copied, setCopied] = useState(false);
  return (
    <div style={{ alignItems: "center", backgroundColor: "#F4F4F5", borderRadius: 8, display: "flex", gap: 12, paddingBlock: 12, paddingInline: 16 }}>
      <span style={{ color: "#71717A", fontFamily: '"Inter", system-ui, sans-serif', fontSize: 13, flexShrink: 0 }}>{label}:</span>
      <span style={{ color: "#18181B", fontFamily: '"Fira Code", "Courier New", monospace', fontSize: 13 }}>{masked}</span>
      <button onClick={() => { if (step) { trackEvent({ pageName: `Onboarding - ${step}`, eventName: "onboarding_code_copied", additionalProperties: { step: step.toLowerCase().replace(/ /g, "_"), snippet: (token || masked).slice(0, 30) } }); } navigator.clipboard.writeText(token || masked); setCopied(true); setTimeout(() => setCopied(false), 2000); }} className="cursor-pointer" style={{ background: "none", border: "none", padding: 0, marginLeft: "auto", flexShrink: 0 }}>
        {copied ? <span style={{ color: "#22C55E", fontSize: 11 }}>Copied</span> : (
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none"><rect x="5" y="5" width="8" height="8" rx="1.5" stroke="#71717A" strokeWidth="1.5" /><path d="M11 3H4.5A1.5 1.5 0 003 4.5V11" stroke="#71717A" strokeWidth="1.5" strokeLinecap="round" /></svg>
        )}
      </button>
    </div>
  );
}

function WaitingIndicator({ text }: { text: string }) {
  return (
    <div style={{ alignItems: "center", backgroundColor: "#F4F4F5", borderRadius: 8, display: "flex", gap: 12, paddingBlock: 12, paddingInline: 16 }}>
      <div style={{ border: "2px solid", borderColor: "#6510F4 #E4E4E7 #E4E4E7 #E4E4E7", borderRadius: "50%", width: 16, height: 16, flexShrink: 0, animation: "spin 0.8s linear infinite" }} />
      <span style={{ color: "#71717A", fontFamily: '"Inter", system-ui, sans-serif', fontSize: 13 }}>{text}</span>
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}

function NavButtons({ onBack, onContinue, continueDisabled, pageName }: { onBack: () => void; onContinue?: () => void; continueDisabled?: boolean; pageName?: string }) {
  return (
    <div style={{ display: "flex", gap: 12 }}>
      <button onClick={() => { if (pageName) { trackEvent({ pageName, eventName: "onboarding_nav_back_clicked" }); } onBack(); }} className="cursor-pointer" style={{ alignItems: "center", borderColor: "#E4E4E7", borderRadius: 8, borderStyle: "solid", borderWidth: 1, background: "white", display: "flex", height: 40, justifyContent: "center", paddingInline: 20 }}>
        <span style={{ color: "#3F3F46", fontFamily: '"Inter", system-ui, sans-serif', fontSize: 13, fontWeight: 500 }}>Back</span>
      </button>
      {onContinue && (
        <button onClick={() => { if (pageName) { trackEvent({ pageName, eventName: "onboarding_nav_continue_clicked" }); } onContinue(); }} disabled={continueDisabled} className="cursor-pointer" style={{ alignItems: "center", backgroundColor: continueDisabled ? "#E4E4E7" : "#6510F4", borderRadius: 8, border: "none", display: "flex", height: 40, justifyContent: "center", paddingInline: 20, opacity: continueDisabled ? 0.6 : 1 }}>
          <span style={{ color: continueDisabled ? "#A1A1AA" : "#FFFFFF", fontFamily: '"Inter", system-ui, sans-serif', fontSize: 13, fontWeight: 500 }}>Continue</span>
        </button>
      )}
    </div>
  );
}

function SkipLink({ onClick, pageName }: { onClick: () => void; pageName?: string }) {
  return (
    <button onClick={() => { if (pageName) { trackEvent({ pageName, eventName: "onboarding_skip_clicked" }); } onClick(); }} className="cursor-pointer" style={{ background: "none", border: "none", color: "#9CA3AF", fontFamily: '"Inter", system-ui, sans-serif', fontSize: 13, paddingTop: 16 }}>
      Skip onboarding and go to dashboard
    </button>
  );
}

// ── Connect Local Cognee SDK ──

export function LocalCogneeStep({ onBack, onSkip, standalone }: { onBack: () => void; onSkip: () => void; standalone?: boolean }) {
  const [connectMode, setConnectMode] = useState<"cloud" | "direct">("cloud");

  return (
    <div style={{ alignItems: "center", display: "flex", flexDirection: "column", flexGrow: 1, gap: standalone ? 16 : 32, paddingBlock: standalone ? 16 : 48, paddingInline: standalone ? 16 : 40, fontFamily: '"Inter", system-ui, sans-serif', overflowY: "auto" }}>
      {!standalone && (
        <div style={{ alignItems: "center", display: "flex", flexDirection: "column", gap: 8 }}>
          <StepBadge step={1} />
          <h1 style={{ color: "#18181B", fontSize: 28, fontWeight: 600, lineHeight: "34px", margin: 0 }}>Connect your local Cognee SDK</h1>
          <p style={{ color: "#71717A", fontSize: 15, lineHeight: "22px", margin: 0, textAlign: "center", maxWidth: 540 }}>
            Use <code style={{ background: "#F4F4F5", padding: "1px 6px", borderRadius: 4, fontSize: 13 }}>cognee.serve()</code> to link your local Python SDK to a remote Cognee instance. All operations (remember, recall, forget, improve) will route to the connected instance.
          </p>
        </div>
      )}

      <div style={{ backgroundColor: "#FFFFFF", borderColor: standalone ? "transparent" : "#E4E4E7", borderRadius: standalone ? 0 : 12, borderStyle: "solid", borderWidth: standalone ? 0 : 1, display: "flex", flexDirection: "column", gap: 20, paddingBlock: 24, paddingInline: 24, width: "100%" }}>

        {/* Connection mode toggle */}
        <div style={{ display: "flex", gap: 0, borderRadius: 8, border: "1px solid #E4E4E7", overflow: "hidden" }}>
          <button onClick={() => { trackEvent({ pageName: "Onboarding - Local Cognee", eventName: "local_cognee_mode_switched", additionalProperties: { mode: "cloud" } }); setConnectMode("cloud"); }} className="cursor-pointer" style={{ flex: 1, padding: "10px 16px", background: connectMode === "cloud" ? "#F0EDFF" : "#fff", border: "none", fontSize: 13, fontWeight: 500, color: connectMode === "cloud" ? "#6510F4" : "#71717A", fontFamily: "inherit", borderRight: "1px solid #E4E4E7" }}>
            Connect to Cognee Cloud
          </button>
          <button onClick={() => { trackEvent({ pageName: "Onboarding - Local Cognee", eventName: "local_cognee_mode_switched", additionalProperties: { mode: "direct" } }); setConnectMode("direct"); }} className="cursor-pointer" style={{ flex: 1, padding: "10px 16px", background: connectMode === "direct" ? "#F0EDFF" : "#fff", border: "none", fontSize: 13, fontWeight: 500, color: connectMode === "direct" ? "#6510F4" : "#71717A", fontFamily: "inherit" }}>
            Connect to any instance
          </button>
        </div>

        {connectMode === "cloud" ? (
          <>
            {/* CLOUD: device code login */}
            <div style={{ display: "flex", gap: 10, background: "#F0EDFF", border: "1px solid #DDD6FE", borderRadius: 8, padding: "10px 14px" }}>
              <span style={{ fontSize: 12, color: "#52525B", lineHeight: "18px" }}>
                Connect your local Cognee SDK to your cloud tenant using the API Base URL and key from the <a href="/api-keys" style={{ color: "#6510F4", textDecoration: "underline" }}>API Keys</a> page. All operations (remember, recall) will route to cloud.
              </span>
            </div>

            <div style={{ alignItems: "center", display: "flex", gap: 12 }}>
              <NumberCircle n={1} />
              <span style={{ color: "#18181B", fontSize: 14, fontWeight: 500 }}>Install Cognee</span>
            </div>
            <CodeBlock step="Local Cognee">pip install cognee</CodeBlock>

            <Divider />

            <div style={{ alignItems: "center", display: "flex", gap: 12 }}>
              <NumberCircle n={2} />
              <span style={{ color: "#18181B", fontSize: 14, fontWeight: 500 }}>Get your API Base URL and API key</span>
            </div>
            <span style={{ color: "#71717A", fontSize: 12, lineHeight: "18px" }}>
              Copy your API Base URL and API key from the API Keys page.
            </span>
            <Link
              href="/api-keys"
              onClick={() => trackEvent({ pageName: "Onboarding - Local Cognee", eventName: "local_go_to_api_keys" })}
              style={{ display: "inline-flex", alignItems: "center", gap: 6, background: "#F0EDFF", color: "#6510F4", border: "1px solid #DDD6FE", borderRadius: 8, padding: "8px 16px", fontSize: 13, fontWeight: 500, textDecoration: "none", width: "fit-content" }}
            >
              Go to API Keys
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#6510F4" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M5 12h14" /><path d="M12 5l7 7-7 7" /></svg>
            </Link>

            <Divider />

            <div style={{ alignItems: "center", display: "flex", gap: 12 }}>
              <NumberCircle n={3} />
              <span style={{ color: "#18181B", fontSize: 14, fontWeight: 500 }}>Connect and use Cognee</span>
            </div>
            <MultiLineCode step="Local Cognee" lines={[
              "import asyncio",
              "import cognee",
              "",
              "async def main():",
              '    await cognee.serve(',
              '        url="<your-api-base-url>",',
              '        api_key="<your-api-key>"',
              "    )",
              "",
              '    await cognee.remember("Einstein developed general relativity in 1915.", datasets=["default_dataset"])',
              "",
              '    results = await cognee.recall("What did Einstein develop?", datasets=["default_dataset"])',
              "    print(results)",
              "",
              "asyncio.run(main())",
            ]} />

            <Divider />

            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              <span style={{ color: "#71717A", fontSize: 12, fontWeight: 500, letterSpacing: "0.04em", textTransform: "uppercase" }}>Re-connecting</span>
              <span style={{ color: "#A1A1AA", fontSize: 12, lineHeight: "18px" }}>
                Credentials are saved at <code style={{ fontSize: 11 }}>~/.cognee/cloud_credentials.json</code>. Subsequent <code style={{ fontSize: 11 }}>cognee.serve()</code> calls reconnect automatically.
              </span>
            </div>
          </>
        ) : (
          <>
            {/* DIRECT: URL + API key */}
            <div style={{ display: "flex", gap: 10, background: "#F0EDFF", border: "1px solid #DDD6FE", borderRadius: 8, padding: "10px 14px" }}>
              <span style={{ fontSize: 12, color: "#52525B", lineHeight: "18px" }}>
                Connect to any Cognee instance — self-hosted, another team member's machine, or a staging environment. Just provide the URL and an optional API key.
              </span>
            </div>

            <div style={{ alignItems: "center", display: "flex", gap: 12 }}>
              <NumberCircle n={1} />
              <span style={{ color: "#18181B", fontSize: 14, fontWeight: 500 }}>Install Cognee</span>
            </div>
            <CodeBlock step="Local Cognee">pip install cognee</CodeBlock>

            <Divider />

            <div style={{ alignItems: "center", display: "flex", gap: 12 }}>
              <NumberCircle n={2} />
              <span style={{ color: "#18181B", fontSize: 14, fontWeight: 500 }}>Connect with URL and API key</span>
            </div>
            <MultiLineCode step="Local Cognee" lines={[
              "import asyncio",
              "import cognee",
              "",
              "async def main():",
              "    # Connect to a specific instance",
              '    await cognee.serve(',
              '        url="https://your-instance.cognee.ai",',
              '        api_key="your-api-key"',
              "    )",
              "",
              "    # Add and process data on the remote instance",
              '    await cognee.remember("Your data here", datasets=["default_dataset"])',
              "",
              "    # Query the remote knowledge graph",
              '    results = await cognee.recall("What do we know about X?", datasets=["default_dataset"])',
              "    print(results)",
              "",
              "asyncio.run(main())",
            ]} />

            <Divider />

            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              <span style={{ color: "#71717A", fontSize: 12, fontWeight: 500, letterSpacing: "0.04em", textTransform: "uppercase" }}>Local to local</span>
              <span style={{ color: "#A1A1AA", fontSize: 12, lineHeight: "18px" }}>
                To connect to a Cognee backend running on the same machine or your local network:
              </span>
              <MultiLineCode step="Local Cognee" lines={[
                "# Inside an async function:",
                '# Same machine',
                'await cognee.serve(url="http://localhost:8000")',
                "",
                "# Local network",
                'await cognee.serve(url="http://192.168.1.50:8000")',
              ]} />
            </div>
          </>
        )}
      </div>

      {!standalone && <NavButtons onBack={onBack} continueDisabled={false} onContinue={onSkip} pageName="Onboarding - Local Cognee" />}
    </div>
  );
}

// ── MCP Server Content (Local / Cloud toggle) ──

function McpServerContent() {
  const [mcpMode, setMcpMode] = useState<"local" | "cloud">("local");

  return (
    <>
      <div style={{ display: "flex", gap: 10, background: "#F0EDFF", border: "1px solid #DDD6FE", borderRadius: 8, padding: "10px 14px" }}>
        <span style={{ fontSize: 12, color: "#52525B", lineHeight: "18px" }}>
          Cognee runs as an <a href="https://modelcontextprotocol.io" target="_blank" rel="noopener noreferrer" style={{ color: "#6510F4", textDecoration: "underline" }} onClick={() => trackEvent({ pageName: "Onboarding - Agent", eventName: "click_out", additionalProperties: { target_url: "https://modelcontextprotocol.io" } })}>MCP (Model Context Protocol)</a> server. Any MCP-compatible client (Claude Desktop, Cursor, VS Code Copilot, etc.) can connect to it as a tool provider.
        </span>
      </div>

      {/* Local / Cloud toggle */}
      <div style={{ display: "flex", gap: 0, borderRadius: 8, border: "1px solid #E4E4E7", overflow: "hidden" }}>
        <button onClick={() => { trackEvent({ pageName: "Onboarding - Agent", eventName: "mcp_mode_switched", additionalProperties: { mode: "local" } }); setMcpMode("local"); }} className="cursor-pointer" style={{ flex: 1, padding: "8px 16px", background: mcpMode === "local" ? "#F0EDFF" : "#fff", border: "none", fontSize: 13, fontWeight: 500, color: mcpMode === "local" ? "#6510F4" : "#71717A", fontFamily: "inherit", borderRight: "1px solid #E4E4E7" }}>
          Local
        </button>
        <button onClick={() => { trackEvent({ pageName: "Onboarding - Agent", eventName: "mcp_mode_switched", additionalProperties: { mode: "cloud" } }); setMcpMode("cloud"); }} className="cursor-pointer" style={{ flex: 1, padding: "8px 16px", background: mcpMode === "cloud" ? "#F0EDFF" : "#fff", border: "none", fontSize: 13, fontWeight: 500, color: mcpMode === "cloud" ? "#6510F4" : "#71717A", fontFamily: "inherit" }}>
          Cloud
        </button>
      </div>

      <div style={{ alignItems: "center", display: "flex", gap: 12 }}>
        <NumberCircle n={1} />
        <span style={{ color: "#18181B", fontSize: 14, fontWeight: 500 }}>Start the MCP server</span>
      </div>

      {mcpMode === "local" ? (
        <>
          <span style={{ color: "#71717A", fontSize: 12, lineHeight: "18px" }}>
            The MCP server starts automatically with <code style={{ fontSize: 11 }}>cognee -ui</code>, or run it standalone:
          </span>
          <CodeBlock step="Agent">cognee-mcp --transport sse --port 8001</CodeBlock>
          <div style={{ display: "flex", gap: 8, background: "#FEF3C7", border: "1px solid #FDE68A", borderRadius: 8, padding: "10px 14px", alignItems: "flex-start" }}>
            <span style={{ fontSize: 16, flexShrink: 0 }}>&#9888;</span>
            <span style={{ fontSize: 12, color: "#92400E", lineHeight: "18px" }}>
              Local mode requires an LLM API key. Set it as an environment variable before starting:<br />
              <code style={{ fontSize: 11, background: "#FDE68A", padding: "1px 4px", borderRadius: 3 }}>LLM_API_KEY=sk-... cognee-mcp --transport sse --port 8001</code>
            </span>
          </div>
        </>
      ) : (
        <>
          <span style={{ color: "#71717A", fontSize: 12, lineHeight: "18px" }}>
            Connect the MCP server to Cognee Cloud. Use your API Base URL and API key from the <Link href="/api-keys" style={{ color: "#6510F4", textDecoration: "underline" }} onClick={() => trackEvent({ pageName: "Onboarding - Agent", eventName: "mcp_go_to_api_keys" })}>API Keys</Link> page.
          </span>
          <MultiLineCode step="Agent" lines={[
            'cognee-mcp --transport sse --port 8001 \\',
            '  --serve-url https://your-instance.cognee.ai \\',
            '  --serve-api-key ck_your_api_key',
          ]} />
        </>
      )}

      <Divider />

      <div style={{ alignItems: "center", display: "flex", gap: 12 }}>
        <NumberCircle n={2} />
        <span style={{ color: "#18181B", fontSize: 14, fontWeight: 500 }}>Add to your MCP client config</span>
      </div>
      <span style={{ color: "#71717A", fontSize: 12, lineHeight: "18px" }}>
        Add Cognee as a tool server in your MCP client&apos;s configuration:
      </span>
      <MultiLineCode step="Agent" lines={[
        '{',
        '  "mcpServers": {',
        '    "cognee": {',
        '      "url": "http://localhost:8001/sse"',
        '    }',
        '  }',
        '}',
      ]} />

      <Divider />

      <div style={{ alignItems: "center", display: "flex", gap: 12 }}>
        <NumberCircle n={3} />
        <span style={{ color: "#18181B", fontSize: 14, fontWeight: 500 }}>Available tools</span>
      </div>
      <span style={{ color: "#71717A", fontSize: 12, lineHeight: "18px" }}>
        Once connected, your MCP client gets these tools:
      </span>
      <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        {[
          { tool: "remember", desc: "Store data in memory (add + cognify in one step)" },
          { tool: "recall", desc: "Search memory with auto-routing" },
          { tool: "cognify", desc: "Build the knowledge graph from ingested data" },
          { tool: "search", desc: "Query the knowledge graph with different search types" },
          { tool: "improve", desc: "Enrich the knowledge graph and bridge session data" },
          { tool: "forget_memory", desc: "Delete data from memory" },
          { tool: "list_data", desc: "List datasets and their contents" },
        ].map((t) => (
          <div key={t.tool} style={{ display: "flex", alignItems: "center", gap: 10, padding: "6px 0" }}>
            <code style={{ background: "#F4F4F5", padding: "2px 8px", borderRadius: 4, fontSize: 12, color: "#6510F4", fontWeight: 500 }}>{t.tool}</code>
            <span style={{ fontSize: 12, color: "#71717A" }}>{t.desc}</span>
          </div>
        ))}
      </div>
    </>
  );
}

// ── Agent Connection ──

export function AgentStep({ onBack, onSkip, standalone }: { onBack: () => void; onSkip: () => void; standalone?: boolean }) {
  const [method, setMethod] = useState<"sdk" | "mcp" | "api">("sdk");

  return (
    <div style={{ alignItems: "center", display: "flex", flexDirection: "column", flexGrow: 1, gap: standalone ? 16 : 32, paddingBlock: standalone ? 16 : 48, paddingInline: standalone ? 16 : 40, fontFamily: '"Inter", system-ui, sans-serif', overflowY: "auto" }}>
      {!standalone && (
        <div style={{ alignItems: "center", display: "flex", flexDirection: "column", gap: 8 }}>
          <StepBadge step={1} />
          <h1 style={{ color: "#18181B", fontSize: 28, fontWeight: 600, lineHeight: "34px", margin: 0 }}>Connect your Agent</h1>
          <p style={{ color: "#71717A", fontSize: 15, lineHeight: "22px", margin: 0, textAlign: "center", maxWidth: 520 }}>
            Give any agent memory by connecting it to Cognee. Choose how your agent should integrate.
          </p>
        </div>
      )}

      <div style={{ backgroundColor: "#FFFFFF", borderColor: standalone ? "transparent" : "#E4E4E7", borderRadius: standalone ? 0 : 12, borderStyle: "solid", borderWidth: standalone ? 0 : 1, display: "flex", flexDirection: "column", gap: 20, paddingBlock: 24, paddingInline: 24, width: "100%" }}>

        {/* Method tabs */}
        <div style={{ display: "flex", gap: 0, borderRadius: 8, border: "1px solid #E4E4E7", overflow: "hidden" }}>
          <button onClick={() => { trackEvent({ pageName: "Onboarding - Agent", eventName: "agent_method_switched", additionalProperties: { method: "sdk" } }); setMethod("sdk"); }} className="cursor-pointer" style={{ flex: 1, padding: "10px 16px", background: method === "sdk" ? "#F0EDFF" : "#fff", border: "none", fontSize: 13, fontWeight: 500, color: method === "sdk" ? "#6510F4" : "#71717A", fontFamily: "inherit", borderRight: "1px solid #E4E4E7" }}>
            Python SDK
          </button>
          <button onClick={() => { trackEvent({ pageName: "Onboarding - Agent", eventName: "agent_method_switched", additionalProperties: { method: "mcp" } }); setMethod("mcp"); }} className="cursor-pointer" style={{ flex: 1, padding: "10px 16px", background: method === "mcp" ? "#F0EDFF" : "#fff", border: "none", fontSize: 13, fontWeight: 500, color: method === "mcp" ? "#6510F4" : "#71717A", fontFamily: "inherit", borderRight: "1px solid #E4E4E7" }}>
            MCP Server
          </button>
          <button onClick={() => { trackEvent({ pageName: "Onboarding - Agent", eventName: "agent_method_switched", additionalProperties: { method: "api" } }); setMethod("api"); }} className="cursor-pointer" style={{ flex: 1, padding: "10px 16px", background: method === "api" ? "#F0EDFF" : "#fff", border: "none", fontSize: 13, fontWeight: 500, color: method === "api" ? "#6510F4" : "#71717A", fontFamily: "inherit" }}>
            REST API
          </button>
        </div>

        {/* ── Python SDK ── */}
        {method === "sdk" && (
          <>
            <div style={{ display: "flex", gap: 10, background: "#F0EDFF", border: "1px solid #DDD6FE", borderRadius: 8, padding: "10px 14px" }}>
              <span style={{ fontSize: 12, color: "#52525B", lineHeight: "18px" }}>
                The Cognee Python SDK works with any agent framework. Use <code style={{ background: "#E4DEFF", padding: "1px 4px", borderRadius: 3, fontSize: 11 }}>cognee.serve(url, api_key)</code> to connect, then <code style={{ background: "#E4DEFF", padding: "1px 4px", borderRadius: 3, fontSize: 11 }}>remember</code> / <code style={{ background: "#E4DEFF", padding: "1px 4px", borderRadius: 3, fontSize: 11 }}>recall</code> from your agent code.
              </span>
            </div>

            <div style={{ alignItems: "center", display: "flex", gap: 12 }}>
              <NumberCircle n={1} />
              <span style={{ color: "#18181B", fontSize: 14, fontWeight: 500 }}>Install Cognee</span>
            </div>
            <CodeBlock step="Agent">pip install cognee</CodeBlock>

            <Divider />

            <div style={{ alignItems: "center", display: "flex", gap: 12 }}>
              <NumberCircle n={2} />
              <span style={{ color: "#18181B", fontSize: 14, fontWeight: 500 }}>Connect your agent</span>
            </div>
            <span style={{ color: "#71717A", fontSize: 12, lineHeight: "18px" }}>
              Use your API Base URL and API key from the <Link href="/api-keys" style={{ color: "#6510F4", textDecoration: "underline" }} onClick={() => trackEvent({ pageName: "Onboarding - Agent", eventName: "agent_go_to_api_keys" })}>API Keys</Link> page.
            </span>
            <MultiLineCode step="Agent" lines={[
              "import asyncio",
              "import cognee",
              "",
              "async def main():",
              "    await cognee.serve(",
              '        url="<your-api-base-url>",',
              '        api_key="<your-api-key>"',
              "    )",
              "",
              "    # Store knowledge from your agent",
              '    await cognee.remember("User prefers dark mode and concise answers.", datasets=["default_dataset"])',
              "",
              "    # Retrieve relevant context for your agent",
              '    results = await cognee.recall("What are the user preferences?", datasets=["default_dataset"])',
              "    for item in results:",
              '        print(item["search_result"])',
              "",
              "asyncio.run(main())",
            ]} />

            <Divider />

            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              <span style={{ color: "#71717A", fontSize: 12, fontWeight: 500, letterSpacing: "0.04em", textTransform: "uppercase" }}>Works with any framework</span>
              <span style={{ color: "#A1A1AA", fontSize: 12, lineHeight: "18px" }}>
                The SDK is framework-agnostic. Call <code style={{ fontSize: 11 }}>cognee.remember()</code> and <code style={{ fontSize: 11 }}>cognee.recall()</code> from within your agent's tool functions, hooks, or middleware — regardless of whether you use LangChain, LlamaIndex, raw OpenAI, or a custom framework.
              </span>
            </div>
          </>
        )}

        {/* ── MCP Server ── */}
        {method === "mcp" && (
          <McpServerContent />
        )}

        {/* ── REST API ── */}
        {method === "api" && (
          <>
            <div style={{ display: "flex", gap: 10, background: "#F0EDFF", border: "1px solid #DDD6FE", borderRadius: 8, padding: "10px 14px" }}>
              <span style={{ fontSize: 12, color: "#52525B", lineHeight: "18px" }}>
                Use the REST API directly from any language or framework. Create an API key, then use it to call the endpoints.
              </span>
            </div>

            <div style={{ alignItems: "center", display: "flex", gap: 12 }}>
              <NumberCircle n={1} />
              <span style={{ color: "#18181B", fontSize: 14, fontWeight: 500 }}>Create an API key</span>
            </div>
            <span style={{ color: "#71717A", fontSize: 12, lineHeight: "18px" }}>
              Use the <Link href="/api-keys" style={{ color: "#6510F4", textDecoration: "underline" }}>API Keys</Link> page to create a key, or call the local backend endpoint directly.
            </span>
            <div style={{ display: "flex", gap: 8, background: "#FEF3C7", border: "1px solid #FDE68A", borderRadius: 8, padding: "10px 14px", alignItems: "flex-start" }}>
              <span style={{ fontSize: 16, flexShrink: 0 }}>&#9888;</span>
              <span style={{ fontSize: 12, color: "#92400E", lineHeight: "18px" }}>
                Newly created API keys are only shown once in the response. Copy and store them securely.
              </span>
            </div>
            <MultiLineCode step="Agent" lines={[
              "# Create an API key",
              'curl -X POST "<your-api-base-url>/api/v1/auth/api-keys" \\',
              '  -H "Content-Type: application/json" \\',
              '  -d \'{"name": "MyAgent"}\'',
              "",
              "# List your API keys",
              'curl "<your-api-base-url>/api/v1/auth/api-keys" \\',
              '  -H "X-Api-Key: <your-api-key>"',
            ]} />

            <Divider />

            <div style={{ alignItems: "center", display: "flex", gap: 12 }}>
              <NumberCircle n={2} />
              <span style={{ color: "#18181B", fontSize: 14, fontWeight: 500 }}>Store and retrieve knowledge</span>
            </div>
            <span style={{ color: "#71717A", fontSize: 12, lineHeight: "18px" }}>
              Use the API key to store and query knowledge.
            </span>
            <MultiLineCode step="Agent" lines={[
              "# Remember — store knowledge",
              'curl -X POST "<your-api-base-url>/api/v1/remember" \\',
              '  -H "X-Api-Key: <api-key>" \\',
              '  -F "data=@document.pdf" \\',
              '  -F "datasetName=my_agent_data"',
              "",
              "# Recall — query the knowledge graph",
              'curl -X POST "<your-api-base-url>/api/v1/recall" \\',
              '  -H "X-Api-Key: <api-key>" \\',
              '  -H "Content-Type: application/json" \\',
              '  -d \'{"query": "What do we know?"}\'',
            ]} />
          </>
        )}
      </div>

      {!standalone && <NavButtons onBack={onBack} continueDisabled={false} onContinue={onSkip} pageName="Onboarding - Agent" />}
    </div>
  );
}

// ── Data Source Ingestion ──

type SourceCategory = "database" | "saas" | "files";

const DB_CONFIGS: Record<string, { example: string }> = {
  PostgreSQL: { example: "postgresql://user:pass@localhost:5432/mydb" },
  MySQL: { example: "mysql+pymysql://user:pass@localhost:3306/mydb" },
  SQLite: { example: "sqlite:///path/to/database.db" },
  "SQL Server": { example: "mssql://user:pass@localhost:1433/mydb" },
};

const SAAS_SOURCES: { name: string; install: string; code: string[]; extracts: string }[] = [
  {
    name: "Slack",
    install: 'pip install "dlt[slack]"',
    code: [
      "from dlt.sources.slack import slack_source",
      "",
      "source = slack_source(",
      '    selected_channels=["general", "engineering"],',
      "    start_date=datetime(2025, 1, 1),",
      ")",
      "",
      'await cognee.add(source, dataset_name="slack_data")',
    ],
    extracts: "channels, messages, users, threads",
  },
  {
    name: "Notion",
    install: 'pip install "dlt[notion]"',
    code: [
      "from dlt.sources.notion import notion_databases",
      "",
      "source = notion_databases()",
      "",
      'await cognee.add(source, dataset_name="notion_data")',
    ],
    extracts: "databases, pages, properties",
  },
  {
    name: "GitHub",
    install: 'pip install "dlt[github]"',
    code: [
      "from dlt.sources.github import github_reactions",
      "",
      'source = github_reactions("your-org", "your-repo")',
      "",
      'await cognee.add(source, dataset_name="github_data")',
    ],
    extracts: "issues, pull requests, comments, reactions",
  },
  {
    name: "HubSpot",
    install: 'pip install "dlt[hubspot]"',
    code: [
      "from dlt.sources.hubspot import hubspot",
      "",
      "source = hubspot()",
      "",
      'await cognee.add(source, dataset_name="hubspot_data")',
    ],
    extracts: "contacts, companies, deals, tickets",
  },
  {
    name: "Google Sheets",
    install: 'pip install "dlt[google_sheets]"',
    code: [
      "from dlt.sources.google_sheets import google_spreadsheet",
      "",
      "source = google_spreadsheet(",
      '    spreadsheet_url="https://docs.google.com/spreadsheets/d/...",',
      ")",
      "",
      'await cognee.add(source, dataset_name="sheets_data")',
    ],
    extracts: "spreadsheet rows with auto-typed columns",
  },
  {
    name: "Jira",
    install: 'pip install "dlt[jira]"',
    code: [
      "from dlt.sources.jira import jira",
      "",
      "source = jira()",
      "",
      'await cognee.add(source, dataset_name="jira_data")',
    ],
    extracts: "issues, projects, users, workflows",
  },
  {
    name: "REST API",
    install: "# No extra install needed",
    code: [
      "from dlt.sources.rest_api import rest_api_source",
      "",
      "source = rest_api_source({",
      '    "client": {"base_url": "https://api.example.com"},',
      '    "resources": ["users", "orders"],',
      "})",
      "",
      'await cognee.add(source, dataset_name="api_data")',
    ],
    extracts: "any REST API endpoints you define",
  },
];

function SourceCategoryTab({ label, active, onClick }: { label: string; active: boolean; onClick: () => void }) {
  return (
    <button onClick={onClick} className="cursor-pointer" style={{
      flex: 1, padding: "10px 16px",
      background: active ? "#F0EDFF" : "#fff",
      border: "none", fontSize: 13, fontWeight: 500,
      color: active ? "#6510F4" : "#71717A",
      fontFamily: "inherit",
    }}>
      {label}
    </button>
  );
}

export function DatabaseStep({ onBack, onSkip, standalone }: { onBack: () => void; onSkip: () => void; standalone?: boolean }) {
  const [category, setCategory] = useState<SourceCategory>("database");
  const [dbType, setDbType] = useState("PostgreSQL");
  const [saasSource, setSaasSource] = useState(SAAS_SOURCES[0].name);
  const [mode, setMode] = useState<"local" | "cloud">("local");

  const dbConfig = DB_CONFIGS[dbType];
  const saasConfig = SAAS_SOURCES.find((s) => s.name === saasSource) || SAAS_SOURCES[0];

  return (
    <div style={{ alignItems: "center", display: "flex", flexDirection: "column", flexGrow: 1, gap: standalone ? 16 : 32, paddingBlock: standalone ? 16 : 48, paddingInline: standalone ? 16 : 40, fontFamily: '"Inter", system-ui, sans-serif', overflowY: "auto" }}>
      {!standalone && (
        <div style={{ alignItems: "center", display: "flex", flexDirection: "column", gap: 8 }}>
          <StepBadge step={1} />
          <h1 style={{ color: "#18181B", fontSize: 28, fontWeight: 600, lineHeight: "34px", margin: 0 }}>Ingest from Any Source</h1>
          <p style={{ color: "#71717A", fontSize: 15, lineHeight: "22px", margin: 0, textAlign: "center", maxWidth: 560 }}>
            Cognee uses <a href="https://dlthub.com" target="_blank" rel="noopener noreferrer" style={{ color: "#6510F4", textDecoration: "underline" }} onClick={() => trackEvent({ pageName: "Onboarding - Database", eventName: "click_out", additionalProperties: { target_url: "https://dlthub.com" } })}>dlt (data load tool)</a> to connect to databases, SaaS platforms, and APIs. Data is extracted, loaded into a knowledge graph with schema and relationships preserved, and made searchable.
          </p>
        </div>
      )}

      <div style={{ backgroundColor: "#FFFFFF", borderColor: standalone ? "transparent" : "#E4E4E7", borderRadius: standalone ? 0 : 12, borderStyle: "solid", borderWidth: standalone ? 0 : 1, display: "flex", flexDirection: "column", gap: 20, paddingBlock: 24, paddingInline: 24, width: "100%" }}>

        {/* Source category tabs */}
        <div style={{ display: "flex", gap: 0, borderRadius: 8, border: "1px solid #E4E4E7", overflow: "hidden" }}>
          <SourceCategoryTab label="Databases" active={category === "database"} onClick={() => { trackEvent({ pageName: "Onboarding - Database", eventName: "database_category_switched", additionalProperties: { category: "database" } }); setCategory("database"); }} />
          <SourceCategoryTab label="SaaS & APIs" active={category === "saas"} onClick={() => { trackEvent({ pageName: "Onboarding - Database", eventName: "database_category_switched", additionalProperties: { category: "saas" } }); setCategory("saas"); }} />
          <SourceCategoryTab label="Files & CSV" active={category === "files"} onClick={() => { trackEvent({ pageName: "Onboarding - Database", eventName: "database_category_switched", additionalProperties: { category: "files" } }); setCategory("files"); }} />
        </div>

        {/* Mode toggle */}
        <div style={{ display: "flex", gap: 0, borderRadius: 8, border: "1px solid #E4E4E7", overflow: "hidden" }}>
          <button onClick={() => { trackEvent({ pageName: "Onboarding - Database", eventName: "database_mode_switched", additionalProperties: { mode: "local" } }); setMode("local"); }} className="cursor-pointer" style={{ flex: 1, padding: "8px 14px", background: mode === "local" ? "#F4F4F5" : "#fff", border: "none", fontSize: 12, fontWeight: 500, color: mode === "local" ? "#18181B" : "#A1A1AA", fontFamily: "inherit", borderRight: "1px solid #E4E4E7" }}>
            Local / Self-hosted
          </button>
          <button onClick={() => { trackEvent({ pageName: "Onboarding - Database", eventName: "database_mode_switched", additionalProperties: { mode: "cloud" } }); setMode("cloud"); }} className="cursor-pointer" style={{ flex: 1, padding: "8px 14px", background: mode === "cloud" ? "#F4F4F5" : "#fff", border: "none", fontSize: 12, fontWeight: 500, color: mode === "cloud" ? "#18181B" : "#A1A1AA", fontFamily: "inherit" }}>
            Cognee Cloud
          </button>
        </div>

        {mode === "cloud" && (
          <div style={{ display: "flex", gap: 10, background: "#FEF3C7", border: "1px solid #FDE68A", borderRadius: 8, padding: "10px 14px", alignItems: "flex-start" }}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#D97706" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0, marginTop: 2 }}>
              <circle cx="12" cy="12" r="10" /><line x1="12" y1="8" x2="12" y2="12" /><line x1="12" y1="16" x2="12.01" y2="16" />
            </svg>
            <span style={{ fontSize: 12, color: "#92400E", lineHeight: "18px" }}>
              For cloud, the Cognee SDK runs locally to access your data sources, then pushes extracted data to your cloud tenant. Call <code style={{ background: "#FEF3C7", fontSize: 11 }}>cognee.serve(url, api_key)</code> first to connect.
            </span>
          </div>
        )}

        <Divider />

        {/* ── DATABASE TAB ── */}
        {category === "database" && (
          <>
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              <span style={{ color: "#71717A", fontSize: 12, fontWeight: 500, letterSpacing: "0.04em", textTransform: "uppercase" }}>Database type</span>
              <select value={dbType} onChange={(e) => { trackEvent({ pageName: "Onboarding - Database", eventName: "database_type_selected", additionalProperties: { db_type: e.target.value } }); setDbType(e.target.value); }} style={{ backgroundColor: "#fff", borderColor: "#E4E4E7", borderRadius: 8, borderStyle: "solid", borderWidth: 1, height: 38, paddingInline: 14, fontSize: 13, fontWeight: 500, color: "#18181B", outline: "none" }}>
                {Object.keys(DB_CONFIGS).map((d) => <option key={d} value={d}>{d}</option>)}
              </select>
            </div>

            <div style={{ display: "flex", gap: 10, background: "#F0EDFF", border: "1px solid #DDD6FE", borderRadius: 8, padding: "10px 14px" }}>
              <span style={{ fontSize: 12, color: "#52525B", lineHeight: "18px" }}>
                Pass your connection string to <code style={{ background: "#E4DEFF", padding: "1px 4px", borderRadius: 3, fontSize: 11 }}>cognee.add()</code>. Cognee auto-discovers tables, extracts schema and foreign key relationships, and builds a searchable knowledge graph.
              </span>
            </div>

            <div style={{ alignItems: "center", display: "flex", gap: 12 }}>
              <NumberCircle n={1} />
              <span style={{ color: "#18181B", fontSize: 14, fontWeight: 500 }}>Install and ingest</span>
            </div>
            <MultiLineCode step="Database" lines={[
              '# Install DLT support',
              'pip install "cognee[dlt]"',
            ]} />
            <MultiLineCode step="Database" lines={[
              "import asyncio",
              "import cognee",
              "",
              "async def main():",
              ...(mode === "cloud" ? [
                '    # Connect to cloud first',
                '    await cognee.serve(',
                '        url="https://api.aws.cognee.ai",',
                '        api_key="your-api-key"',
                "    )",
                "",
              ] : []),
              `    await cognee.remember("${dbConfig.example}", datasets=["default_dataset"])`,
              "",
              "asyncio.run(main())",
            ]} />

            <Divider />
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              <span style={{ color: "#71717A", fontSize: 12, fontWeight: 500, letterSpacing: "0.04em", textTransform: "uppercase" }}>Optional: filter with SQL</span>
              <MultiLineCode step="Database" lines={[
                "# Inside an async function:",
                "await cognee.add(",
                `    "${dbConfig.example}",`,
                '    query="SELECT * FROM orders WHERE status = \'active\'"',
                ")",
              ]} />
            </div>
          </>
        )}

        {/* ── SAAS & APIS TAB ── */}
        {category === "saas" && (
          <>
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              <span style={{ color: "#71717A", fontSize: 12, fontWeight: 500, letterSpacing: "0.04em", textTransform: "uppercase" }}>Data source</span>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                {SAAS_SOURCES.map((s) => (
                  <button key={s.name} onClick={() => { trackEvent({ pageName: "Onboarding - Database", eventName: "saas_source_selected", additionalProperties: { source: s.name } }); setSaasSource(s.name); }} className="cursor-pointer" style={{
                    padding: "6px 14px", borderRadius: 6, fontSize: 13, fontWeight: 500, fontFamily: "inherit",
                    background: saasSource === s.name ? "#F0EDFF" : "#fff",
                    border: saasSource === s.name ? "1px solid #DDD6FE" : "1px solid #E4E4E7",
                    color: saasSource === s.name ? "#6510F4" : "#3F3F46",
                  }}>
                    {s.name}
                  </button>
                ))}
              </div>
            </div>

            <div style={{ display: "flex", gap: 10, background: "#F0EDFF", border: "1px solid #DDD6FE", borderRadius: 8, padding: "10px 14px" }}>
              <span style={{ fontSize: 12, color: "#52525B", lineHeight: "18px" }}>
                <strong>{saasConfig.name}</strong> extracts: {saasConfig.extracts}. Cognee converts the extracted data into a knowledge graph with entities, relationships, and embeddings.
              </span>
            </div>

            <div style={{ alignItems: "center", display: "flex", gap: 12 }}>
              <NumberCircle n={1} />
              <span style={{ color: "#18181B", fontSize: 14, fontWeight: 500 }}>Install</span>
            </div>
            <CodeBlock step="Database">{saasConfig.install}</CodeBlock>

            <Divider />

            <div style={{ alignItems: "center", display: "flex", gap: 12 }}>
              <NumberCircle n={2} />
              <span style={{ color: "#18181B", fontSize: 14, fontWeight: 500 }}>Create source and ingest</span>
            </div>
            <MultiLineCode step="Database" lines={[
              "import asyncio",
              "import cognee",
              "",
              "async def main():",
              ...(mode === "cloud" ? [
                '    await cognee.serve(',
                '        url="https://api.aws.cognee.ai",',
                '        api_key="your-api-key"',
                "    )",
                "",
              ] : []),
              ...saasConfig.code.map((l: string) => `    ${l}`),
              '    await cognee.cognify()  # Build knowledge graph',
              "",
              "asyncio.run(main())",
            ]} />
          </>
        )}

        {/* ── FILES & CSV TAB ── */}
        {category === "files" && (
          <>
            <div style={{ display: "flex", gap: 10, background: "#F0EDFF", border: "1px solid #DDD6FE", borderRadius: 8, padding: "10px 14px" }}>
              <span style={{ fontSize: 12, color: "#52525B", lineHeight: "18px" }}>
                Upload files directly, pass file paths, or point to CSV files. Cognee auto-detects CSV and treats it as structured data using DLT. Other files (PDF, DOCX, TXT, MD) are processed as documents.
              </span>
            </div>

            <div style={{ alignItems: "center", display: "flex", gap: 12 }}>
              <NumberCircle n={1} />
              <span style={{ color: "#18181B", fontSize: 14, fontWeight: 500 }}>Documents (PDF, DOCX, TXT, MD)</span>
            </div>
            <MultiLineCode step="Database" lines={[
              "import asyncio",
              "import cognee",
              "",
              "async def main():",
              ...(mode === "cloud" ? [
                '    await cognee.serve(',
                '        url="https://api.aws.cognee.ai",',
                '        api_key="your-api-key"',
                "    )",
                "",
              ] : []),
              '    # Single file',
              '    await cognee.remember("/path/to/report.pdf", datasets=["default_dataset"])',
              "",
              "    # Multiple files",
              "    await cognee.remember([",
              '        "/path/to/doc1.pdf",',
              '        "/path/to/doc2.docx",',
              '        "Some inline text content",',
              '    ], datasets=["default_dataset"])',
              "",
              "asyncio.run(main())",
            ]} />

            <Divider />

            <div style={{ alignItems: "center", display: "flex", gap: 12 }}>
              <NumberCircle n={2} />
              <span style={{ color: "#18181B", fontSize: 14, fontWeight: 500 }}>CSV files (structured data via DLT)</span>
            </div>
            <span style={{ color: "#71717A", fontSize: 12, lineHeight: "18px" }}>
              CSV files are auto-detected and ingested as structured tables with column types and relationships preserved.
            </span>
            <MultiLineCode step="Database" lines={[
              "# Inside an async function:",
              'await cognee.remember("/path/to/data.csv", datasets=["default_dataset"])',
            ]} />

            <Divider />

            <div style={{ alignItems: "center", display: "flex", gap: 12 }}>
              <NumberCircle n={3} />
              <span style={{ color: "#18181B", fontSize: 14, fontWeight: 500 }}>Cloud storage (S3, GCS)</span>
            </div>
            <MultiLineCode step="Database" lines={[
              "# Inside an async function:",
              'await cognee.remember("s3://my-bucket/documents/report.pdf", datasets=["default_dataset"])',
            ]} />
          </>
        )}
      </div>

      {!standalone && <NavButtons onBack={onBack} continueDisabled={false} onContinue={onSkip} pageName="Onboarding - Database" />}
    </div>
  );
}
