"use client";

import React, { useState } from "react";

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

function CopyBtn({ text, copied, onCopy }: { text: string; copied: boolean; onCopy: () => void }) {
  return (
    <button onClick={onCopy} className="cursor-pointer hover:opacity-80" style={{ background: copied ? "#22C55E22" : "#ffffff10", border: "none", borderRadius: 6, padding: "4px 10px", flexShrink: 0, display: "flex", alignItems: "center", gap: 4, transition: "background 150ms" }} title="Copy">
      {copied ? (
        <><svg width="12" height="12" viewBox="0 0 16 16" fill="none"><path d="M3.5 8.5L6.5 11.5L12.5 4.5" stroke="#22C55E" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" /></svg><span style={{ color: "#22C55E", fontSize: 11, fontWeight: 500 }}>Copied</span></>
      ) : (
        <><svg width="12" height="12" viewBox="0 0 16 16" fill="none"><rect x="5" y="5" width="8" height="8" rx="1.5" stroke="#9CA3AF" strokeWidth="1.5" /><path d="M11 3H4.5A1.5 1.5 0 003 4.5V11" stroke="#9CA3AF" strokeWidth="1.5" strokeLinecap="round" /></svg><span style={{ color: "#9CA3AF", fontSize: 11 }}>Copy</span></>
      )}
    </button>
  );
}

function CodeBlock({ children }: { children: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <div style={{ backgroundColor: "#0F0F10", borderRadius: 10, border: "1px solid #2A2A2E", display: "flex", alignItems: "center", justifyContent: "space-between", paddingBlock: 14, paddingInline: 20, gap: 12 }}>
      <code style={{ color: "#E4E4E7", fontFamily: '"Fira Code", "SF Mono", "Courier New", monospace', fontSize: 13.5, lineHeight: "20px", wordBreak: "break-all" }}>{colorizeLine(children)}</code>
      <CopyBtn text={children} copied={copied} onCopy={() => { navigator.clipboard.writeText(children); setCopied(true); setTimeout(() => setCopied(false), 2000); }} />
    </div>
  );
}

function MultiLineCode({ lines }: { lines: string[] }) {
  const full = lines.join("\n");
  const [copied, setCopied] = useState(false);
  return (
    <div style={{ backgroundColor: "#0F0F10", borderRadius: 10, border: "1px solid #2A2A2E", display: "flex", flexDirection: "column", paddingBlock: 16, paddingInline: 20, position: "relative", gap: 0 }}>
      <div style={{ position: "absolute", top: 12, right: 12 }}>
        <CopyBtn text={full} copied={copied} onCopy={() => { navigator.clipboard.writeText(full); setCopied(true); setTimeout(() => setCopied(false), 2000); }} />
      </div>
      {lines.map((line, i) => (
        <div key={i} style={{ display: "flex", gap: 16, minHeight: 22 }}>
          <span style={{ color: "#4A4A4F", fontFamily: '"Fira Code", "SF Mono", monospace', fontSize: 12, lineHeight: "22px", userSelect: "none", width: 20, textAlign: "right", flexShrink: 0 }}>{i + 1}</span>
          <code style={{ fontFamily: '"Fira Code", "SF Mono", "Courier New", monospace', fontSize: 13.5, lineHeight: "22px", whiteSpace: "pre" }}>{colorizeLine(line)}</code>
        </div>
      ))}
    </div>
  );
}

function TokenDisplay({ label, token }: { label: string; token: string }) {
  const masked = token.length > 8 ? token.slice(0, 6) + "****" + token.slice(-4) : token || "sk-cog-****3f8a";
  const [copied, setCopied] = useState(false);
  return (
    <div style={{ alignItems: "center", backgroundColor: "#F4F4F5", borderRadius: 8, display: "flex", gap: 12, paddingBlock: 12, paddingInline: 16 }}>
      <span style={{ color: "#71717A", fontFamily: '"Inter", system-ui, sans-serif', fontSize: 13, flexShrink: 0 }}>{label}:</span>
      <span style={{ color: "#18181B", fontFamily: '"Fira Code", "Courier New", monospace', fontSize: 13 }}>{masked}</span>
      <button onClick={() => { navigator.clipboard.writeText(token || masked); setCopied(true); setTimeout(() => setCopied(false), 2000); }} className="cursor-pointer" style={{ background: "none", border: "none", padding: 0, marginLeft: "auto", flexShrink: 0 }}>
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

function NavButtons({ onBack, onContinue, continueDisabled }: { onBack: () => void; onContinue?: () => void; continueDisabled?: boolean }) {
  return (
    <div style={{ display: "flex", gap: 12 }}>
      <button onClick={onBack} className="cursor-pointer" style={{ alignItems: "center", borderColor: "#E4E4E7", borderRadius: 8, borderStyle: "solid", borderWidth: 1, background: "white", display: "flex", height: 40, justifyContent: "center", paddingInline: 20 }}>
        <span style={{ color: "#3F3F46", fontFamily: '"Inter", system-ui, sans-serif', fontSize: 13, fontWeight: 500 }}>Back</span>
      </button>
      {onContinue && (
        <button onClick={onContinue} disabled={continueDisabled} className="cursor-pointer" style={{ alignItems: "center", backgroundColor: continueDisabled ? "#E4E4E7" : "#6510F4", borderRadius: 8, border: "none", display: "flex", height: 40, justifyContent: "center", paddingInline: 20, opacity: continueDisabled ? 0.6 : 1 }}>
          <span style={{ color: continueDisabled ? "#A1A1AA" : "#FFFFFF", fontFamily: '"Inter", system-ui, sans-serif', fontSize: 13, fontWeight: 500 }}>Continue</span>
        </button>
      )}
    </div>
  );
}

function SkipLink({ onClick }: { onClick: () => void }) {
  return (
    <button onClick={onClick} className="cursor-pointer" style={{ background: "none", border: "none", color: "#9CA3AF", fontFamily: '"Inter", system-ui, sans-serif', fontSize: 13, paddingTop: 16 }}>
      Skip onboarding and go to dashboard
    </button>
  );
}

// ── Connect Local Cognee SDK ──

export function LocalCogneeStep({ onBack, onSkip }: { onBack: () => void; onSkip: () => void }) {
  const [connectMode, setConnectMode] = useState<"cloud" | "direct">("cloud");

  return (
    <div style={{ alignItems: "center", display: "flex", flexDirection: "column", flexGrow: 1, gap: 32, paddingBlock: 48, paddingInline: 40, fontFamily: '"Inter", system-ui, sans-serif', overflowY: "auto" }}>
      <div style={{ alignItems: "center", display: "flex", flexDirection: "column", gap: 8 }}>
        <StepBadge step={1} />
        <h1 style={{ color: "#18181B", fontSize: 28, fontWeight: 600, lineHeight: "34px", margin: 0 }}>Connect your local Cognee SDK</h1>
        <p style={{ color: "#71717A", fontSize: 15, lineHeight: "22px", margin: 0, textAlign: "center", maxWidth: 540 }}>
          Use <code style={{ background: "#F4F4F5", padding: "1px 6px", borderRadius: 4, fontSize: 13 }}>cognee.serve()</code> to link your local Python SDK to a remote Cognee instance. All operations (remember, recall, forget, improve) will route to the connected instance.
        </p>
      </div>

      <div style={{ backgroundColor: "#FFFFFF", borderColor: "#E4E4E7", borderRadius: 12, borderStyle: "solid", borderWidth: 1, display: "flex", flexDirection: "column", gap: 20, paddingBlock: 24, paddingInline: 24, width: 620 }}>

        {/* Connection mode toggle */}
        <div style={{ display: "flex", gap: 0, borderRadius: 8, border: "1px solid #E4E4E7", overflow: "hidden" }}>
          <button onClick={() => setConnectMode("cloud")} className="cursor-pointer" style={{ flex: 1, padding: "10px 16px", background: connectMode === "cloud" ? "#F0EDFF" : "#fff", border: "none", fontSize: 13, fontWeight: 500, color: connectMode === "cloud" ? "#6510F4" : "#71717A", fontFamily: "inherit", borderRight: "1px solid #E4E4E7" }}>
            Connect to Cognee Cloud
          </button>
          <button onClick={() => setConnectMode("direct")} className="cursor-pointer" style={{ flex: 1, padding: "10px 16px", background: connectMode === "direct" ? "#F0EDFF" : "#fff", border: "none", fontSize: 13, fontWeight: 500, color: connectMode === "direct" ? "#6510F4" : "#71717A", fontFamily: "inherit" }}>
            Connect to any instance
          </button>
        </div>

        {connectMode === "cloud" ? (
          <>
            {/* CLOUD: device code login */}
            <div style={{ display: "flex", gap: 10, background: "#F0EDFF", border: "1px solid #DDD6FE", borderRadius: 8, padding: "10px 14px" }}>
              <span style={{ fontSize: 12, color: "#52525B", lineHeight: "18px" }}>
                <strong>One command.</strong> Cognee opens your browser for login, discovers your cloud tenant, creates an API key, and connects — all automatically. Your credentials are saved locally for future sessions.
              </span>
            </div>

            <div style={{ alignItems: "center", display: "flex", gap: 12 }}>
              <NumberCircle n={1} />
              <span style={{ color: "#18181B", fontSize: 14, fontWeight: 500 }}>Install Cognee</span>
            </div>
            <CodeBlock>pip install cognee</CodeBlock>

            <Divider />

            <div style={{ alignItems: "center", display: "flex", gap: 12 }}>
              <NumberCircle n={2} />
              <span style={{ color: "#18181B", fontSize: 14, fontWeight: 500 }}>Connect to cloud</span>
            </div>
            <span style={{ color: "#71717A", fontSize: 12, lineHeight: "18px" }}>
              This opens a browser window for authentication. After login, your tenant is discovered automatically and the SDK connects to it.
            </span>
            <MultiLineCode lines={[
              "import cognee",
              "",
              "await cognee.serve()  # Opens browser for login",
              "# Connected to Cognee Cloud at https://tenant-xxx.cognee.ai",
            ]} />

            <Divider />

            <div style={{ alignItems: "center", display: "flex", gap: 12 }}>
              <NumberCircle n={3} />
              <span style={{ color: "#18181B", fontSize: 14, fontWeight: 500 }}>Use Cognee — everything routes to cloud</span>
            </div>
            <MultiLineCode lines={[
              "# Add data — ingested into your cloud tenant",
              'await cognee.remember("Einstein developed general relativity in 1915.")',
              "",
              "# Search — queries your cloud knowledge graph",
              'results = await cognee.recall("What did Einstein develop?")',
              "",
              "# Disconnect when done (credentials saved for next time)",
              "await cognee.disconnect()",
            ]} />

            <Divider />

            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              <span style={{ color: "#71717A", fontSize: 12, fontWeight: 500, letterSpacing: "0.04em", textTransform: "uppercase" }}>Re-connecting</span>
              <span style={{ color: "#A1A1AA", fontSize: 12, lineHeight: "18px" }}>
                Credentials are saved at <code style={{ fontSize: 11 }}>~/.cognee/cloud_credentials.json</code>. Next time you call <code style={{ fontSize: 11 }}>cognee.serve()</code>, it reconnects instantly without re-authenticating (until the token expires).
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
            <CodeBlock>pip install cognee</CodeBlock>

            <Divider />

            <div style={{ alignItems: "center", display: "flex", gap: 12 }}>
              <NumberCircle n={2} />
              <span style={{ color: "#18181B", fontSize: 14, fontWeight: 500 }}>Connect with URL and API key</span>
            </div>
            <MultiLineCode lines={[
              "import cognee",
              "",
              "# Connect to a specific instance",
              'await cognee.serve(',
              '    url="https://your-instance.cognee.ai",',
              '    api_key="your-api-key"',
              ")",
              "",
              "# Or use environment variables",
              "# COGNEE_SERVICE_URL=https://your-instance.cognee.ai",
              "# COGNEE_API_KEY=your-api-key",
              "await cognee.serve()  # Reads from env",
            ]} />

            <Divider />

            <div style={{ alignItems: "center", display: "flex", gap: 12 }}>
              <NumberCircle n={3} />
              <span style={{ color: "#18181B", fontSize: 14, fontWeight: 500 }}>Use Cognee — everything routes to the instance</span>
            </div>
            <MultiLineCode lines={[
              '# Add and process data on the remote instance',
              'await cognee.remember("Your data here", dataset_name="my_project")',
              "",
              '# Query the remote knowledge graph',
              'results = await cognee.recall("What do we know about X?")',
              "",
              "# Disconnect",
              "await cognee.disconnect()",
            ]} />

            <Divider />

            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              <span style={{ color: "#71717A", fontSize: 12, fontWeight: 500, letterSpacing: "0.04em", textTransform: "uppercase" }}>Local to local</span>
              <span style={{ color: "#A1A1AA", fontSize: 12, lineHeight: "18px" }}>
                To connect to a Cognee backend running on the same machine or your local network:
              </span>
              <MultiLineCode lines={[
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

      <NavButtons onBack={onBack} continueDisabled={false} onContinue={onSkip} />
    </div>
  );
}

// ── Agent Connection ──

export function AgentStep({ onBack, onSkip }: { onBack: () => void; onSkip: () => void }) {
  const [method, setMethod] = useState<"sdk" | "mcp" | "api">("sdk");

  return (
    <div style={{ alignItems: "center", display: "flex", flexDirection: "column", flexGrow: 1, gap: 32, paddingBlock: 48, paddingInline: 40, fontFamily: '"Inter", system-ui, sans-serif', overflowY: "auto" }}>
      <div style={{ alignItems: "center", display: "flex", flexDirection: "column", gap: 8 }}>
        <StepBadge step={1} />
        <h1 style={{ color: "#18181B", fontSize: 28, fontWeight: 600, lineHeight: "34px", margin: 0 }}>Connect your Agent</h1>
        <p style={{ color: "#71717A", fontSize: 15, lineHeight: "22px", margin: 0, textAlign: "center", maxWidth: 520 }}>
          Give any agent memory by connecting it to Cognee. Choose how your agent should integrate.
        </p>
      </div>

      <div style={{ backgroundColor: "#FFFFFF", borderColor: "#E4E4E7", borderRadius: 12, borderStyle: "solid", borderWidth: 1, display: "flex", flexDirection: "column", gap: 20, paddingBlock: 24, paddingInline: 24, width: 620 }}>

        {/* Method tabs */}
        <div style={{ display: "flex", gap: 0, borderRadius: 8, border: "1px solid #E4E4E7", overflow: "hidden" }}>
          <button onClick={() => setMethod("sdk")} className="cursor-pointer" style={{ flex: 1, padding: "10px 16px", background: method === "sdk" ? "#F0EDFF" : "#fff", border: "none", fontSize: 13, fontWeight: 500, color: method === "sdk" ? "#6510F4" : "#71717A", fontFamily: "inherit", borderRight: "1px solid #E4E4E7" }}>
            Python SDK
          </button>
          <button onClick={() => setMethod("mcp")} className="cursor-pointer" style={{ flex: 1, padding: "10px 16px", background: method === "mcp" ? "#F0EDFF" : "#fff", border: "none", fontSize: 13, fontWeight: 500, color: method === "mcp" ? "#6510F4" : "#71717A", fontFamily: "inherit", borderRight: "1px solid #E4E4E7" }}>
            MCP Server
          </button>
          <button onClick={() => setMethod("api")} className="cursor-pointer" style={{ flex: 1, padding: "10px 16px", background: method === "api" ? "#F0EDFF" : "#fff", border: "none", fontSize: 13, fontWeight: 500, color: method === "api" ? "#6510F4" : "#71717A", fontFamily: "inherit" }}>
            REST API
          </button>
        </div>

        {/* ── Python SDK ── */}
        {method === "sdk" && (
          <>
            <div style={{ display: "flex", gap: 10, background: "#F0EDFF", border: "1px solid #DDD6FE", borderRadius: 8, padding: "10px 14px" }}>
              <span style={{ fontSize: 12, color: "#52525B", lineHeight: "18px" }}>
                The Cognee Python SDK works with any agent framework. Use <code style={{ background: "#E4DEFF", padding: "1px 4px", borderRadius: 3, fontSize: 11 }}>cognee.serve()</code> to connect, then <code style={{ background: "#E4DEFF", padding: "1px 4px", borderRadius: 3, fontSize: 11 }}>remember</code> / <code style={{ background: "#E4DEFF", padding: "1px 4px", borderRadius: 3, fontSize: 11 }}>recall</code> from your agent code.
              </span>
            </div>

            <div style={{ alignItems: "center", display: "flex", gap: 12 }}>
              <NumberCircle n={1} />
              <span style={{ color: "#18181B", fontSize: 14, fontWeight: 500 }}>Install Cognee</span>
            </div>
            <CodeBlock>pip install cognee</CodeBlock>

            <Divider />

            <div style={{ alignItems: "center", display: "flex", gap: 12 }}>
              <NumberCircle n={2} />
              <span style={{ color: "#18181B", fontSize: 14, fontWeight: 500 }}>Connect and give your agent memory</span>
            </div>
            <MultiLineCode lines={[
              "import cognee",
              "",
              "# Connect to Cognee (cloud or self-hosted)",
              "await cognee.serve()  # Cloud: browser login",
              '# await cognee.serve(url="http://localhost:8000")  # Local',
              "",
              "# Store knowledge from your agent",
              'await cognee.remember("User prefers dark mode and concise answers.")',
              "",
              "# Retrieve relevant context for your agent",
              'results = await cognee.recall("What are the user preferences?")',
              "",
              "# Use results in your agent's prompt/context",
              "for item in results:",
              '    print(item["search_result"])',
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
          <>
            <div style={{ display: "flex", gap: 10, background: "#F0EDFF", border: "1px solid #DDD6FE", borderRadius: 8, padding: "10px 14px" }}>
              <span style={{ fontSize: 12, color: "#52525B", lineHeight: "18px" }}>
                Cognee runs as an <a href="https://modelcontextprotocol.io" target="_blank" rel="noopener noreferrer" style={{ color: "#6510F4", textDecoration: "underline" }}>MCP (Model Context Protocol)</a> server. Any MCP-compatible client (Claude Desktop, Cursor, VS Code Copilot, etc.) can connect to it as a tool provider.
              </span>
            </div>

            <div style={{ alignItems: "center", display: "flex", gap: 12 }}>
              <NumberCircle n={1} />
              <span style={{ color: "#18181B", fontSize: 14, fontWeight: 500 }}>Start the MCP server</span>
            </div>
            <span style={{ color: "#71717A", fontSize: 12, lineHeight: "18px" }}>
              The MCP server starts automatically with <code style={{ fontSize: 11 }}>cognee -ui</code>, or run it standalone:
            </span>
            <CodeBlock>cognee mcp --transport sse --port 8001</CodeBlock>

            <Divider />

            <div style={{ alignItems: "center", display: "flex", gap: 12 }}>
              <NumberCircle n={2} />
              <span style={{ color: "#18181B", fontSize: 14, fontWeight: 500 }}>Add to your MCP client config</span>
            </div>
            <span style={{ color: "#71717A", fontSize: 12, lineHeight: "18px" }}>
              Add Cognee as a tool server in your MCP client's configuration:
            </span>
            <MultiLineCode lines={[
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
                { tool: "cognee_add", desc: "Ingest text or files into a dataset" },
                { tool: "cognee_cognify", desc: "Build the knowledge graph" },
                { tool: "cognee_search", desc: "Query the knowledge graph" },
                { tool: "cognee_get_datasets", desc: "List available datasets" },
              ].map((t) => (
                <div key={t.tool} style={{ display: "flex", alignItems: "center", gap: 10, padding: "6px 0" }}>
                  <code style={{ background: "#F4F4F5", padding: "2px 8px", borderRadius: 4, fontSize: 12, color: "#6510F4", fontWeight: 500 }}>{t.tool}</code>
                  <span style={{ fontSize: 12, color: "#71717A" }}>{t.desc}</span>
                </div>
              ))}
            </div>
          </>
        )}

        {/* ── REST API ── */}
        {method === "api" && (
          <>
            <div style={{ display: "flex", gap: 10, background: "#F0EDFF", border: "1px solid #DDD6FE", borderRadius: 8, padding: "10px 14px" }}>
              <span style={{ fontSize: 12, color: "#52525B", lineHeight: "18px" }}>
                Use the REST API directly from any language or framework. Authenticate with an API key and call the endpoints.
              </span>
            </div>

            <div style={{ alignItems: "center", display: "flex", gap: 12 }}>
              <NumberCircle n={1} />
              <span style={{ color: "#18181B", fontSize: 14, fontWeight: 500 }}>Get your API key</span>
            </div>
            <span style={{ color: "#71717A", fontSize: 12, lineHeight: "18px" }}>
              Create an API key from the <a href="/api-keys" style={{ color: "#6510F4", textDecoration: "underline" }}>API Keys</a> page, or register your agent programmatically:
            </span>
            <MultiLineCode lines={[
              "# Register an agent user",
              'curl -X POST http://localhost:8000/api/v1/auth/register \\',
              '  -H "Content-Type: application/json" \\',
              '  -d \'{"email": "MyAgent-001@cognee.agent",',
              '       "password": "secret", "is_verified": true}\'',
              "",
              "# Login and get a token",
              'curl -X POST http://localhost:8000/api/v1/auth/login \\',
              '  -d "username=MyAgent-001@cognee.agent&password=secret"',
              "",
              "# Create an API key (use the token from login)",
              'curl -X POST http://localhost:8000/api/v1/auth/api-keys \\',
              '  -H "Authorization: Bearer <token>" \\',
              '  -H "Content-Type: application/json" \\',
              '  -d \'{"name": "MyAgent-001"}\'',
            ]} />

            <Divider />

            <div style={{ alignItems: "center", display: "flex", gap: 12 }}>
              <NumberCircle n={2} />
              <span style={{ color: "#18181B", fontSize: 14, fontWeight: 500 }}>Add data and query</span>
            </div>
            <MultiLineCode lines={[
              "# Add data",
              'curl -X POST http://localhost:8000/api/v1/add \\',
              '  -H "X-Api-Key: <your-key>" \\',
              '  -F "data=@document.pdf" \\',
              '  -F "datasetName=my_agent_data"',
              "",
              "# Build knowledge graph",
              'curl -X POST http://localhost:8000/api/v1/cognify \\',
              '  -H "X-Api-Key: <your-key>" \\',
              '  -H "Content-Type: application/json" \\',
              '  -d \'{"datasets": ["my_agent_data"]}\'',
              "",
              "# Search",
              'curl -X POST http://localhost:8000/api/v1/search \\',
              '  -H "X-Api-Key: <your-key>" \\',
              '  -H "Content-Type: application/json" \\',
              '  -d \'{"query": "What do we know?"}\'',
            ]} />
          </>
        )}
      </div>

      <NavButtons onBack={onBack} continueDisabled={false} onContinue={onSkip} />
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

export function DatabaseStep({ onBack, onSkip }: { onBack: () => void; onSkip: () => void }) {
  const [category, setCategory] = useState<SourceCategory>("database");
  const [dbType, setDbType] = useState("PostgreSQL");
  const [saasSource, setSaasSource] = useState(SAAS_SOURCES[0].name);
  const [mode, setMode] = useState<"local" | "cloud">("local");

  const dbConfig = DB_CONFIGS[dbType];
  const saasConfig = SAAS_SOURCES.find((s) => s.name === saasSource) || SAAS_SOURCES[0];

  return (
    <div style={{ alignItems: "center", display: "flex", flexDirection: "column", flexGrow: 1, gap: 32, paddingBlock: 48, paddingInline: 40, fontFamily: '"Inter", system-ui, sans-serif', overflowY: "auto" }}>
      <div style={{ alignItems: "center", display: "flex", flexDirection: "column", gap: 8 }}>
        <StepBadge step={1} />
        <h1 style={{ color: "#18181B", fontSize: 28, fontWeight: 600, lineHeight: "34px", margin: 0 }}>Ingest from Any Source</h1>
        <p style={{ color: "#71717A", fontSize: 15, lineHeight: "22px", margin: 0, textAlign: "center", maxWidth: 560 }}>
          Cognee uses <a href="https://dlthub.com" target="_blank" rel="noopener noreferrer" style={{ color: "#6510F4", textDecoration: "underline" }}>dlt (data load tool)</a> to connect to databases, SaaS platforms, and APIs. Data is extracted, loaded into a knowledge graph with schema and relationships preserved, and made searchable.
        </p>
      </div>

      <div style={{ backgroundColor: "#FFFFFF", borderColor: "#E4E4E7", borderRadius: 12, borderStyle: "solid", borderWidth: 1, display: "flex", flexDirection: "column", gap: 20, paddingBlock: 24, paddingInline: 24, width: 640 }}>

        {/* Source category tabs */}
        <div style={{ display: "flex", gap: 0, borderRadius: 8, border: "1px solid #E4E4E7", overflow: "hidden" }}>
          <SourceCategoryTab label="Databases" active={category === "database"} onClick={() => setCategory("database")} />
          <SourceCategoryTab label="SaaS & APIs" active={category === "saas"} onClick={() => setCategory("saas")} />
          <SourceCategoryTab label="Files & CSV" active={category === "files"} onClick={() => setCategory("files")} />
        </div>

        {/* Mode toggle */}
        <div style={{ display: "flex", gap: 0, borderRadius: 8, border: "1px solid #E4E4E7", overflow: "hidden" }}>
          <button onClick={() => setMode("local")} className="cursor-pointer" style={{ flex: 1, padding: "8px 14px", background: mode === "local" ? "#F4F4F5" : "#fff", border: "none", fontSize: 12, fontWeight: 500, color: mode === "local" ? "#18181B" : "#A1A1AA", fontFamily: "inherit", borderRight: "1px solid #E4E4E7" }}>
            Local / Self-hosted
          </button>
          <button onClick={() => setMode("cloud")} className="cursor-pointer" style={{ flex: 1, padding: "8px 14px", background: mode === "cloud" ? "#F4F4F5" : "#fff", border: "none", fontSize: 12, fontWeight: 500, color: mode === "cloud" ? "#18181B" : "#A1A1AA", fontFamily: "inherit" }}>
            Cognee Cloud
          </button>
        </div>

        {mode === "cloud" && (
          <div style={{ display: "flex", gap: 10, background: "#FEF3C7", border: "1px solid #FDE68A", borderRadius: 8, padding: "10px 14px", alignItems: "flex-start" }}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#D97706" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0, marginTop: 2 }}>
              <circle cx="12" cy="12" r="10" /><line x1="12" y1="8" x2="12" y2="12" /><line x1="12" y1="16" x2="12.01" y2="16" />
            </svg>
            <span style={{ fontSize: 12, color: "#92400E", lineHeight: "18px" }}>
              For cloud, the Cognee SDK runs locally to access your data sources, then pushes extracted data to your cloud tenant. Run <code style={{ background: "#FEF3C7", fontSize: 11 }}>await cognee.serve()</code> first to connect.
            </span>
          </div>
        )}

        <Divider />

        {/* ── DATABASE TAB ── */}
        {category === "database" && (
          <>
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              <span style={{ color: "#71717A", fontSize: 12, fontWeight: 500, letterSpacing: "0.04em", textTransform: "uppercase" }}>Database type</span>
              <select value={dbType} onChange={(e) => setDbType(e.target.value)} style={{ backgroundColor: "#fff", borderColor: "#E4E4E7", borderRadius: 8, borderStyle: "solid", borderWidth: 1, height: 38, paddingInline: 14, fontSize: 13, fontWeight: 500, color: "#18181B", outline: "none" }}>
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
            <MultiLineCode lines={[
              '# Install DLT support',
              'pip install "cognee[dlt]"',
              "",
              "import cognee",
              ...(mode === "cloud" ? ['await cognee.serve()  # Connect to cloud first', ""] : []),
              `await cognee.remember("${dbConfig.example}", dataset_name="my_db")`,
            ]} />

            <Divider />
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              <span style={{ color: "#71717A", fontSize: 12, fontWeight: 500, letterSpacing: "0.04em", textTransform: "uppercase" }}>Optional: filter with SQL</span>
              <MultiLineCode lines={[
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
                  <button key={s.name} onClick={() => setSaasSource(s.name)} className="cursor-pointer" style={{
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
            <CodeBlock>{saasConfig.install}</CodeBlock>

            <Divider />

            <div style={{ alignItems: "center", display: "flex", gap: 12 }}>
              <NumberCircle n={2} />
              <span style={{ color: "#18181B", fontSize: 14, fontWeight: 500 }}>Create source and ingest</span>
            </div>
            <MultiLineCode lines={[
              "import cognee",
              ...(mode === "cloud" ? ["await cognee.serve()  # Connect to cloud", ""] : []),
              ...saasConfig.code,
              'await cognee.cognify()  # Build knowledge graph',
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
            <MultiLineCode lines={[
              "import cognee",
              ...(mode === "cloud" ? ["await cognee.serve()", ""] : []),
              '# Single file',
              'await cognee.remember("/path/to/report.pdf")',
              "",
              "# Multiple files",
              "await cognee.remember([",
              '    "/path/to/doc1.pdf",',
              '    "/path/to/doc2.docx",',
              '    "Some inline text content",',
              "])",
            ]} />

            <Divider />

            <div style={{ alignItems: "center", display: "flex", gap: 12 }}>
              <NumberCircle n={2} />
              <span style={{ color: "#18181B", fontSize: 14, fontWeight: 500 }}>CSV files (structured data via DLT)</span>
            </div>
            <span style={{ color: "#71717A", fontSize: 12, lineHeight: "18px" }}>
              CSV files are auto-detected and ingested as structured tables with column types and relationships preserved.
            </span>
            <MultiLineCode lines={[
              '# Auto-detected as structured data',
              'await cognee.remember("/path/to/data.csv", dataset_name="sales")',
            ]} />

            <Divider />

            <div style={{ alignItems: "center", display: "flex", gap: 12 }}>
              <NumberCircle n={3} />
              <span style={{ color: "#18181B", fontSize: 14, fontWeight: 500 }}>Cloud storage (S3, GCS)</span>
            </div>
            <MultiLineCode lines={[
              '# S3 path',
              'await cognee.remember("s3://my-bucket/documents/report.pdf")',
            ]} />
          </>
        )}
      </div>

      <NavButtons onBack={onBack} continueDisabled={false} onContinue={onSkip} />
    </div>
  );
}
