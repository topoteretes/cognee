"use client";

import { useState } from "react";

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

function CodeBlock({ children }: { children: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <div style={{ alignItems: "center", backgroundColor: "#18181B", borderRadius: 8, display: "flex", justifyContent: "space-between", paddingBlock: 14, paddingInline: 20 }}>
      <span style={{ color: "#A1A1AA", fontFamily: '"Fira Code", "Courier New", monospace', fontSize: 13, lineHeight: "18px" }}>{children}</span>
      <button onClick={() => { navigator.clipboard.writeText(children); setCopied(true); setTimeout(() => setCopied(false), 2000); }} className="cursor-pointer" style={{ background: "none", border: "none", padding: 0, flexShrink: 0 }}>
        {copied ? <span style={{ color: "#22C55E", fontSize: 11 }}>Copied</span> : (
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none"><rect x="5" y="5" width="8" height="8" rx="1.5" stroke="#71717A" strokeWidth="1.5" /><path d="M11 3H4.5A1.5 1.5 0 003 4.5V11" stroke="#71717A" strokeWidth="1.5" strokeLinecap="round" /></svg>
        )}
      </button>
    </div>
  );
}

function MultiLineCode({ lines }: { lines: string[] }) {
  const full = lines.join("\n");
  const [copied, setCopied] = useState(false);
  return (
    <div style={{ backgroundColor: "#18181B", borderRadius: 8, display: "flex", flexDirection: "column", paddingBlock: 16, paddingInline: 20, position: "relative" }}>
      {lines.map((line, i) => (
        <span key={i} style={{ color: line.includes(":") && !line.startsWith("export") ? "#71717A" : "#A1A1AA", fontFamily: '"Fira Code", "Courier New", monospace', fontSize: 12, lineHeight: "20px" }}>{line}</span>
      ))}
      <button onClick={() => { navigator.clipboard.writeText(full); setCopied(true); setTimeout(() => setCopied(false), 2000); }} className="cursor-pointer" style={{ background: "none", border: "none", padding: 0, position: "absolute", top: 16, right: 20, flexShrink: 0 }}>
        {copied ? <span style={{ color: "#22C55E", fontSize: 11 }}>Copied</span> : (
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none"><rect x="5" y="5" width="8" height="8" rx="1.5" stroke="#71717A" strokeWidth="1.5" /><path d="M11 3H4.5A1.5 1.5 0 003 4.5V11" stroke="#71717A" strokeWidth="1.5" strokeLinecap="round" /></svg>
        )}
      </button>
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

// ── Local Cognee Connection ──

export function LocalCogneeStep({ onBack, onSkip }: { onBack: () => void; onSkip: () => void }) {
  return (
    <div style={{ alignItems: "center", display: "flex", flexDirection: "column", flexGrow: 1, gap: 32, paddingBlock: 48, paddingInline: 80, fontFamily: '"Inter", system-ui, sans-serif' }}>
      <div style={{ alignItems: "center", display: "flex", flexDirection: "column", gap: 8 }}>
        <StepBadge step={1} />
        <h1 style={{ color: "#18181B", fontSize: 28, fontWeight: 600, lineHeight: "34px", margin: 0 }}>Connect local Cognee</h1>
        <p style={{ color: "#71717A", fontSize: 15, lineHeight: "22px", margin: 0, textAlign: "center" }}>Sync your local datasets and memory to Cognee Cloud.</p>
      </div>

      <div style={{ backgroundColor: "#FFFFFF", borderColor: "#E4E4E7", borderRadius: 12, borderStyle: "solid", borderWidth: 1, display: "flex", flexDirection: "column", gap: 24, paddingBlock: 28, paddingInline: 28, width: 560 }}>
        <div style={{ alignItems: "center", display: "flex", gap: 12 }}>
          <NumberCircle n={1} />
          <span style={{ color: "#18181B", fontSize: 14, fontWeight: 500 }}>Copy your Cloud auth token</span>
        </div>
        <TokenDisplay label="Your token" token="" />

        <Divider />

        <div style={{ alignItems: "center", display: "flex", gap: 12 }}>
          <NumberCircle n={2} />
          <span style={{ color: "#18181B", fontSize: 14, fontWeight: 500 }}>Add it to your local .env file</span>
        </div>
        <CodeBlock>COGNEE_CLOUD_AUTH_TOKEN=sk-cog-...3f8a</CodeBlock>

        <Divider />

        <div style={{ alignItems: "center", display: "flex", gap: 12 }}>
          <NumberCircle n={3} />
          <span style={{ color: "#18181B", fontSize: 14, fontWeight: 500 }}>Start your local server and sync</span>
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          <CodeBlock>cognee serve</CodeBlock>
          <span style={{ color: "#A1A1AA", fontSize: 12 }}>Then trigger sync: curl -X POST http://localhost:8000/api/v1/sync</span>
        </div>

        <Divider />
        <WaitingIndicator text="Waiting for your local Cognee to sync..." />
      </div>

      <NavButtons onBack={onBack} continueDisabled />
      <SkipLink onClick={onSkip} />
    </div>
  );
}

// ── Agent Connection (OpenClaw) ──

export function AgentStep({ onBack, onSkip }: { onBack: () => void; onSkip: () => void }) {
  const [framework, setFramework] = useState("OpenClaw");
  const frameworks = ["OpenClaw", "CrewAI", "LangGraph", "AutoGen", "Custom"];

  return (
    <div style={{ alignItems: "center", display: "flex", flexDirection: "column", flexGrow: 1, gap: 32, paddingBlock: 48, paddingInline: 80, fontFamily: '"Inter", system-ui, sans-serif' }}>
      <div style={{ alignItems: "center", display: "flex", flexDirection: "column", gap: 8 }}>
        <StepBadge step={1} />
        <h1 style={{ color: "#18181B", fontSize: 28, fontWeight: 600, lineHeight: "34px", margin: 0 }}>Connect your Agent</h1>
        <p style={{ color: "#71717A", fontSize: 15, lineHeight: "22px", margin: 0, textAlign: "center", maxWidth: 480 }}>Choose your agent framework and follow the setup instructions to connect to Cognee Cloud.</p>
      </div>

      <div style={{ backgroundColor: "#FFFFFF", borderColor: "#E4E4E7", borderRadius: 12, borderStyle: "solid", borderWidth: 1, display: "flex", flexDirection: "column", gap: 24, paddingBlock: 28, paddingInline: 28, width: 560 }}>
        {/* Framework selector */}
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          <span style={{ color: "#71717A", fontSize: 12, fontWeight: 500, letterSpacing: "0.04em", textTransform: "uppercase" }}>Agent framework</span>
          <select
            value={framework}
            onChange={(e) => setFramework(e.target.value)}
            style={{ alignItems: "center", backgroundColor: "#FFFFFF", borderColor: "#E4E4E7", borderRadius: 8, borderStyle: "solid", borderWidth: 1, height: 40, paddingInline: 14, fontSize: 14, fontWeight: 500, color: "#18181B", outline: "none" }}
          >
            {frameworks.map((f) => <option key={f} value={f}>{f}</option>)}
          </select>
          <div style={{ alignItems: "center", display: "flex", gap: 8 }}>
            <span style={{ color: "#A1A1AA", fontSize: 12 }}>Also available:</span>
            <div style={{ display: "flex", gap: 6 }}>
              {frameworks.filter((f) => f !== framework).map((f) => (
                <button key={f} onClick={() => setFramework(f)} className="cursor-pointer" style={{ backgroundColor: "#F4F4F5", borderRadius: 4, border: "none", paddingBlock: 2, paddingInline: 8 }}>
                  <span style={{ color: "#71717A", fontSize: 11 }}>{f}</span>
                </button>
              ))}
            </div>
          </div>
        </div>

        <Divider />

        <div style={{ alignItems: "center", display: "flex", gap: 12 }}>
          <NumberCircle n={1} />
          <span style={{ color: "#18181B", fontSize: 14, fontWeight: 500 }}>Install the Cognee {framework} plugin</span>
        </div>
        <CodeBlock>{`pip install cognee-${framework.toLowerCase()}`}</CodeBlock>

        <Divider />

        <div style={{ alignItems: "center", display: "flex", gap: 12 }}>
          <NumberCircle n={2} />
          <span style={{ color: "#18181B", fontSize: 14, fontWeight: 500 }}>Copy your Cloud API key</span>
        </div>
        <TokenDisplay label="API key" token="" />

        <Divider />

        <div style={{ alignItems: "center", display: "flex", gap: 12 }}>
          <NumberCircle n={3} />
          <span style={{ color: "#18181B", fontSize: 14, fontWeight: 500 }}>Add to your {framework} config</span>
        </div>
        <MultiLineCode lines={[
          "plugins:",
          "  entries:",
          `    cognee-${framework.toLowerCase()}:`,
          "      enabled: true",
          "    config:",
          '      mode: "cloud"',
          '      baseUrl: "https://tenant-xxx.cloud.cognee.ai/api"',
          '      apiKey: "${COGNEE_API_KEY}"',
        ]} />

        {/* Or env vars */}
        <div style={{ alignItems: "center", display: "flex", gap: 12 }}>
          <div style={{ backgroundColor: "#E4E4E7", flexGrow: 1, height: 1 }} />
          <span style={{ color: "#A1A1AA", fontSize: 11, fontWeight: 500, letterSpacing: "0.04em", textTransform: "uppercase" }}>Or set env variables</span>
          <div style={{ backgroundColor: "#E4E4E7", flexGrow: 1, height: 1 }} />
        </div>
        <MultiLineCode lines={[
          "export COGNEE_MODE=cloud",
          "export COGNEE_BASE_URL=https://...",
          "export COGNEE_API_KEY=your-api-key",
        ]} />

        <Divider />
        <WaitingIndicator text={`Waiting for your ${framework} agent to connect...`} />
      </div>

      <NavButtons onBack={onBack} continueDisabled />
      <SkipLink onClick={onSkip} />
    </div>
  );
}

// ── Database Connection ──

export function DatabaseStep({ onBack, onSkip }: { onBack: () => void; onSkip: () => void }) {
  const [dbType, setDbType] = useState("PostgreSQL");
  const dbTypes = ["PostgreSQL", "MySQL", "SQLite", "MongoDB"];

  return (
    <div style={{ alignItems: "center", display: "flex", flexDirection: "column", flexGrow: 1, gap: 32, paddingBlock: 48, paddingInline: 80, fontFamily: '"Inter", system-ui, sans-serif' }}>
      <div style={{ alignItems: "center", display: "flex", flexDirection: "column", gap: 8 }}>
        <StepBadge step={1} />
        <h1 style={{ color: "#18181B", fontSize: 28, fontWeight: 600, lineHeight: "34px", margin: 0 }}>Connect a Database</h1>
        <p style={{ color: "#71717A", fontSize: 15, lineHeight: "22px", margin: 0, textAlign: "center", maxWidth: 480 }}>Ingest data directly from your database into Cognee.</p>
      </div>

      <div style={{ backgroundColor: "#FFFFFF", borderColor: "#E4E4E7", borderRadius: 12, borderStyle: "solid", borderWidth: 1, display: "flex", flexDirection: "column", gap: 24, paddingBlock: 28, paddingInline: 28, width: 560 }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          <span style={{ color: "#71717A", fontSize: 12, fontWeight: 500, letterSpacing: "0.04em", textTransform: "uppercase" }}>Database type</span>
          <select
            value={dbType}
            onChange={(e) => setDbType(e.target.value)}
            style={{ backgroundColor: "#FFFFFF", borderColor: "#E4E4E7", borderRadius: 8, borderStyle: "solid", borderWidth: 1, height: 40, paddingInline: 14, fontSize: 14, fontWeight: 500, color: "#18181B", outline: "none" }}
          >
            {dbTypes.map((d) => <option key={d} value={d}>{d}</option>)}
          </select>
        </div>

        <Divider />

        <div style={{ alignItems: "center", display: "flex", gap: 12 }}>
          <NumberCircle n={1} />
          <span style={{ color: "#18181B", fontSize: 14, fontWeight: 500 }}>Configure your connection</span>
        </div>
        <MultiLineCode lines={[
          `DB_PROVIDER=${dbType.toLowerCase()}`,
          `DB_HOST=localhost`,
          `DB_PORT=${dbType === "PostgreSQL" ? "5432" : dbType === "MySQL" ? "3306" : "27017"}`,
          `DB_USERNAME=cognee`,
          `DB_PASSWORD=your-password`,
          `DB_NAME=cognee_db`,
        ]} />

        <Divider />

        <div style={{ alignItems: "center", display: "flex", gap: 12 }}>
          <NumberCircle n={2} />
          <span style={{ color: "#18181B", fontSize: 14, fontWeight: 500 }}>Run the migration pipeline</span>
        </div>
        <CodeBlock>{`cognee migrate --source ${dbType.toLowerCase()}`}</CodeBlock>
        <span style={{ color: "#A1A1AA", fontSize: 12 }}>This will scan your database tables and ingest data into the knowledge graph.</span>

        <Divider />
        <WaitingIndicator text="Waiting for database connection..." />
      </div>

      <NavButtons onBack={onBack} continueDisabled />
      <SkipLink onClick={onSkip} />
    </div>
  );
}
