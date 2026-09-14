"use client";

import { useState, type ReactNode } from "react";
import { CLAUDE_DESKTOP_MCP_ENTRY, fillTemplate } from "@/data/prompts";

const MONO_FONT = 'ui-monospace, Menlo, Monaco, "Cascadia Mono", "Segoe UI Mono", "Roboto Mono", monospace';

export function imgIcon(src: string, alt: string) {
  return <img src={src} alt={alt} style={{ width: 24, height: 24, objectFit: "contain" }} />;
}

export function credStep(baseUrl: string, apiKey: string, loading: boolean) {
  return {
    title: "Set your API credentials",
    description: "Open a terminal and run these commands to configure your Cognee endpoint and key.",
    code: `export COGNEE_BASE_URL="${baseUrl}"`,
    codeToCopy: `export COGNEE_BASE_URL="${baseUrl}"\nexport COGNEE_API_KEY="${apiKey}"`,
    loading,
  };
}
export function ApiIcon() {
  return <svg width="24" height="24" viewBox="0 0 24 24" fill="none"><rect x="3" y="6" width="18" height="12" rx="2" stroke="rgba(237,236,234,0.7)" strokeWidth="1.5"/><path d="M7 9L4 12L7 15" stroke="#BC9BFF" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/><path d="M17 9L20 12L17 15" stroke="#BC9BFF" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/><line x1="13" y1="8" x2="11" y2="16" stroke="rgba(237,236,234,0.5)" strokeWidth="1.5" strokeLinecap="round"/></svg>;
}

/** The "Install uv (provides uvx)" step shared by every MCP-based connector. */
export function installUvStep() {
  return {
    title: "Install uv (provides uvx)",
    description: (
      <>
        <strong style={{ color: "#EDECEA" }}>Open your terminal</strong> and{" "}
        <strong style={{ color: "#EDECEA" }}>install uv</strong> using one of the below methods:
      </>
    ),
    codeBlocks: [
      { label: "Homebrew", code: "brew install uv" },
      { label: "Install script", code: "curl -LsSf https://astral.sh/uv/install.sh | sh" },
    ],
  };
}

// A subtle callout that visually separates a "gotcha"/troubleshooting note from
// the main step instruction.
export function InfoBox({ children }: { children: ReactNode }) {
  return (
    <div style={{ display: "flex", gap: 9, background: "rgba(188,155,255,0.08)", border: "1px solid rgba(188,155,255,0.22)", borderRadius: 8, padding: "10px 12px" }}>
      <svg width="15" height="15" viewBox="0 0 16 16" fill="none" style={{ flexShrink: 0, marginTop: 1 }}>
        <circle cx="8" cy="8" r="6.5" stroke="#BC9BFF" strokeWidth="1.3" />
        <path d="M8 7.2V11" stroke="#BC9BFF" strokeWidth="1.4" strokeLinecap="round" />
        <circle cx="8" cy="4.9" r="0.9" fill="#BC9BFF" />
      </svg>
      <div style={{ fontSize: 12.5, color: "rgba(237,236,234,0.6)", lineHeight: 1.6 }}>{children}</div>
    </div>
  );
}

function CopyButton({ copied, onCopy, ariaLabel }: { copied: boolean; onCopy: () => void; ariaLabel: string }) {
  return (
    <button onClick={(e) => { e.stopPropagation(); onCopy(); }} aria-label={ariaLabel} style={{ position: "absolute", top: 8, right: 10, background: "rgba(24,24,27,0.85)", border: "none", cursor: "pointer", padding: 2, borderRadius: 4, zIndex: 2 }}>
      {copied
        ? <svg width="14" height="14" viewBox="0 0 16 16" fill="none"><path d="M3.5 8.5L6.5 11.5L12.5 4.5" stroke="#22C55E" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" /></svg>
        : <svg width="14" height="14" viewBox="0 0 16 16" fill="none"><rect x="5" y="5" width="8" height="8" rx="1.5" stroke="#6B7280" strokeWidth="1.5" /><path d="M11 3H4.5A1.5 1.5 0 003 4.5V11" stroke="#6B7280" strokeWidth="1.5" strokeLinecap="round" /></svg>}
    </button>
  );
}

/**
 * A read-only, annotated preview of claude_desktop_config.json: the user's
 * existing keys are dimmed and the inserted Cognee block is highlighted, so
 * they can eyeball their open file against it. Copy grabs just the
 * merge-ready "mcpServers" fragment (no outer braces) with a trailing comma,
 * so it stays valid JSON when inserted above the user's existing keys —
 * matching the "}," the preview shows.
 */
export function ConfigPreview({ baseUrl, apiKey, loading }: { baseUrl: string; apiKey: string; loading: boolean }) {
  const [copied, setCopied] = useState(false);
  function doCopy() {
    if (loading) return;
    navigator.clipboard.writeText(fillTemplate(CLAUDE_DESKTOP_MCP_ENTRY, baseUrl, apiKey) + ",");
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  }
  const inserted = [
    '  "mcpServers": {',
    '    "cognee": {',
    '      "command": "uvx",',
    '      "args": [',
    '        "cognee-mcp@latest",',
    `        "--api-url", "${loading ? "…" : baseUrl}",`,
    `        "--api-token", "${loading ? "…" : apiKey}"`,
    "      ]",
    "    }",
    "  },",
  ];
  const dim = "rgba(237,236,234,0.4)";
  return (
    <div style={{ position: "relative", background: "#18181B", borderRadius: 8, marginBottom: 4 }}>
      <CopyButton copied={copied} onCopy={doCopy} ariaLabel={copied ? "Copied" : "Copy the highlighted block"} />
      <div style={{ overflowX: "auto", padding: "12px 0", fontFamily: MONO_FONT, fontSize: 12.5, lineHeight: 1.75, whiteSpace: "pre" }}>
        <div style={{ padding: "0 14px", color: dim }}>{"{"}</div>
        {inserted.map((line, i) => (
          <div key={i} style={{ padding: "0 14px", color: loading ? "rgba(237,236,234,0.5)" : "#EDECEA", background: "rgba(34,197,94,0.13)", borderLeft: "2px solid #22C55E" }}>{line}</div>
        ))}
        <div style={{ padding: "0 14px", color: dim }}>{'  "preferences": { … }'}</div>
        <div style={{ padding: "0 14px", color: dim }}>{"}"}</div>
      </div>
    </div>
  );
}

/**
 * Cursor's ~/.cursor/mcp.json opens with the "mcpServers" wrapper already
 * scaffolded by "Add Custom MCP", so we preview the whole file for context
 * but highlight only the "cognee" entry. The copy button grabs exactly that
 * highlighted entry (not the dimmed wrapper), so it drops straight into the
 * existing mcpServers object.
 */
export function CursorConfigPreview({ baseUrl, apiKey, loading }: { baseUrl: string; apiKey: string; loading: boolean }) {
  const [copied, setCopied] = useState(false);
  const cogneeBlock = [
    '    "cognee": {',
    '      "command": "uvx",',
    '      "args": ["cognee-mcp"],',
    '      "env": {',
    `        "COGNEE_BASE_URL": "${loading ? "…" : baseUrl}",`,
    `        "COGNEE_API_KEY": "${loading ? "…" : apiKey}"`,
    "      }",
    "    }",
  ];
  function doCopy() {
    if (loading) return;
    navigator.clipboard.writeText(cogneeBlock.join("\n"));
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  }
  const dim = "rgba(237,236,234,0.4)";
  return (
    <div style={{ position: "relative", background: "#18181B", borderRadius: 8, marginBottom: 4 }}>
      <CopyButton copied={copied} onCopy={doCopy} ariaLabel={copied ? "Copied" : "Copy the cognee entry"} />
      <div style={{ overflowX: "auto", padding: "12px 0", fontFamily: MONO_FONT, fontSize: 12.5, lineHeight: 1.75, whiteSpace: "pre" }}>
        <div style={{ padding: "0 14px", color: dim }}>{"{"}</div>
        <div style={{ padding: "0 14px", color: dim }}>{'  "mcpServers": {'}</div>
        {cogneeBlock.map((line, i) => (
          <div key={i} style={{ padding: "0 14px", color: loading ? "rgba(237,236,234,0.5)" : "#EDECEA", background: "rgba(34,197,94,0.13)", borderLeft: "2px solid #22C55E" }}>{line}</div>
        ))}
        <div style={{ padding: "0 14px", color: dim }}>{"  }"}</div>
        <div style={{ padding: "0 14px", color: dim }}>{"}"}</div>
      </div>
    </div>
  );
}

/**
 * Gemini CLI has no UI to add an MCP server — you edit ~/.gemini/settings.json
 * by hand — so we show the FULL file the user should end up with (nothing
 * hidden in a command) and highlight the "cognee" entry. Copy grabs the whole
 * block for the common empty-file case; the step text covers merging into an
 * existing mcpServers. Credentials use --api-url/--api-token (API mode) so the
 * reader can see exactly which value is the tenant URL and which is the key.
 */
export function GeminiConfigPreview({ baseUrl, apiKey, loading }: { baseUrl: string; apiKey: string; loading: boolean }) {
  const [copied, setCopied] = useState(false);
  const fullBlock = [
    "{",
    '  "mcpServers": {',
    '    "cognee": {',
    '      "command": "uvx",',
    '      "args": [',
    '        "cognee-mcp@latest",',
    `        "--api-url", "${loading ? "…" : baseUrl}",`,
    `        "--api-token", "${loading ? "…" : apiKey}"`,
    "      ]",
    "    }",
    "  }",
    "}",
  ];
  // The "cognee" entry spans lines 2–9 (its opening brace through its closing
  // brace) — highlighted so users merging into an existing file can see the
  // exact lines to lift out.
  const isEntry = (i: number): boolean => i >= 2 && i <= 9;
  function doCopy() {
    if (loading) return;
    navigator.clipboard.writeText(fullBlock.join("\n"));
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  }
  return (
    <div style={{ position: "relative", background: "#18181B", borderRadius: 8, marginBottom: 4 }}>
      <CopyButton copied={copied} onCopy={doCopy} ariaLabel={copied ? "Copied" : "Copy the full config"} />
      <div style={{ overflowX: "auto", padding: "12px 0", fontFamily: MONO_FONT, fontSize: 12.5, lineHeight: 1.75, whiteSpace: "pre" }}>
        {fullBlock.map((line, i) => (
          <div key={i} style={{ padding: "0 14px", color: loading ? "rgba(237,236,234,0.5)" : "#EDECEA", background: isEntry(i) ? "rgba(34,197,94,0.13)" : "transparent", borderLeft: isEntry(i) ? "2px solid #22C55E" : "2px solid transparent" }}>{line}</div>
        ))}
      </div>
    </div>
  );
}
