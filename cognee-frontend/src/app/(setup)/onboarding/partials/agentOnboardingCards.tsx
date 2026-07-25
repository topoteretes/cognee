"use client";

import { useState } from "react";
import { CLAUDE_MARKETPLACE_ADD, CLAUDE_PLUGIN_INSTALL, CODEX_HOOKS_ENABLE, CODEX_MARKETPLACE_ADD, CODEX_PLUGIN_INSTALL, UPLOAD_MEMORY_PROMPT, UPLOAD_SAMPLE_PROMPT, RECALL_SAMPLE_PROMPT } from "@/data/prompts";
import { useOnboardingTrackEvent } from "../useOnboardingTrackEvent";

export interface AgentOnboardingCard {
  title: string;
  description: string;
  node?: React.ReactNode;
}

// Identifies which snippet was copied — `onboarding_creds_copied` fires for every
// copy button on the page (creds, install commands, prompts, /exit), so this is
// the only way to tell a real credentials copy from the rest downstream.
export type OnboardingCopyTarget =
  | "api_credentials"
  | "marketplace_add"
  | "plugin_install"
  | "hooks_enable"
  | "upload_memory_prompt"
  | "upload_sample_prompt"
  | "exit_command"
  | "recall_sample_prompt";

// Single-line code block: shows ONE line, truncates the rest with an ellipsis (…)
// so long commands never wrap or overflow on small screens. `code` is what's
// shown; `toCopy` (when set) is the full multi-line command that's copied.
export function OnboardingInlineCode({ code, toCopy, loading, placeholder = "Preparing…", agent, copyTarget }: {
  code: string; toCopy?: string; loading?: boolean; placeholder?: string; agent?: "claude-code" | "codex"; copyTarget: OnboardingCopyTarget;
}) {
  const [copied, setCopied] = useState(false);
  const track = useOnboardingTrackEvent();
  const copy = () => {
    if (loading) return;
    navigator.clipboard.writeText(toCopy ?? code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
    track({ pageName: "Onboarding", eventName: "onboarding_creds_copied", additionalProperties: { copy_target: copyTarget, ...(agent ? { agent } : {}) } });
  };
  return (
    <div
      onClick={copy}
      className="cursor-pointer"
      style={{ background: "#18181B", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 8, padding: "11px 14px", display: "flex", alignItems: "center", justifyContent: "space-between", gap: 10, cursor: loading ? "wait" : "pointer", width: "100%" }}
    >
      <pre style={{ margin: 0, fontSize: 12.5, fontFamily: 'ui-monospace, Menlo, Monaco, "Cascadia Mono", "Segoe UI Mono", "Roboto Mono", monospace', color: loading ? "#585B70" : "rgba(237,236,234,0.85)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis", flex: 1, minWidth: 0 }}>
        <code>{loading ? placeholder : code}</code>
      </pre>
      <button
        onClick={(e) => { e.stopPropagation(); copy(); }}
        className="cursor-pointer"
        style={{ background: "#27272A", border: "1px solid #3F3F46", borderRadius: 4, padding: "4px 8px", fontSize: 11, color: loading ? "rgba(237,236,234,0.35)" : "rgba(237,236,234,0.65)", flexShrink: 0 }}
      >
        {copied ? "Copied!" : "Copy"}
      </button>
    </div>
  );
}

// Live connection indicator for the "connect & recall" step: a pulsing dim dot
// while we wait for the agent's first session, flipping to a solid green dot +
// "Connected" once a new session is detected in Cognee Cloud.
export function ConnectStatus({ verified }: { verified: boolean }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12.5 }}>
      <span
        className={verified ? undefined : "ob-pulse"}
        style={{
          width: 8, height: 8, borderRadius: "50%", flexShrink: 0,
          background: verified ? "#22C55E" : "rgba(237,236,234,0.4)",
          boxShadow: verified ? "0 0 0 3px rgba(34,197,94,0.18)" : "none",
        }}
      />
      <span style={{ color: verified ? "#22C55E" : "rgba(237,236,234,0.5)" }}>
        {verified ? "Connected — activity detected in Cognee Cloud" : "Waiting for a connection…"}
      </span>
    </div>
  );
}

export function buildAgentOnboardingCards(params: {
  agent: "claude-code" | "codex";
  name: string;
  baseUrl: string;
  credsCode: string;
  credsReady: boolean;
  connectVerified: boolean;
  goToDashboard: () => void;
}): AgentOnboardingCard[] {
  const { agent, name, baseUrl, credsCode, credsReady, connectVerified, goToDashboard } = params;

  const credsCard: AgentOnboardingCard = {
    title: "Copy your API credentials",
    description: "Open a terminal and run these to point your agent at your Cognee memory.",
    node: <OnboardingInlineCode code={`export COGNEE_BASE_URL="${baseUrl}"`} toCopy={credsCode} loading={!credsReady} placeholder="Preparing your credentials…" agent={agent} copyTarget="api_credentials" />,
  };
  const allSetCard: AgentOnboardingCard = {
    title: "You're all set",
    description: "The loop you just saw — upload, exit, reopen, and your agent still remembers — is the whole point. Here's why it works:",
    node: (
      <div style={{ display: "flex", flexDirection: "column", gap: 12, width: "100%" }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 8, padding: 14, borderRadius: 10, background: "rgba(237,236,234,0.04)", border: "1px solid rgba(237,236,234,0.10)" }}>
          <div style={{ fontSize: 13, fontWeight: 600, color: "#EDECEA" }}>What just happened</div>
          <ul style={{ margin: 0, paddingLeft: 18, fontSize: 13, lineHeight: "19px", color: "rgba(237,236,234,0.6)" }}>
            <li>The Cognee plugin hooks into your agent&apos;s lifecycle — no curl or manual API calls — and captures your session as you work.</li>
            <li>When a session ends (e.g. you <strong style={{ color: "#EDECEA" }}>exit</strong>), it consolidates that session into your Cognee Cloud knowledge graph.</li>
            <li>On every new session, it automatically recalls your memory back from the cloud.</li>
          </ul>
          <div style={{ fontSize: 13, lineHeight: "19px", color: "rgba(237,236,234,0.6)" }}>
            That&apos;s why, after running <code>/exit</code> and reopening, your agent still knew what you uploaded — sessions are disposable; your memory isn&apos;t.
          </div>
        </div>
        <button onClick={goToDashboard} className="cursor-pointer" style={{ background: "#BC9BFF", border: "none", borderRadius: 8, padding: "11px 32px", fontSize: 14, fontWeight: 500, color: "#1e1e1c", letterSpacing: "-0.01em" }}>
          Go to Dashboard →
        </button>
      </div>
    ),
  };

  return agent === "claude-code"
    ? [
        credsCard,
        {
          title: "Install the Cognee plugin",
          description: "Run these in your terminal one at a time — register the Cognee marketplace, then install the memory plugin.",
          node: (
            <div style={{ display: "flex", flexDirection: "column", gap: 8, width: "100%" }}>
              <OnboardingInlineCode code={CLAUDE_MARKETPLACE_ADD} agent={agent} copyTarget="marketplace_add" />
              <OnboardingInlineCode code={CLAUDE_PLUGIN_INSTALL} agent={agent} copyTarget="plugin_install" />
            </div>
          ),
        },
        {
          title: "Upload something to Cognee",
          description: "Pick one and paste it into Claude — it stores the content in your Cognee memory so you can recall it in the next step.",
          node: (
            <div style={{ display: "flex", flexDirection: "column", gap: 14, width: "100%" }}>
              <div>
                <div style={{ fontSize: 13, fontWeight: 600, color: "#EDECEA", marginBottom: 6 }}>Option A · Your existing memory</div>
                <OnboardingInlineCode code={UPLOAD_MEMORY_PROMPT} agent={agent} copyTarget="upload_memory_prompt" />
              </div>
              <div>
                <div style={{ fontSize: 13, fontWeight: 600, color: "#EDECEA", marginBottom: 6 }}>Option B · Try it with a sample</div>
                <OnboardingInlineCode code={UPLOAD_SAMPLE_PROMPT} agent={agent} copyTarget="upload_sample_prompt" />
              </div>
              <ConnectStatus verified={connectVerified} />
            </div>
          ),
        },
        {
          title: connectVerified ? "Connected — activity detected" : "Recall it from Cognee",
          description: connectVerified
            ? "We detected your new session in Cognee Cloud — you're connected. You're all set."
            : "Now ask Claude a question about what you just uploaded — it should answer from Cognee Cloud. (For the sample, use the question below.) This step completes on its own once your session shows up.",
          node: (
            <div style={{ display: "flex", flexDirection: "column", gap: 10, width: "100%" }}>
              <div>
                <div style={{ fontSize: 13, fontWeight: 600, color: "#EDECEA", marginBottom: 6 }}>First, run this to start a fresh session</div>
                <OnboardingInlineCode code="/exit" agent={agent} copyTarget="exit_command" />
              </div>
              <div>
                <div style={{ fontSize: 13, fontWeight: 600, color: "#EDECEA", marginBottom: 6 }}>Then ask</div>
                <OnboardingInlineCode code={RECALL_SAMPLE_PROMPT} agent={agent} copyTarget="recall_sample_prompt" />
              </div>
              <ConnectStatus verified={connectVerified} />
            </div>
          ),
        },
        allSetCard,
      ]
    : [
        credsCard,
        {
          title: "Install the Cognee plugin",
          description: "Run these in your terminal one at a time — enable Codex hooks, register the Cognee marketplace, then install the memory plugin.",
          node: (
            <div style={{ display: "flex", flexDirection: "column", gap: 8, width: "100%" }}>
              <OnboardingInlineCode code={CODEX_HOOKS_ENABLE} agent={agent} copyTarget="hooks_enable" />
              <OnboardingInlineCode code={CODEX_MARKETPLACE_ADD} agent={agent} copyTarget="marketplace_add" />
              <OnboardingInlineCode code={CODEX_PLUGIN_INSTALL} agent={agent} copyTarget="plugin_install" />
            </div>
          ),
        },
        {
          title: "Upload something to Cognee",
          description: `Pick one and paste it into ${name} — it stores the content in your Cognee memory so you can recall it in the next step.`,
          node: (
            <div style={{ display: "flex", flexDirection: "column", gap: 14, width: "100%" }}>
              <div>
                <div style={{ fontSize: 13, fontWeight: 600, color: "#EDECEA", marginBottom: 6 }}>Option A · Your existing memory</div>
                <OnboardingInlineCode code={UPLOAD_MEMORY_PROMPT} agent={agent} copyTarget="upload_memory_prompt" />
              </div>
              <div>
                <div style={{ fontSize: 13, fontWeight: 600, color: "#EDECEA", marginBottom: 6 }}>Option B · Try it with a sample</div>
                <OnboardingInlineCode code={UPLOAD_SAMPLE_PROMPT} agent={agent} copyTarget="upload_sample_prompt" />
              </div>
              <ConnectStatus verified={connectVerified} />
            </div>
          ),
        },
        {
          title: connectVerified ? "Connected — activity detected" : "Recall it from Cognee",
          description: connectVerified
            ? "We detected your new session in Cognee Cloud — you're connected. You're all set."
            : `Now ask ${name} a question about what you just uploaded — it should answer from Cognee Cloud. (For the sample, use the question below.) This step completes on its own once your session shows up.`,
          node: (
            <div style={{ display: "flex", flexDirection: "column", gap: 10, width: "100%" }}>
              <div>
                <div style={{ fontSize: 13, fontWeight: 600, color: "#EDECEA", marginBottom: 6 }}>First, run this to start a fresh session</div>
                <OnboardingInlineCode code="/exit" agent={agent} copyTarget="exit_command" />
              </div>
              <div>
                <div style={{ fontSize: 13, fontWeight: 600, color: "#EDECEA", marginBottom: 6 }}>Then ask</div>
                <OnboardingInlineCode code={RECALL_SAMPLE_PROMPT} agent={agent} copyTarget="recall_sample_prompt" />
              </div>
              <ConnectStatus verified={connectVerified} />
            </div>
          ),
        },
        allSetCard,
      ];
}
