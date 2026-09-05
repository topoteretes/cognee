import type { ReactNode } from "react";
import { openclawSkillInstall, genericSkillInstall } from "@/data/prompts";
import { agentSetupSteps, terminalName } from "@/modules/integrations/agentSetupSteps";
import { curlBin, exportEnvVar, homePath } from "@/utils/osCommands";
import type { PreferredOs } from "@/ui/layout/OsPreferenceContext";

export interface AciStepDef {
  title: string;
  description: string;
  code?: string;
  codeToCopy?: string;
  loading?: boolean;
  codeBlocks?: { code: string; toCopy?: string; label?: string }[];
  skillPath?: string;
  skillContent?: string;
  /** Numbered terminal block with one "copy the lot" button — the shape the
   *  onboarding flow uses. */
  lines?: string[];
  copyLabel?: string;
  /** Extra visual below the block (the status-line callout). */
  content?: ReactNode;
}

export type AciAgentKey = "upload" | "claude-code" | "codex" | "openclaw" | "api-mcp";

export interface AciCardConfig {
  key: AciAgentKey;
  name: string;
  /** One-line card description — kept short so it never wraps in the card grid. */
  description: string;
  /** Fixed card width (px). Content-based, not equal-stretch: agents ~248,
   *  API/MCP a touch narrower, Company Dataset a touch wider. */
  width: number;
}

export const CARDS_CFG: AciCardConfig[] = [
  { key: "claude-code", name: "Claude Code",   description: "Memory for every project",     width: 248 },
  { key: "codex",       name: "Codex",         description: "Wire Codex to your graph",     width: 248 },
  { key: "openclaw",    name: "Openclaw",       description: "Connect via AGENTS.md",         width: 248 },
  { key: "api-mcp",     name: "API / MCP",      description: "Via REST API or MCP",         width: 228 },
  { key: "upload",      name: "Company Dataset",  description: "Upload docs to build memory",   width: 268 },
];

interface StepOptions {
  baseUrl: string;
  resolvedKey: string;
  isInitializing: boolean;
  os: PreferredOs;
}

export function getSteps(key: AciAgentKey, opts: StepOptions): AciStepDef[] {
  const { baseUrl, resolvedKey, isInitializing, os } = opts;

  // Shell exports, not the ~/.cognee/.env file — only the Claude Code and Codex
  // plugins read that file, and those two return agentSetupSteps() below. The
  // cards that use this step (Openclaw, API / MCP) go on to run a skill that
  // curls with "X-Api-Key: $COGNEE_API_KEY", so they need the variables in the
  // shell's own environment (CLO-532).
  const credStep: AciStepDef = {
    title: "Set your API credentials",
    description: `Run these in the ${terminalName(os)} you will start your agent from — they apply to that session. Add them to your shell profile to keep them across terminals.`,
    code: exportEnvVar(os, "COGNEE_BASE_URL", baseUrl),
    codeToCopy: `${exportEnvVar(os, "COGNEE_BASE_URL", baseUrl)}\n${exportEnvVar(os, "COGNEE_API_KEY", resolvedKey)}`,
    loading: isInitializing,
  };

  // Claude Code and Codex share the three-step shape onboarding settled on.
  if (key === "claude-code" || key === "codex") {
    return agentSetupSteps(key, { os, baseUrl, apiKey: resolvedKey, loading: isInitializing });
  }

  if (key === "openclaw") return [
    credStep,
    {
      title: "Install the Cognee skill",
      description: "Click below to copy the install command to your clipboard, then paste and run it in your local terminal. Nothing is sent to our servers — the skill file is written on your own machine.",
      skillPath: homePath(os, "/.openclaw/skills/cognee/SKILL.md"),
      skillContent: openclawSkillInstall(os),
    },
    {
      title: "Test the connection",
      description: `Open Openclaw in your project and ask: "What do you know from cognee?" — if it responds with knowledge from your dataset, you're connected.`,
    },
  ];

  if (key === "api-mcp") return [
    credStep,
    {
      title: "Query the REST API",
      description: "Send a recall query to your Cognee endpoint from any HTTP client or language.",
      code: `${curlBin(os)} -X POST ${baseUrl}/api/v1/recall`,
      codeToCopy: os === "windows"
        ? `${curlBin(os)} -X POST ${baseUrl}/api/v1/recall -H "X-Api-Key: ${resolvedKey}" -H "Content-Type: application/json" -d '{"query": "What are the main entities?"}'`
        : `curl -X POST ${baseUrl}/api/v1/recall \\\n  -H "X-Api-Key: ${resolvedKey}" \\\n  -H "Content-Type: application/json" \\\n  -d '{"query": "What are the main entities?"}'`,
      loading: isInitializing,
    },
    {
      title: "Or install the Cognee skill",
      description: "Prefer skills? Run this command from your project root to create the skill file, then point your agent at it. The skill teaches your agent to call the Cognee API using the credentials from step 1.",
      code: "skills/cognee/SKILL.md",
      codeToCopy: genericSkillInstall(os),
    },
    {
      title: "Test the connection",
      description: `Ask your agent: "What do you know from cognee?" — Cognee's memory should respond with knowledge from your dataset.`,
    },
  ];

  return [];
}
