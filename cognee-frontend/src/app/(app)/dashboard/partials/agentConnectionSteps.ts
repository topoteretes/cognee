import {
  CLAUDE_MARKETPLACE_ADD,
  CLAUDE_PLUGIN_INSTALL,
  CODEX_HOOKS_ENABLE,
  CODEX_MARKETPLACE_ADD,
  CODEX_PLUGIN_INSTALL,
  OPENCLAW_SKILL_INSTALL,
  GENERIC_SKILL_INSTALL,
  UPLOAD_MEMORY_PROMPT,
  UPLOAD_SAMPLE_PROMPT,
  RECALL_SAMPLE_PROMPT,
} from "@/data/prompts";

export interface AciStepDef {
  title: string;
  description: string;
  code?: string;
  codeToCopy?: string;
  loading?: boolean;
  codeBlocks?: { code: string; toCopy?: string; label?: string }[];
  skillPath?: string;
  skillContent?: string;
}

export type AciAgentKey = "upload" | "claude-code" | "codex" | "openclaw" | "api-mcp";

export interface AciCardConfig {
  key: AciAgentKey;
  name: string;
  description: string;
}

export const CARDS_CFG: AciCardConfig[] = [
  { key: "claude-code", name: "Claude Code",   description: "Give Claude Code persistent memory across all your projects" },
  { key: "codex",       name: "Codex",         description: "Connect OpenAI Codex to your knowledge graph via the Cognee plugin" },
  { key: "openclaw",    name: "Openclaw",       description: "Connect Openclaw to your knowledge graph via AGENTS.md" },
  { key: "api-mcp",     name: "API / MCP",      description: "Connect any agent or app via the REST API or MCP" },
  { key: "upload",      name: "Company Brain",  description: "Upload PDFs, docs, and data to build your knowledge graph" },
];

interface StepOptions {
  baseUrl: string;
  resolvedKey: string;
  credsCode: string;
  isInitializing: boolean;
  connectVerified: boolean;
}

export function getSteps(key: AciAgentKey, opts: StepOptions): AciStepDef[] {
  const { baseUrl, resolvedKey, credsCode, isInitializing, connectVerified } = opts;

  const credStep: AciStepDef = {
    title: "Set your API credentials",
    description: "Open a terminal and run these commands to configure your Cognee endpoint and key.",
    code: `export COGNEE_BASE_URL="${baseUrl}"`,
    codeToCopy: credsCode,
    loading: isInitializing,
  };

  if (key === "claude-code") return [
    credStep,
    {
      title: "Install the Cognee plugin",
      description: "Run these in your terminal one at a time — register the Cognee marketplace, then install the memory plugin.",
      codeBlocks: [{ code: CLAUDE_MARKETPLACE_ADD }, { code: CLAUDE_PLUGIN_INSTALL }],
    },
    {
      title: "Upload something to Cognee",
      description: "Pick one and paste it into Claude — it stores the content in your Cognee memory so you can recall it in the next step.",
      codeBlocks: [
        { label: "Option A · Your existing memory", code: UPLOAD_MEMORY_PROMPT },
        { label: "Option B · Try it with a sample", code: UPLOAD_SAMPLE_PROMPT },
      ],
    },
    {
      title: connectVerified ? "Connected — session detected ✓" : "Recall it from Cognee",
      description: connectVerified
        ? "We detected your new session in Cognee Cloud — you're connected."
        : "First run /exit to close the session — that syncs it into Cognee Cloud — then reopen Claude Code and ask the question below. Answering from a fresh session proves it's recalling from your cloud memory.",
      codeBlocks: [{ code: "/exit" }, { code: RECALL_SAMPLE_PROMPT }],
    },
    {
      title: "You're all set",
      description: "The Cognee plugin hooks into Claude Code's lifecycle — no curl or manual API calls — and captures your session as you work. When a session ends (e.g. /exit), it consolidates that session into your Cognee Cloud knowledge graph, and every new session automatically recalls it back. Sessions are disposable; your memory isn't.",
    },
  ];

  if (key === "codex") return [
    credStep,
    {
      title: "Install the Cognee plugin",
      description: "Run these in your terminal one at a time — enable Codex hooks, register the Cognee marketplace, then install the memory plugin.",
      codeBlocks: [{ code: CODEX_HOOKS_ENABLE }, { code: CODEX_MARKETPLACE_ADD }, { code: CODEX_PLUGIN_INSTALL }],
    },
    {
      title: "Upload something to Cognee",
      description: "Pick one and paste it into Codex — it stores the content in your Cognee memory so you can recall it in the next step.",
      codeBlocks: [
        { label: "Option A · Your existing memory", code: UPLOAD_MEMORY_PROMPT },
        { label: "Option B · Try it with a sample", code: UPLOAD_SAMPLE_PROMPT },
      ],
    },
    {
      title: "Recall it from Cognee",
      description: "First run /exit to close the session — that syncs it into Cognee Cloud — then reopen Codex and ask the question below. Answering from a fresh session proves it's recalling from your cloud memory.",
      codeBlocks: [{ code: "/exit" }, { code: RECALL_SAMPLE_PROMPT }],
    },
    {
      title: "You're all set",
      description: "The Cognee plugin hooks into Codex's lifecycle — no curl or manual API calls — and captures your session as you work. When a session ends (e.g. /exit), it consolidates that session into your Cognee Cloud knowledge graph, and every new session automatically recalls it back. Sessions are disposable; your memory isn't.",
    },
  ];

  if (key === "openclaw") return [
    credStep,
    {
      title: "Install the Cognee skill",
      description: "Click below to copy the install command to your clipboard, then paste and run it in your local terminal. Nothing is sent to our servers — the skill file is written on your own machine.",
      skillPath: "~/.openclaw/skills/cognee/SKILL.md",
      skillContent: OPENCLAW_SKILL_INSTALL,
    },
    {
      title: "Test the connection",
      description: `Open Openclaw in your project and ask: "What do you know from cognee?" — if it responds with knowledge from your brain, you're connected.`,
    },
  ];

  if (key === "api-mcp") return [
    credStep,
    {
      title: "Query the REST API",
      description: "Send a recall query to your Cognee endpoint from any HTTP client or language.",
      code: `curl -X POST ${baseUrl}/api/v1/recall`,
      codeToCopy: `curl -X POST ${baseUrl}/api/v1/recall \\\n  -H "X-Api-Key: ${resolvedKey}" \\\n  -H "Content-Type: application/json" \\\n  -d '{"query": "What are the main entities?"}'`,
      loading: isInitializing,
    },
    {
      title: "Or install the Cognee skill",
      description: "Prefer skills? Run this command from your project root to create the skill file, then point your agent at it. The skill teaches your agent to call the Cognee API using the credentials from step 1.",
      code: "skills/cognee/SKILL.md",
      codeToCopy: GENERIC_SKILL_INSTALL,
    },
    {
      title: "Test the connection",
      description: `Ask your agent: "What do you know from cognee?" — Cognee's memory should respond with knowledge from your brain.`,
    },
  ];

  return [];
}
