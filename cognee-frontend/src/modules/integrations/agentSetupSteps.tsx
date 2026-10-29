import {
  CLAUDE_MARKETPLACE_ADD,
  CLAUDE_PLUGIN_INSTALL,
  CODEX_HOOKS_ENABLE,
  CODEX_MARKETPLACE_ADD,
  CODEX_PLUGIN_INSTALL,
} from "@/data/prompts";
import type { ReactNode } from "react";
import { StatusLineScreenshot } from "@/ui/elements/AgentSetupBlocks";
import { writeCogneeEnvFile, homePath, COGNEE_ENV_FILE_POSIX } from "@/utils/osCommands";
import type { PreferredOs } from "@/ui/layout/OsPreferenceContext";

// The single source of truth for "how do I connect a plugin agent".
//
// These instructions used to exist three times over — in onboarding, in the
// dashboard's connect strip, and in the integrations wizard — and had already
// drifted: the integrations copy was still telling people to `export` variables
// that die with the terminal, and had no notion of Windows at all. Every surface
// now builds from here.
export type PluginAgent = "claude-code" | "codex";

export function agentName(agent: PluginAgent): string {
  return agent === "claude-code" ? "Claude Code" : "Codex";
}

/** The command that launches the agent — the last thing the user runs. */
export function startCommand(agent: PluginAgent): string {
  return agent === "claude-code" ? "claude" : "codex";
}

/**
 * Marketplace registration + plugin install, run from the shell rather than
 * with the in-app `/plugin` commands: installing before launch means the first
 * session bootstraps memory on its own, with no restart.
 */
export function pluginInstallCommands(agent: PluginAgent): string[] {
  return agent === "claude-code"
    ? [CLAUDE_MARKETPLACE_ADD, CLAUDE_PLUGIN_INSTALL]
    : [CODEX_HOOKS_ENABLE, CODEX_MARKETPLACE_ADD, CODEX_PLUGIN_INSTALL];
}

/** Writes ~/.cognee/.env, which both plugins read at session start (CLO-532). */
export function credentialsCommand(os: PreferredOs, baseUrl: string, apiKey: string): string {
  return writeCogneeEnvFile(os, { COGNEE_BASE_URL: baseUrl, COGNEE_API_KEY: apiKey });
}

/** Human-readable path of that file, for step copy. */
export function credentialsPath(os: PreferredOs): string {
  return homePath(os, COGNEE_ENV_FILE_POSIX);
}

/** What the shell is called on this OS — the wizards say where to run things. */
export function terminalName(os: PreferredOs): string {
  return os === "windows" ? "PowerShell" : "Terminal";
}

export interface AgentSetupStep {
  title: string;
  description: string;
  /** Rendered as the numbered terminal block with a single "copy the lot" button. */
  lines?: string[];
  copyLabel?: string;
  loading?: boolean;
  /** Extra visual below the block (the status-line callout). */
  content?: ReactNode;
}

/** The header line every surface shows above the steps. */
export function setupIntro(agent: PluginAgent, os: PreferredOs): string {
  return `Run this setup in your ${terminalName(os)} — not inside ${agentName(agent)}.`;
}

/**
 * The exact three actions the onboarding flow uses — same titles, same copy,
 * same blocks. Open the terminal, paste one combined block, start the agent.
 *
 * Credentials and the plugin install are deliberately ONE block: split blocks
 * made it ambiguous whether they belonged in the terminal or inside the agent,
 * which was the single biggest source of confusion in the old five-step
 * version. That version also walked the user through storing and recalling a
 * sample before declaring success — a guided first run, but noise in a panel
 * someone opens to re-copy one command.
 */
export function agentSetupSteps(agent: PluginAgent, opts: {
  os: PreferredOs;
  baseUrl: string;
  apiKey: string;
  loading?: boolean;
}): AgentSetupStep[] {
  const { os, baseUrl, apiKey, loading = false } = opts;
  const name = agentName(agent);

  return [
    {
      title: `Open ${terminalName(os)}`,
      description: "",
    },
    {
      title: "Copy and run",
      description: "Saves your credentials and installs the Cognee plugin.",
      lines: [credentialsCommand(os, baseUrl, apiKey), ...pluginInstallCommands(agent)],
      copyLabel: "Copy all",
      loading,
    },
    {
      title: `Start ${name}`,
      description: agent === "claude-code"
        ? "You should now see Cognee in your status bar."
        : "Memory connects automatically from here.",
      lines: [startCommand(agent)],
      // Claude Code only: the screenshot is a Claude Code window, and Codex's
      // status line is not currently active.
      content: agent === "claude-code" ? <StatusLineScreenshot /> : undefined,
    },
  ];
}
