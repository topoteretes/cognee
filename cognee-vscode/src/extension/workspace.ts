import { promises as fs } from "node:fs";
import * as path from "node:path";

import * as vscode from "vscode";

import { parseOriginUrl } from "../core";

/** Absolute path of the first workspace folder, or undefined when none is open. */
export function getWorkspaceRoot(): string | undefined {
  const folders = vscode.workspace.workspaceFolders;
  if (!folders || folders.length === 0) {
    return undefined;
  }
  return folders[0].uri.fsPath;
}

/**
 * Best-effort resolution of the `origin` remote URL by reading the repository's
 * git config directly. Dependency-free and good enough for stable dataset
 * scoping; returns null for non-git folders (the caller falls back to the path).
 */
export async function getGitRemote(workspaceRoot: string): Promise<string | null> {
  try {
    const configPath = await resolveGitConfigPath(workspaceRoot);
    if (!configPath) {
      return null;
    }
    const content = await fs.readFile(configPath, "utf8");
    return parseOriginUrl(content);
  } catch {
    return null;
  }
}

async function resolveGitConfigPath(workspaceRoot: string): Promise<string | null> {
  const dotGit = path.join(workspaceRoot, ".git");
  try {
    const stat = await fs.stat(dotGit);
    if (stat.isDirectory()) {
      return path.join(dotGit, "config");
    }
    if (stat.isFile()) {
      // Worktrees and submodules use a ".git" file: "gitdir: <path>".
      const pointer = await fs.readFile(dotGit, "utf8");
      const match = /gitdir:\s*(.+)/.exec(pointer);
      if (!match) {
        return null;
      }
      let gitDir = match[1].trim();
      if (!path.isAbsolute(gitDir)) {
        gitDir = path.resolve(workspaceRoot, gitDir);
      }
      return path.join(gitDir, "config");
    }
  } catch {
    return null;
  }
  return null;
}
