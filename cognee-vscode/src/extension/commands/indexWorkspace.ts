import { promises as fs } from "node:fs";
import * as path from "node:path";

import * as vscode from "vscode";

import { buildIgnoreFilter } from "../ignore";
import { basenameOf } from "../paths";
import type { Runtime } from "../runtime";

const INCLUDE_GLOB =
  "**/*.{ts,tsx,js,jsx,mjs,cjs,py,go,rs,java,kt,rb,php,c,h,cc,cpp,hpp,cs,swift,scala," +
  "md,mdx,txt,rst,json,yaml,yml,toml,ini,cfg,sh,bash,sql,graphql,html,css,scss,less,vue,svelte}";

const EXCLUDE_GLOB =
  "**/{node_modules,.git,dist,build,out,.next,.nuxt,.venv,venv,__pycache__,.mypy_cache," +
  ".pytest_cache,coverage,.idea,.vscode-test,target,bin,obj}/**";

const MAX_FILES = 400;

/**
 * Index the workspace into project memory: discover eligible files, show a
 * preflight summary, and (on confirmation) remember each one with progress and
 * cancellation. Opt-in by design — nothing is sent until the user confirms.
 */
export async function indexWorkspace(runtime: Runtime): Promise<void> {
  // Dependency/build/VCS directories are never useful project memory, so they
  // are always excluded — passing null here would disable VS Code's defaults too.
  const found = await vscode.workspace.findFiles(INCLUDE_GLOB, EXCLUDE_GLOB, MAX_FILES);

  // When enabled, honor the project's own ignore files so git-ignored (and
  // .cogneeignore-listed) files — build output, secrets, envs — aren't uploaded.
  const isIgnored = runtime.config.respectGitignore
    ? await loadIgnoreFilter(runtime.workspaceRoot)
    : undefined;

  const maxBytes = runtime.config.maxFileSizeKb * 1024;
  const eligible: { uri: vscode.Uri; size: number }[] = [];
  let totalBytes = 0;
  for (const uri of found) {
    if (isIgnored?.(vscode.workspace.asRelativePath(uri, false))) {
      continue;
    }
    try {
      const stat = await vscode.workspace.fs.stat(uri);
      if (stat.size > 0 && stat.size <= maxBytes) {
        eligible.push({ uri, size: stat.size });
        totalBytes += stat.size;
      }
    } catch {
      // Unreadable entry — skip it.
    }
  }

  if (eligible.length === 0) {
    void vscode.window.showInformationMessage(
      "Cognee: no eligible files to index (check size limit and ignore settings).",
    );
    return;
  }

  const capNote = found.length >= MAX_FILES ? ` (capped at ${MAX_FILES})` : "";
  const proceed = await vscode.window.showWarningMessage(
    `Index ${eligible.length} file(s), ${formatBytes(totalBytes)}${capNote} into "${runtime.datasetName}"? ` +
      "This sends file contents to your configured Cognee backend.",
    { modal: true },
    "Index workspace",
  );
  if (proceed !== "Index workspace") {
    return;
  }

  await runIndexing(runtime, eligible);
}

async function runIndexing(
  runtime: Runtime,
  files: { uri: vscode.Uri; size: number }[],
): Promise<void> {
  await vscode.window.withProgress(
    { location: vscode.ProgressLocation.Notification, title: "Cognee: indexing workspace…", cancellable: true },
    async (progress, token) => {
      const controller = new AbortController();
      token.onCancellationRequested(() => controller.abort());

      let indexed = 0;
      let failed = 0;
      const increment = 100 / files.length;

      for (const file of files) {
        if (token.isCancellationRequested) {
          break;
        }
        const label = vscode.workspace.asRelativePath(file.uri, false);
        progress.report({ message: label, increment });
        try {
          const bytes = await vscode.workspace.fs.readFile(file.uri);
          const text = Buffer.from(bytes).toString("utf8");
          if (!text.trim()) {
            continue;
          }
          await runtime.client.remember(`Source: ${label}\n\n${text}`, {
            datasetName: runtime.datasetName,
            filename: basenameOf(label),
            runInBackground: true,
            signal: controller.signal,
          });
          await runtime.pathIndex.record(label);
          indexed += 1;
        } catch (error) {
          failed += 1;
          runtime.logger.error(`index failed for ${label}`, error);
        }
      }

      runtime.logger.info(`index complete: ${indexed} indexed, ${failed} failed.`);
      if (failed > 0) {
        void vscode.window.showWarningMessage(
          `Cognee: indexed ${indexed} file(s); ${failed} failed. See the Cognee output for details.`,
        );
      } else {
        void vscode.window.showInformationMessage(
          `Cognee: queued ${indexed} file(s) into project memory (${runtime.datasetName}).`,
        );
      }
    },
  );
}

/** Read the workspace's `.gitignore` and `.cogneeignore` into a path matcher. */
async function loadIgnoreFilter(
  workspaceRoot: string,
): Promise<((relativePath: string) => boolean) | undefined> {
  const read = async (name: string): Promise<string | undefined> => {
    try {
      return await fs.readFile(path.join(workspaceRoot, name), "utf8");
    } catch {
      return undefined; // absent or unreadable — nothing to add
    }
  };
  const [gitignore, cogneeignore] = await Promise.all([read(".gitignore"), read(".cogneeignore")]);
  return buildIgnoreFilter(gitignore, cogneeignore);
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
