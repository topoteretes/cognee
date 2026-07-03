import * as vscode from "vscode";

import {
  deriveDatasetName,
  HttpCogneeClient,
  validateConfig,
  type CogneeClient,
  type CogneeConfig,
} from "../core";
import type { Logger } from "./logger";
import { readConfig } from "./settings";
import { getGitRemote, getWorkspaceRoot } from "./workspace";

/** Secret-storage key for the Cognee API key (preferred over synced settings). */
export const SECRET_API_KEY = "cognee.apiKey";

/** Everything a command needs to talk to Cognee for the active workspace. */
export interface Runtime {
  config: CogneeConfig;
  client: CogneeClient;
  datasetName: string;
  workspaceRoot: string;
  logger: Logger;
}

/**
 * Build a {@link Runtime} for the active workspace, or surface a clear message
 * and return undefined when preconditions aren't met (no folder, invalid config).
 * Resolved fresh per command so settings changes take effect immediately.
 */
export async function resolveRuntime(
  context: vscode.ExtensionContext,
  logger: Logger,
): Promise<Runtime | undefined> {
  const workspaceRoot = getWorkspaceRoot();
  if (!workspaceRoot) {
    void vscode.window.showWarningMessage(
      "Cognee: open a folder or workspace to use project memory.",
    );
    return undefined;
  }

  const config = readConfig();
  const validation = validateConfig(config);
  if (!validation.ok) {
    void vscode.window.showErrorMessage(
      `Cognee configuration problem: ${validation.errors.join(" ")}`,
    );
    return undefined;
  }

  const storedKey = (await context.secrets.get(SECRET_API_KEY))?.trim();
  const apiKey = storedKey || config.apiKey;

  const gitRemote = await getGitRemote(workspaceRoot);
  const datasetName = deriveDatasetName({
    workspaceRoot,
    gitRemote,
    override: config.datasetOverride,
  });

  const client = new HttpCogneeClient({
    endpoint: config.endpoint,
    apiKey,
    timeoutMs: config.requestTimeoutMs,
  });

  logger.info(`Resolved dataset "${datasetName}" for endpoint ${config.endpoint}.`);
  return { config, client, datasetName, workspaceRoot, logger };
}
