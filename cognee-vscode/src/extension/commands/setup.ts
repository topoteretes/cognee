import * as vscode from "vscode";

import { HttpCogneeClient, validateConfig } from "../../core";
import type { Logger } from "../logger";
import { SECRET_API_KEY } from "../runtime";
import { readConfig } from "../settings";
import { getWorkspaceRoot } from "../workspace";

/**
 * Guided onboarding: capture the endpoint and (optionally) an API key, store the
 * key in the OS keychain via secret storage, then run a health check so the user
 * knows their connection works.
 *
 * The health check does not require an open folder — only Remember/Ask do — so
 * Set Up is useful even on an empty window.
 */
export async function setup(context: vscode.ExtensionContext, logger: Logger): Promise<void> {
  const current = readConfig();

  const endpoint = await vscode.window.showInputBox({
    prompt: "Cognee endpoint",
    value: current.endpoint,
    placeHolder: "http://localhost:8011 or https://<tenant>.cognee.ai",
    validateInput: (value) =>
      /^https?:\/\//i.test(value.trim()) ? undefined : "Must start with http:// or https://",
  });
  if (endpoint === undefined) {
    return;
  }
  await vscode.workspace
    .getConfiguration("cognee")
    .update("endpoint", endpoint.trim(), vscode.ConfigurationTarget.Global);

  const apiKey = await vscode.window.showInputBox({
    prompt: "Cognee API key (leave empty for a local server)",
    password: true,
    placeHolder: "Stored securely in the OS keychain — not in settings",
  });
  if (apiKey !== undefined) {
    if (apiKey.trim()) {
      await context.secrets.store(SECRET_API_KEY, apiKey.trim());
    } else {
      await context.secrets.delete(SECRET_API_KEY);
    }
  }

  const saved = readConfig();
  const validation = validateConfig(saved);
  if (!validation.ok) {
    void vscode.window.showErrorMessage(`Cognee: ${validation.errors.join(" ")}`);
    return;
  }

  const storedKey = (await context.secrets.get(SECRET_API_KEY))?.trim();
  const client = new HttpCogneeClient({
    endpoint: saved.endpoint,
    apiKey: storedKey || saved.apiKey,
    timeoutMs: saved.requestTimeoutMs,
  });

  const reachable = await client.health();
  logger.info(`setup: health check ${reachable ? "ok" : "unreachable"} for ${saved.endpoint}`);

  if (!reachable) {
    void vscode.window.showWarningMessage(
      `Cognee: settings saved, but ${saved.endpoint} isn't reachable yet. ` +
        "Start the Cognee server or re-check the endpoint/key.",
    );
    return;
  }

  const hasFolder = getWorkspaceRoot() !== undefined;
  void vscode.window.showInformationMessage(
    hasFolder
      ? `Cognee: connected to ${saved.endpoint}. Ready — try "Cognee: Ask My Project Memory".`
      : `Cognee: connected to ${saved.endpoint}. Open a folder to start capturing project memory.`,
  );
}
