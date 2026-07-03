import * as vscode from "vscode";

import { normalizeEndpoint, type CogneeConfig } from "../core";

/** Read the extension's configuration from VS Code settings into a plain core config. */
export function readConfig(scope?: vscode.ConfigurationScope): CogneeConfig {
  const settings = vscode.workspace.getConfiguration("cognee", scope);

  return {
    endpoint: normalizeEndpoint(settings.get<string>("endpoint", "http://localhost:8011")),
    apiKey: settings.get<string>("apiKey", "").trim() || undefined,
    datasetOverride: settings.get<string>("datasetOverride", "").trim() || undefined,
    searchType: settings.get<string>("searchType", "auto"),
    topK: settings.get<number>("topK", 15),
    includeReferences: settings.get<boolean>("includeReferences", true),
    respectGitignore: settings.get<boolean>("ingestion.respectGitignore", true),
    maxFileSizeKb: settings.get<number>("ingestion.maxFileSizeKb", 512),
    requestTimeoutMs: settings.get<number>("requestTimeoutMs", 300000),
  };
}
