import * as vscode from "vscode";

import { describeError } from "../../core";
import type { Runtime } from "../runtime";

/** Remember the current selection (or the whole file when nothing is selected). */
export async function rememberSelection(runtime: Runtime): Promise<void> {
  const editor = vscode.window.activeTextEditor;
  if (!editor) {
    void vscode.window.showWarningMessage("Cognee: open a file and select text to remember.");
    return;
  }

  const { selection, document } = editor;
  const text = selection.isEmpty ? document.getText() : document.getText(selection);
  if (!text.trim()) {
    void vscode.window.showWarningMessage("Cognee: nothing to remember (the selection is empty).");
    return;
  }

  const label = vscode.workspace.asRelativePath(document.uri, false);
  const lineRange = selection.isEmpty
    ? undefined
    : `lines ${selection.start.line + 1}-${selection.end.line + 1}`;

  await ingest(runtime, decorate(text, label, lineRange), basename(document.uri), `Remembering from ${label}…`);
}

/** Remember an entire file, invoked from the explorer context menu or palette. */
export async function rememberFile(runtime: Runtime, resource?: vscode.Uri): Promise<void> {
  const uri = resource ?? vscode.window.activeTextEditor?.document.uri;
  if (!uri) {
    void vscode.window.showWarningMessage("Cognee: no file to remember.");
    return;
  }

  let bytes: Uint8Array;
  try {
    bytes = await vscode.workspace.fs.readFile(uri);
  } catch (error) {
    runtime.logger.error("rememberFile read failed", error);
    void vscode.window.showErrorMessage("Cognee: couldn't read that file.");
    return;
  }

  const text = Buffer.from(bytes).toString("utf8");
  if (!text.trim()) {
    void vscode.window.showWarningMessage("Cognee: that file is empty.");
    return;
  }

  const label = vscode.workspace.asRelativePath(uri, false);
  await ingest(runtime, decorate(text, label), basename(uri), `Remembering ${label}…`);
}

async function ingest(
  runtime: Runtime,
  data: string,
  filename: string,
  title: string,
): Promise<void> {
  await vscode.window.withProgress(
    { location: vscode.ProgressLocation.Notification, title, cancellable: true },
    async (_progress, token) => {
      const controller = new AbortController();
      token.onCancellationRequested(() => controller.abort());
      try {
        const result = await runtime.client.remember(data, {
          datasetName: runtime.datasetName,
          filename,
          signal: controller.signal,
        });
        runtime.logger.info(`remember → status=${result.status ?? "ok"} dataset=${runtime.datasetName}`);
        void vscode.window.showInformationMessage(
          `Cognee: remembered into project memory (${runtime.datasetName}).`,
        );
      } catch (error) {
        runtime.logger.error("remember failed", error);
        void vscode.window.showErrorMessage(`Cognee: ${describeError(error)}`);
      }
    },
  );
}

/** Prepend a provenance header so the source is preserved in memory. */
function decorate(text: string, label: string, lineRange?: string): string {
  const source = lineRange ? `${label} (${lineRange})` : label;
  return `Source: ${source}\n\n${text}`;
}

function basename(uri: vscode.Uri): string {
  return uri.path.split("/").pop() || "memory.txt";
}
