import * as vscode from "vscode";

import { describeError } from "../../core";
import { basenameOf } from "../paths";
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

  const relativePath = vscode.workspace.asRelativePath(document.uri, false);
  // A full-line selection ends at column 0 of the following line; that trailing
  // line holds no selected text, so exclude it from the recorded range.
  const lastLine =
    selection.end.character === 0 && selection.end.line > selection.start.line
      ? selection.end.line
      : selection.end.line + 1;
  const lineRange = selection.isEmpty
    ? undefined
    : `lines ${selection.start.line + 1}-${lastLine}`;

  await ingest(
    runtime,
    decorate(text, relativePath, lineRange),
    basenameOf(relativePath),
    `Remembering from ${relativePath}…`,
    relativePath,
  );
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

  const relativePath = vscode.workspace.asRelativePath(uri, false);
  await ingest(
    runtime,
    decorate(text, relativePath),
    basenameOf(relativePath),
    `Remembering ${relativePath}…`,
    relativePath,
  );
}

/** Remember a free-form note typed by the user (no file backing). */
export async function rememberNote(runtime: Runtime): Promise<void> {
  const note = await vscode.window.showInputBox({
    prompt: "Remember a note in project memory",
    placeHolder: "e.g. API fields use snake_case; auth tokens live in the session cache.",
    ignoreFocusOut: true,
  });
  // `undefined` = dismissed; an empty/whitespace note is nothing to store.
  if (note === undefined || !note.trim()) {
    return;
  }

  await ingest(runtime, note.trim(), "note.md", "Remembering note…");
}

/**
 * Ingest text into project memory. When the content comes from a file, pass its
 * workspace-relative path so the source is recorded and future citations resolve
 * straight to the exact file; free-form notes omit it.
 */
async function ingest(
  runtime: Runtime,
  data: string,
  filename: string,
  title: string,
  relativePath?: string,
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
        if (relativePath) {
          await runtime.pathIndex.record(relativePath);
        }
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
function decorate(text: string, relativePath: string, lineRange?: string): string {
  const source = lineRange ? `${relativePath} (${lineRange})` : relativePath;
  return `Source: ${source}\n\n${text}`;
}
