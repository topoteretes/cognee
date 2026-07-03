import * as vscode from "vscode";

import type { Citation } from "../../core";
import { contentMatchesSnippet, snippetSearchNeedles } from "./resolve";

const MAX_CANDIDATES = 50;

/**
 * Resolve a citation to a workspace file and reveal the cited region.
 *
 * Citations are chunk/document-level (by filename), so when several files share
 * the cited basename we disambiguate by which one actually contains the snippet,
 * and only ask the user to choose when that's inconclusive. When the source
 * can't be found or has changed, the user is told rather than misled.
 */
export async function openCitation(citation: Citation): Promise<void> {
  const basename = citation.documentName.split(/[\\/]/).pop()?.trim();
  if (!basename) {
    return;
  }

  const candidates = await vscode.workspace.findFiles(
    `**/${escapeGlob(basename)}`,
    "**/node_modules/**",
    MAX_CANDIDATES,
  );
  if (candidates.length === 0) {
    void vscode.window.showWarningMessage(
      `Cognee: couldn't locate "${citation.documentName}" in this workspace (source may have moved).`,
    );
    return;
  }

  const uri = await selectCitationTarget(candidates, citation, basename);
  if (!uri) {
    return; // user dismissed the disambiguation picker
  }

  const document = await vscode.workspace.openTextDocument(uri);
  const editor = await vscode.window.showTextDocument(document);
  if (citation.snippet) {
    revealSnippet(editor, citation.snippet);
  }
}

/**
 * Choose which candidate file the citation refers to:
 * 1. a single match wins outright;
 * 2. otherwise prefer the file whose content contains the cited snippet;
 * 3. otherwise let the user pick, shallowest path first.
 */
async function selectCitationTarget(
  candidates: vscode.Uri[],
  citation: Citation,
  basename: string,
): Promise<vscode.Uri | undefined> {
  if (candidates.length === 1) {
    return candidates[0];
  }

  if (citation.snippet) {
    for (const uri of candidates) {
      try {
        const bytes = await vscode.workspace.fs.readFile(uri);
        if (contentMatchesSnippet(Buffer.from(bytes).toString("utf8"), citation.snippet)) {
          return uri;
        }
      } catch {
        // Unreadable candidate — skip it.
      }
    }
  }

  const items = candidates
    .map((uri) => ({ label: vscode.workspace.asRelativePath(uri, false), uri }))
    .sort((a, b) => a.label.length - b.label.length);
  const picked = await vscode.window.showQuickPick(items, {
    placeHolder: `Multiple files named "${basename}" — choose the cited source`,
  });
  return picked?.uri;
}

function revealSnippet(editor: vscode.TextEditor, snippet: string): void {
  const haystack = editor.document.getText();
  for (const needle of snippetSearchNeedles(snippet)) {
    const index = haystack.indexOf(needle);
    if (index === -1) {
      continue;
    }
    const range = new vscode.Range(
      editor.document.positionAt(index),
      editor.document.positionAt(index + needle.length),
    );
    editor.selection = new vscode.Selection(range.start, range.end);
    editor.revealRange(range, vscode.TextEditorRevealType.InCenter);
    return;
  }
}

function escapeGlob(name: string): string {
  return name.replace(/[*?{}[\]]/g, "");
}
