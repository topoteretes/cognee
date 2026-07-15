import * as vscode from "vscode";

import type { Citation } from "../../core";
import { basenameOf, knownPathMatches, normalizeRelativePath } from "../paths";
import type { PathIndex } from "../pathIndex";
import {
  contentMatchesSnippet,
  extractSourcePath,
  findSnippetRange,
  stripProvenanceHeader,
} from "./resolve";

const MAX_CANDIDATES = 50;

/**
 * Resolve a citation to a workspace file and reveal the cited region.
 *
 * Resolution order, most reliable first:
 * 0. the exact `Source: <path>` the extension recorded at ingest (deterministic);
 * 1. the file(s) we actually ingested for this document name (path index);
 * 2. the file whose content contains the cited snippet;
 * 3. a user pick (only when still genuinely ambiguous).
 *
 * When the source can't be found or has changed, the user is told rather than
 * silently sent to the wrong file.
 */
export async function openCitation(citation: Citation, pathIndex?: PathIndex): Promise<void> {
  // The reveal text is the snippet without our ingest-time `Source: …` header,
  // so needles match the file's real content.
  const revealText = stripProvenanceHeader(citation.snippet);

  // 0. Deterministic: our own provenance header names the exact file.
  const sourcePath = extractSourcePath(citation.snippet);
  if (sourcePath) {
    const exact = await findWorkspaceFile(sourcePath);
    if (exact) {
      await openAndReveal(exact, revealText);
      return;
    }
  }

  const basename = basenameOf(citation.documentName);
  if (!basename) {
    return;
  }

  const candidates = await findCandidates(basename);
  if (candidates.length === 0) {
    void vscode.window.showWarningMessage(
      `Cognee: couldn't locate "${citation.documentName}" in this workspace (source may have moved).`,
    );
    return;
  }

  const knownPaths = pathIndex?.pathsFor(citation.documentName) ?? [];
  const uri = await selectCitationTarget(candidates, basename, knownPaths, revealText);
  if (!uri) {
    return; // user dismissed the disambiguation picker
  }

  await openAndReveal(uri, revealText);
}

/** Open a document and reveal the cited region when a snippet is available. */
async function openAndReveal(uri: vscode.Uri, revealText: string): Promise<void> {
  const document = await vscode.workspace.openTextDocument(uri);
  const editor = await vscode.window.showTextDocument(document);
  if (revealText) {
    revealSnippet(editor, revealText);
  }
}

/** Resolve a workspace-relative path to an existing file, across all folders. */
async function findWorkspaceFile(relativePath: string): Promise<vscode.Uri | undefined> {
  const normalized = normalizeRelativePath(relativePath);
  if (!normalized) {
    return undefined;
  }
  for (const folder of vscode.workspace.workspaceFolders ?? []) {
    const uri = vscode.Uri.joinPath(folder.uri, normalized);
    try {
      const stat = await vscode.workspace.fs.stat(uri);
      if ((stat.type & vscode.FileType.File) !== 0) {
        return uri;
      }
    } catch {
      // Not under this folder — try the next.
    }
  }
  return undefined;
}

/** Find files matching the cited basename, with a stem fallback if it had no extension. */
async function findCandidates(basename: string): Promise<vscode.Uri[]> {
  const direct = await vscode.workspace.findFiles(
    `**/${escapeGlob(basename)}`,
    "**/node_modules/**",
    MAX_CANDIDATES,
  );
  if (direct.length > 0 || basename.includes(".")) {
    return direct;
  }
  // The backend may drop the extension from the document name; try `<name>.*`.
  return vscode.workspace.findFiles(`**/${escapeGlob(basename)}.*`, "**/node_modules/**", MAX_CANDIDATES);
}

async function selectCitationTarget(
  candidates: vscode.Uri[],
  basename: string,
  knownPaths: string[],
  revealText: string,
): Promise<vscode.Uri | undefined> {
  // 1. Prefer files we actually ingested for this document name.
  const relativePaths = candidates.map((uri) => vscode.workspace.asRelativePath(uri, false));
  const knownIndexes = knownPathMatches(relativePaths, knownPaths);
  const pool = knownIndexes.length > 0 ? knownIndexes.map((index) => candidates[index]) : candidates;

  if (pool.length === 1) {
    return pool[0];
  }

  // 2. Prefer the file whose content contains the cited snippet.
  if (revealText) {
    for (const uri of pool) {
      try {
        const bytes = await vscode.workspace.fs.readFile(uri);
        if (contentMatchesSnippet(Buffer.from(bytes).toString("utf8"), revealText)) {
          return uri;
        }
      } catch {
        // Unreadable candidate — skip it.
      }
    }
  }

  // 3. Ask the user, shallowest path first.
  const items = pool
    .map((uri) => ({ label: vscode.workspace.asRelativePath(uri, false), uri }))
    .sort((a, b) => a.label.length - b.label.length);
  const picked = await vscode.window.showQuickPick(items, {
    placeHolder: `Multiple files named "${basename}" — choose the cited source`,
  });
  return picked?.uri;
}

function revealSnippet(editor: vscode.TextEditor, snippet: string): void {
  const match = findSnippetRange(editor.document.getText(), snippet);
  if (!match) {
    return;
  }
  const range = new vscode.Range(
    editor.document.positionAt(match[0]),
    editor.document.positionAt(match[1]),
  );
  editor.selection = new vscode.Selection(range.start, range.end);
  editor.revealRange(range, vscode.TextEditorRevealType.InCenter);
}

function escapeGlob(name: string): string {
  return name.replace(/[*?{}[\]]/g, "");
}
