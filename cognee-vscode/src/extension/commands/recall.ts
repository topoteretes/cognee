import * as vscode from "vscode";

import { describeError, resolveSearchType } from "../../core";
import type { Runtime } from "../runtime";
import { openCitation } from "../ui/citations";
import { dedupeByFile, rankCitations, renderRecall, type RenderedRecall } from "../ui/render";

/** Prompt for a query, recall project memory, and present the answer + sources. */
export async function recallCommand(runtime: Runtime): Promise<void> {
  const query = await vscode.window.showInputBox({
    prompt: "Ask your project memory",
    placeHolder: "e.g. How is authentication handled in this project?",
  });
  if (!query || !query.trim()) {
    return;
  }

  const rendered = await runRecall(runtime, query.trim());
  if (!rendered) {
    return;
  }
  if (rendered.isEmpty) {
    void vscode.window.showInformationMessage(
      "Cognee: no answer yet — remember or index something first.",
    );
    return;
  }

  await presentResult(runtime, rendered);
}

/**
 * Execute a recall against the workspace dataset with a progress notification.
 * Shared by the command and the "Ask my project memory" panel. Returns undefined
 * on error (already surfaced to the user).
 */
export async function runRecall(
  runtime: Runtime,
  query: string,
): Promise<RenderedRecall | undefined> {
  return vscode.window.withProgress(
    { location: vscode.ProgressLocation.Notification, title: "Cognee: recalling…", cancellable: true },
    async (_progress, token) => {
      const controller = new AbortController();
      token.onCancellationRequested(() => controller.abort());
      try {
        const items = await runtime.client.recall(query, {
          datasets: [runtime.datasetName],
          searchType: resolveSearchType(runtime.config.searchType),
          topK: runtime.config.topK,
          includeReferences: runtime.config.includeReferences,
          signal: controller.signal,
        });
        const rendered = renderRecall(items);
        // Present sources the professional way: rank by relevance to the query
        // (Cognee returns the Evidence list unranked), then collapse repeated
        // chunks of the same file so each cited file appears once. Only the
        // source list is reordered/de-duplicated — never the answer.
        const citations = dedupeByFile(rankCitations(rendered.citations, query));
        return { ...rendered, citations };
      } catch (error) {
        runtime.logger.error("recall failed", error);
        void vscode.window.showErrorMessage(`Cognee: ${describeError(error)}`);
        return undefined;
      }
    },
  );
}

async function presentResult(runtime: Runtime, rendered: RenderedRecall): Promise<void> {
  const document = await vscode.workspace.openTextDocument({
    language: "markdown",
    content: toMarkdown(rendered),
  });
  await vscode.window.showTextDocument(document, { preview: true });

  if (rendered.citations.length === 0) {
    return;
  }

  const picked = await vscode.window.showQuickPick(
    rendered.citations.map((citation) => ({
      label: citation.documentName,
      description: citation.chunkId ?? "",
      detail: citation.snippet ?? "",
      citation,
    })),
    { placeHolder: "Open a cited source" },
  );
  if (picked) {
    await openCitation(picked.citation, runtime.pathIndex);
  }
}

function toMarkdown(rendered: RenderedRecall): string {
  const lines = ["# Cognee — project memory", "", rendered.answer.trim() || "_No answer._"];
  if (rendered.citations.length > 0) {
    lines.push("", "## Sources");
    for (const citation of rendered.citations) {
      const snippet = citation.snippet ? ` — ${citation.snippet}` : "";
      lines.push(`- \`${citation.documentName}\`${snippet}`);
    }
  }
  return lines.join("\n");
}
