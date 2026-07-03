import * as vscode from "vscode";

import { describeError } from "../../core";
import type { Runtime } from "../runtime";

const CLEAR_MEMORY = "Clear memory (keep files)";
const DELETE_ALL = "Delete everything";

/**
 * Forget the workspace's project memory. Offers a safe default (clear the graph
 * and embeddings but keep raw files so it can be re-indexed) and a destructive
 * option (delete the whole dataset). Both are confirmed via a modal.
 */
export async function forgetProject(runtime: Runtime): Promise<void> {
  const choice = await vscode.window.showWarningMessage(
    `Forget project memory for "${runtime.datasetName}"?`,
    {
      modal: true,
      detail:
        `"${CLEAR_MEMORY}" clears the knowledge graph and embeddings so you can re-index.\n` +
        `"${DELETE_ALL}" permanently removes the dataset and its ingested data.`,
    },
    CLEAR_MEMORY,
    DELETE_ALL,
  );
  if (choice !== CLEAR_MEMORY && choice !== DELETE_ALL) {
    return;
  }

  const memoryOnly = choice === CLEAR_MEMORY;
  try {
    await runtime.client.forget({ dataset: runtime.datasetName, memoryOnly });
    runtime.logger.info(`forget dataset=${runtime.datasetName} memoryOnly=${memoryOnly}`);
    void vscode.window.showInformationMessage(
      memoryOnly
        ? `Cognee: cleared memory for ${runtime.datasetName} (files kept).`
        : `Cognee: deleted dataset ${runtime.datasetName}.`,
    );
  } catch (error) {
    runtime.logger.error("forget failed", error);
    void vscode.window.showErrorMessage(`Cognee: ${describeError(error)}`);
  }
}
