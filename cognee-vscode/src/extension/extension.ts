import * as vscode from "vscode";

import { forgetProject } from "./commands/forget";
import { indexWorkspace } from "./commands/indexWorkspace";
import { recallCommand } from "./commands/recall";
import { rememberFile, rememberNote, rememberSelection } from "./commands/remember";
import { setup } from "./commands/setup";
import { Logger } from "./logger";
import { resolveRuntime } from "./runtime";
import { AskPanel } from "./ui/askPanel";

export function activate(context: vscode.ExtensionContext): void {
  const logger = new Logger();
  context.subscriptions.push(logger);

  context.subscriptions.push(
    vscode.commands.registerCommand("cognee.rememberSelection", async () => {
      const runtime = await resolveRuntime(context, logger);
      if (runtime) {
        await rememberSelection(runtime);
      }
    }),
    vscode.commands.registerCommand("cognee.rememberFile", async (resource?: vscode.Uri) => {
      const runtime = await resolveRuntime(context, logger);
      if (runtime) {
        await rememberFile(runtime, resource);
      }
    }),
    vscode.commands.registerCommand("cognee.rememberNote", async () => {
      const runtime = await resolveRuntime(context, logger);
      if (runtime) {
        await rememberNote(runtime);
      }
    }),
    vscode.commands.registerCommand("cognee.recall", async () => {
      const runtime = await resolveRuntime(context, logger);
      if (runtime) {
        await recallCommand(runtime);
      }
    }),
    vscode.commands.registerCommand("cognee.indexWorkspace", async () => {
      const runtime = await resolveRuntime(context, logger);
      if (runtime) {
        await indexWorkspace(runtime);
      }
    }),
    vscode.commands.registerCommand("cognee.forgetProject", async () => {
      const runtime = await resolveRuntime(context, logger);
      if (runtime) {
        await forgetProject(runtime);
      }
    }),
    vscode.commands.registerCommand("cognee.askProjectMemory", () => {
      AskPanel.show(() => resolveRuntime(context, logger));
    }),
    vscode.commands.registerCommand("cognee.setup", () => setup(context, logger)),
  );

  logger.info("Cognee extension activated.");
}

export function deactivate(): void {
  // Nothing to clean up beyond context.subscriptions.
}
