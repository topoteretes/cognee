import * as vscode from "vscode";

/** Thin wrapper over a VS Code output channel with timestamped, levelled lines. */
export class Logger implements vscode.Disposable {
  private readonly channel: vscode.OutputChannel;

  constructor() {
    this.channel = vscode.window.createOutputChannel("Cognee");
  }

  info(message: string): void {
    this.write("info", message);
  }

  error(message: string, error?: unknown): void {
    const detail = error instanceof Error ? `${error.name}: ${error.message}` : error ? String(error) : "";
    this.write("error", detail ? `${message} — ${detail}` : message);
  }

  show(): void {
    this.channel.show(true);
  }

  dispose(): void {
    this.channel.dispose();
  }

  private write(level: "info" | "error", message: string): void {
    this.channel.appendLine(`[${new Date().toISOString()}] [${level}] ${message}`);
  }
}
