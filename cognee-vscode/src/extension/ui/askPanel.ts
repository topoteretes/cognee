import * as vscode from "vscode";

import type { Citation } from "../../core";
import { runRecall } from "../commands/recall";
import type { PathIndex } from "../pathIndex";
import type { Runtime } from "../runtime";
import { openCitation } from "./citations";

/** Lazily resolves a fresh runtime so settings changes apply on every question. */
export type RuntimeResolver = () => Promise<Runtime | undefined>;

interface AskMessage {
  type: "ask";
  query: string;
}
interface OpenCitationMessage {
  type: "openCitation";
  index: number;
}
type InboundMessage = AskMessage | OpenCitationMessage;

/**
 * The "Ask my project memory" panel: a single webview that queries the
 * workspace's Cognee dataset and renders answers with clickable citations.
 */
export class AskPanel {
  private static current: AskPanel | undefined;

  private readonly disposables: vscode.Disposable[] = [];
  private citations: Citation[] = [];
  private pathIndex: PathIndex | undefined;

  static show(resolveRuntime: RuntimeResolver): void {
    const column = vscode.window.activeTextEditor?.viewColumn ?? vscode.ViewColumn.One;
    if (AskPanel.current) {
      AskPanel.current.panel.reveal(column);
      return;
    }
    const panel = vscode.window.createWebviewPanel(
      "cognee.askPanel",
      "Ask My Project Memory",
      column,
      { enableScripts: true, retainContextWhenHidden: true },
    );
    AskPanel.current = new AskPanel(panel, resolveRuntime);
  }

  private constructor(
    private readonly panel: vscode.WebviewPanel,
    private readonly resolveRuntime: RuntimeResolver,
  ) {
    this.panel.webview.html = this.render(this.panel.webview);
    this.panel.onDidDispose(() => this.dispose(), null, this.disposables);
    this.panel.webview.onDidReceiveMessage(
      (message: InboundMessage) => void this.handleMessage(message),
      null,
      this.disposables,
    );
  }

  private async handleMessage(message: InboundMessage): Promise<void> {
    if (message.type === "ask") {
      await this.handleAsk(message.query);
    } else if (message.type === "openCitation") {
      const citation = this.citations[message.index];
      if (citation) {
        await openCitation(citation, this.pathIndex);
      }
    }
  }

  private async handleAsk(rawQuery: string): Promise<void> {
    const query = rawQuery.trim();
    if (!query) {
      return;
    }
    this.post({ type: "status", state: "loading" });

    const runtime = await this.resolveRuntime();
    if (!runtime) {
      this.post({ type: "status", state: "error", message: "Configure Cognee first (Cognee: Set Up)." });
      return;
    }
    this.pathIndex = runtime.pathIndex;

    const rendered = await runRecall(runtime, query);
    if (!rendered) {
      this.post({ type: "status", state: "error", message: "Recall failed — see the Cognee output channel." });
      return;
    }

    this.citations = rendered.citations;
    this.post({
      type: "answer",
      answer: rendered.isEmpty ? "" : rendered.answer,
      citations: rendered.citations.map((citation, index) => ({
        index,
        documentName: citation.documentName,
        snippet: citation.snippet ?? "",
      })),
    });
  }

  private post(message: unknown): void {
    void this.panel.webview.postMessage(message);
  }

  private dispose(): void {
    AskPanel.current = undefined;
    this.panel.dispose();
    while (this.disposables.length) {
      this.disposables.pop()?.dispose();
    }
  }

  private render(webview: vscode.Webview): string {
    const nonce = getNonce();
    const csp = [
      "default-src 'none'",
      `style-src ${webview.cspSource} 'unsafe-inline'`,
      `script-src 'nonce-${nonce}'`,
    ].join("; ");

    return /* html */ `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta http-equiv="Content-Security-Policy" content="${csp}" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Ask My Project Memory</title>
  <style>
    body { font-family: var(--vscode-font-family); color: var(--vscode-foreground);
           padding: 12px 16px; font-size: var(--vscode-font-size); }
    h1 { font-size: 1.1em; margin: 0 0 12px; }
    .row { display: flex; gap: 8px; }
    textarea { flex: 1; resize: vertical; min-height: 44px; padding: 6px 8px;
      color: var(--vscode-input-foreground); background: var(--vscode-input-background);
      border: 1px solid var(--vscode-input-border, transparent); border-radius: 4px;
      font-family: inherit; font-size: inherit; }
    button { color: var(--vscode-button-foreground); background: var(--vscode-button-background);
      border: none; padding: 6px 14px; border-radius: 4px; cursor: pointer; }
    button:hover { background: var(--vscode-button-hoverBackground); }
    #status { min-height: 18px; margin: 10px 0; color: var(--vscode-descriptionForeground); }
    #status.error { color: var(--vscode-errorForeground); }
    #answer { white-space: pre-wrap; line-height: 1.5; margin-top: 4px; }
    h2 { font-size: 0.95em; margin: 18px 0 6px; color: var(--vscode-descriptionForeground); }
    .cite { display: block; width: 100%; text-align: left; margin: 4px 0; padding: 6px 8px;
      background: var(--vscode-editor-inactiveSelectionBackground); color: var(--vscode-foreground);
      border: none; border-radius: 4px; cursor: pointer; }
    .cite:hover { background: var(--vscode-list-hoverBackground); }
    .cite .name { font-weight: 600; }
    .cite .snippet { display: block; color: var(--vscode-descriptionForeground);
      font-size: 0.9em; margin-top: 2px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  </style>
</head>
<body>
  <h1>Ask my project memory</h1>
  <div class="row">
    <textarea id="query" placeholder="e.g. How is authentication handled in this project?"></textarea>
    <button id="ask">Ask</button>
  </div>
  <div id="status"></div>
  <div id="answer"></div>
  <div id="sources"></div>

  <script nonce="${nonce}">
    const vscode = acquireVsCodeApi();
    const queryEl = document.getElementById('query');
    const statusEl = document.getElementById('status');
    const answerEl = document.getElementById('answer');
    const sourcesEl = document.getElementById('sources');

    function ask() {
      const query = queryEl.value.trim();
      if (!query) { return; }
      vscode.postMessage({ type: 'ask', query });
    }

    document.getElementById('ask').addEventListener('click', ask);
    queryEl.addEventListener('keydown', (event) => {
      if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') { ask(); }
    });

    window.addEventListener('message', (event) => {
      const message = event.data;
      if (message.type === 'status') {
        statusEl.className = message.state === 'error' ? 'error' : '';
        statusEl.textContent = message.state === 'loading'
          ? 'Recalling…'
          : (message.message || '');
        if (message.state !== 'answer') { /* keep answer until replaced */ }
      } else if (message.type === 'answer') {
        statusEl.className = '';
        statusEl.textContent = '';
        answerEl.textContent = message.answer || 'No answer yet — remember or index something first.';
        renderSources(message.citations || []);
      }
    });

    function renderSources(citations) {
      sourcesEl.innerHTML = '';
      if (!citations.length) { return; }
      const heading = document.createElement('h2');
      heading.textContent = 'Sources';
      sourcesEl.appendChild(heading);
      for (const citation of citations) {
        const button = document.createElement('button');
        button.className = 'cite';
        const name = document.createElement('span');
        name.className = 'name';
        name.textContent = citation.documentName;
        button.appendChild(name);
        if (citation.snippet) {
          const snippet = document.createElement('span');
          snippet.className = 'snippet';
          snippet.textContent = citation.snippet;
          button.appendChild(snippet);
        }
        button.addEventListener('click', () => {
          vscode.postMessage({ type: 'openCitation', index: citation.index });
        });
        sourcesEl.appendChild(button);
      }
    }
  </script>
</body>
</html>`;
  }
}

function getNonce(): string {
  const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  let text = "";
  for (let i = 0; i < 32; i += 1) {
    text += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return text;
}
