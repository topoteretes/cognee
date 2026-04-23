/**
 * Cognee workspace MCP App: graph visualization + file upload.
 * The UI fetches its own graph data on mount so it works as the entry
 * for either visualize_graph_ui or upload_file_ui.
 */
import type { App, McpUiHostContext } from "@modelcontextprotocol/ext-apps";
import { useApp } from "@modelcontextprotocol/ext-apps/react";
import type { CallToolResult } from "@modelcontextprotocol/sdk/types.js";
import { StrictMode, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";

const MAX_UPLOAD_BYTES = 10 * 1024 * 1024;

function arrayBufferToBase64(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer);
  let binary = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}

function textOf(result: CallToolResult): string | undefined {
  return result.content?.find((c) => c.type === "text")?.text;
}

function htmlOf(result: CallToolResult): string | null {
  const s = result.structuredContent as { html?: string } | undefined;
  return typeof s?.html === "string" ? s.html : null;
}

function resolveWrapperHeight(
  hostContext: McpUiHostContext | undefined,
): number | string {
  const dims = hostContext?.containerDimensions as
    | { height?: number; maxHeight?: number }
    | undefined;
  if (typeof dims?.height === "number") return dims.height;
  if (typeof dims?.maxHeight === "number") return dims.maxHeight;
  if (hostContext?.displayMode === "fullscreen") return "100%";
  return 600;
}

function Toolbar({
  app,
  onGraphHtml,
  setGraphStatus,
}: {
  app: App;
  onGraphHtml: (html: string | null) => void;
  setGraphStatus: (s: string) => void;
}) {
  const [status, setStatus] = useState("");
  const [busy, setBusy] = useState(false);

  const handleFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;
    if (file.size > MAX_UPLOAD_BYTES) {
      setStatus(`"${file.name}" is ${(file.size / 1024 / 1024).toFixed(1)} MB (10 MB max).`);
      return;
    }
    setBusy(true);
    setStatus(`Uploading ${file.name}...`);
    try {
      const content_base64 = arrayBufferToBase64(await file.arrayBuffer());
      const result = await app.callServerTool({
        name: "cognify_file",
        arguments: { filename: file.name, content_base64 },
      });
      const text = textOf(result) ?? JSON.stringify(result.content);
      setStatus(result.isError ? `Error: ${text}` : text);
    } catch (err) {
      setStatus(`Error: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setBusy(false);
    }
  };

  const handleRefresh = async () => {
    setBusy(true);
    setGraphStatus("Loading graph...");
    try {
      const result = await app.callServerTool({ name: "visualize_graph_ui", arguments: {} });
      if (result.isError) {
        setGraphStatus(`Error: ${textOf(result) ?? "refresh failed"}`);
        return;
      }
      const html = htmlOf(result);
      if (html) {
        onGraphHtml(html);
        setGraphStatus("");
      } else {
        onGraphHtml(null);
        setGraphStatus("No graph data yet.");
      }
    } catch (err) {
      setGraphStatus(`Error: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div
      style={{
        display: "flex",
        gap: 12,
        alignItems: "center",
        padding: "8px 12px",
        borderBottom: "1px solid #e5e7eb",
        fontSize: "0.875rem",
      }}
    >
      <label style={{ fontWeight: 500 }}>Upload file:</label>
      <input type="file" onChange={handleFile} disabled={busy} />
      <button type="button" onClick={handleRefresh} disabled={busy}>
        Refresh graph
      </button>
      {status && <span style={{ color: "#4b5563", flex: 1 }}>{status}</span>}
    </div>
  );
}

function GraphPane({
  html,
  status,
}: {
  html: string | null;
  status: string;
}) {
  if (html) {
    return (
      <iframe
        title="Cognee Knowledge Graph"
        srcDoc={html}
        style={{ flex: 1, width: "100%", border: 0 }}
        sandbox="allow-scripts allow-same-origin"
      />
    );
  }
  return (
    <div style={{ flex: 1, padding: 16, color: "#6b7280" }}>
      {status || "No graph data yet. Upload a file above or cognify data to get started."}
    </div>
  );
}

function WorkspaceApp() {
  const [hostContext, setHostContext] = useState<McpUiHostContext | undefined>();
  const [graphHtml, setGraphHtml] = useState<string | null>(null);
  const [graphStatus, setGraphStatus] = useState<string>("Loading graph...");

  const { app, error } = useApp({
    appInfo: { name: "Cognee Workspace", version: "0.1.0" },
    capabilities: {},
    onAppCreated: (app) => {
      app.ontoolresult = async (result) => {
        const html = htmlOf(result);
        if (html) {
          setGraphHtml(html);
          setGraphStatus("");
        }
      };
      app.onhostcontextchanged = (params) =>
        setHostContext((prev) => ({ ...prev, ...params }));
      app.onerror = console.error;
    },
  });

  useEffect(() => {
    if (app) setHostContext(app.getHostContext());
  }, [app]);

  useEffect(() => {
    if (!app) return;
    (async () => {
      try {
        const result = await app.callServerTool({ name: "visualize_graph_ui", arguments: {} });
        if (result.isError) {
          setGraphStatus(`Error: ${textOf(result) ?? "failed to load graph"}`);
          return;
        }
        const html = htmlOf(result);
        if (html) {
          setGraphHtml(html);
          setGraphStatus("");
        } else {
          setGraphStatus("No graph data yet. Upload a file above or cognify data to get started.");
        }
      } catch (err) {
        setGraphStatus(`Error: ${err instanceof Error ? err.message : String(err)}`);
      }
    })();
  }, [app]);

  const wrapperStyle: React.CSSProperties = {
    width: "100%",
    height: resolveWrapperHeight(hostContext),
    display: "flex",
    flexDirection: "column",
    overflow: "hidden",
  };

  if (error) return <div style={wrapperStyle}>ERROR: {error.message}</div>;
  if (!app) return <div style={wrapperStyle}>Connecting...</div>;

  return (
    <div style={wrapperStyle}>
      <Toolbar app={app} onGraphHtml={setGraphHtml} setGraphStatus={setGraphStatus} />
      <GraphPane html={graphHtml} status={graphStatus} />
    </div>
  );
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <WorkspaceApp />
  </StrictMode>,
);
