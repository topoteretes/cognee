/**
 * MCP App UI for Cognee's visualize_graph_ui tool.
 * Receives pre-rendered graph HTML via structuredContent.html and renders it
 * in a nested sandboxed iframe via srcdoc.
 */
import type { McpUiHostContext } from "@modelcontextprotocol/ext-apps";
import { useApp } from "@modelcontextprotocol/ext-apps/react";
import type { CallToolResult } from "@modelcontextprotocol/sdk/types.js";
import { StrictMode, useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";

const WRAPPER_STYLE: React.CSSProperties = {
  width: "100%",
  display: "flex",
  flexDirection: "column",
  overflow: "hidden",
};

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

function VisualizeGraphApp() {
  const [toolResult, setToolResult] = useState<CallToolResult | null>(null);
  const [hostContext, setHostContext] = useState<McpUiHostContext | undefined>();

  const { app, error } = useApp({
    appInfo: { name: "Cognee Graph Visualization", version: "0.1.0" },
    capabilities: {},
    onAppCreated: (app) => {
      app.onteardown = async () => ({});
      app.ontoolresult = async (result) => setToolResult(result);
      app.onhostcontextchanged = (params) =>
        setHostContext((prev) => ({ ...prev, ...params }));
      app.onerror = console.error;
    },
  });

  useEffect(() => {
    if (app) setHostContext(app.getHostContext());
  }, [app]);

  const html = useMemo(() => {
    const structured = toolResult?.structuredContent as { html?: string } | undefined;
    return typeof structured?.html === "string" ? structured.html : null;
  }, [toolResult]);

  const wrapperStyle: React.CSSProperties = {
    ...WRAPPER_STYLE,
    height: resolveWrapperHeight(hostContext),
  };

  if (error) return <div style={wrapperStyle}>ERROR: {error.message}</div>;
  if (!app) return <div style={wrapperStyle}>Connecting...</div>;
  if (!html) return <div style={wrapperStyle}>Waiting for graph data from Cognee...</div>;

  return (
    <div style={wrapperStyle}>
      <iframe
        title="Cognee Knowledge Graph"
        srcDoc={html}
        style={{ flex: 1, width: "100%", border: 0 }}
        sandbox="allow-scripts allow-same-origin"
      />
    </div>
  );
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <VisualizeGraphApp />
  </StrictMode>,
);
