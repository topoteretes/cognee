/**
 * Cognee workspace MCP App: graph visualization + file upload.
 * The UI fetches its own graph data on mount so it works as the entry
 * for either visualize_graph_ui or upload_file_ui.
 */
import type { App, McpUiHostContext } from "@modelcontextprotocol/ext-apps";
import { useApp } from "@modelcontextprotocol/ext-apps/react";
import type { CallToolResult } from "@modelcontextprotocol/sdk/types.js";
import { StrictMode, useEffect, useRef, useState } from "react";
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

interface Dataset {
  id: string;
  name: string;
}

interface DataItem {
  id: string;
  name: string;
}

function datasetsOf(result: CallToolResult): Dataset[] {
  const s = result.structuredContent as { datasets?: Dataset[] } | undefined;
  return Array.isArray(s?.datasets) ? s.datasets : [];
}

function dataItemsOf(result: CallToolResult): DataItem[] {
  const s = result.structuredContent as { data?: DataItem[] } | undefined;
  return Array.isArray(s?.data) ? s.data : [];
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

function DatasetPicker({
  app,
  datasets,
  selectedDataset,
  disabled,
  dataRefreshToken,
  onSelectDataset,
  onDatasetsChanged,
  setStatus,
  setBusy,
}: {
  app: App;
  datasets: Dataset[];
  selectedDataset: string | null;
  disabled: boolean;
  dataRefreshToken: number;
  onSelectDataset: (name: string | null) => void;
  onDatasetsChanged: () => Promise<void>;
  setStatus: (s: string) => void;
  setBusy: (b: boolean) => void;
}) {
  const [open, setOpen] = useState(false);
  const [confirming, setConfirming] = useState<string | null>(null);
  const [newName, setNewName] = useState("");
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [dataItems, setDataItems] = useState<Record<string, DataItem[]>>({});
  const [loadingData, setLoadingData] = useState<Set<string>>(new Set());
  const [confirmingData, setConfirmingData] = useState<string | null>(null);
  const rootRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) {
        setOpen(false);
        setConfirming(null);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  const handleCreate = async () => {
    const name = newName.trim();
    if (!name) return;
    setBusy(true);
    setStatus(`Creating '${name}'...`);
    try {
      const result = await app.callServerTool({
        name: "create_dataset_json",
        arguments: { name },
      });
      if (result.isError) {
        setStatus(`Error: ${textOf(result) ?? "create failed"}`);
        return;
      }
      setNewName("");
      await onDatasetsChanged();
      onSelectDataset(name);
      setStatus(`Created '${name}'.`);
      setOpen(false);
    } catch (err) {
      setStatus(`Error: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setBusy(false);
    }
  };

  const loadDataItems = async (datasetId: string) => {
    setLoadingData((s) => new Set(s).add(datasetId));
    try {
      const result = await app.callServerTool({
        name: "list_dataset_data_json",
        arguments: { dataset_id: datasetId },
      });
      if (result.isError) {
        setStatus(`Error: ${textOf(result) ?? "failed to load data items"}`);
        return;
      }
      setDataItems((prev) => ({ ...prev, [datasetId]: dataItemsOf(result) }));
    } catch (err) {
      setStatus(`Error: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setLoadingData((s) => {
        const next = new Set(s);
        next.delete(datasetId);
        return next;
      });
    }
  };

  const toggleExpand = (datasetId: string) => {
    setExpanded((s) => {
      const next = new Set(s);
      if (next.has(datasetId)) {
        next.delete(datasetId);
      } else {
        next.add(datasetId);
        void loadDataItems(datasetId);
      }
      return next;
    });
  };

  useEffect(() => {
    if (dataRefreshToken === 0) return;
    setDataItems({});
    expanded.forEach((id) => void loadDataItems(id));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dataRefreshToken]);

  const handleDeleteData = async (datasetId: string, dataId: string) => {
    setBusy(true);
    setStatus("Deleting data item...");
    try {
      const result = await app.callServerTool({
        name: "delete",
        arguments: { data_id: dataId, dataset_id: datasetId },
      });
      if (result.isError) {
        setStatus(`Error: ${textOf(result) ?? "delete failed"}`);
        return;
      }
      setStatus(textOf(result) ?? "Data item deleted.");
      setConfirmingData(null);
      await loadDataItems(datasetId);
    } catch (err) {
      setStatus(`Error: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setBusy(false);
    }
  };

  const handleDelete = async (name: string) => {
    setBusy(true);
    setStatus(`Deleting '${name}'...`);
    try {
      const result = await app.callServerTool({
        name: "delete_dataset",
        arguments: { dataset_name: name },
      });
      if (result.isError) {
        setStatus(`Error: ${textOf(result) ?? "delete failed"}`);
        return;
      }
      setStatus(textOf(result) ?? `Deleted '${name}'.`);
      setConfirming(null);
      if (selectedDataset === name) onSelectDataset(null);
      await onDatasetsChanged();
    } catch (err) {
      setStatus(`Error: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div ref={rootRef} style={{ position: "relative" }}>
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        disabled={disabled}
        style={{ minWidth: 160, textAlign: "left", padding: "2px 8px" }}
      >
        {selectedDataset ?? "(no dataset)"} ▾
      </button>
      {open && (
        <div
          style={{
            position: "absolute",
            top: "calc(100% + 2px)",
            left: 0,
            zIndex: 10,
            minWidth: 240,
            background: "#fff",
            border: "1px solid #d1d5db",
            borderRadius: 4,
            boxShadow: "0 2px 8px rgba(0,0,0,0.1)",
            maxHeight: 260,
            overflowY: "auto",
          }}
        >
          {datasets.length === 0 && (
            <div style={{ padding: 8, color: "#6b7280" }}>No datasets yet.</div>
          )}
          {datasets.map((d) => {
            if (confirming === d.name) {
              return (
                <div
                  key={d.id}
                  style={{
                    padding: "6px 8px",
                    display: "flex",
                    gap: 6,
                    alignItems: "center",
                    background: "#fef2f2",
                  }}
                >
                  <span style={{ flex: 1, fontSize: "0.8rem" }}>Delete "{d.name}"?</span>
                  <button
                    type="button"
                    onClick={() => handleDelete(d.name)}
                    style={{ color: "#b91c1c" }}
                  >
                    Delete
                  </button>
                  <button type="button" onClick={() => setConfirming(null)}>
                    Cancel
                  </button>
                </div>
              );
            }
            const isSelected = selectedDataset === d.name;
            const isExpanded = expanded.has(d.id);
            const items = dataItems[d.id];
            const isLoading = loadingData.has(d.id);
            return (
              <div key={d.id}>
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 4,
                    padding: "4px 8px",
                    background: isSelected ? "#eef2ff" : "transparent",
                  }}
                >
                  <button
                    type="button"
                    onClick={() => toggleExpand(d.id)}
                    title={isExpanded ? "Hide data" : "Show data"}
                    style={{
                      border: "none",
                      background: "transparent",
                      cursor: "pointer",
                      color: "#6b7280",
                      width: 16,
                      padding: 0,
                    }}
                  >
                    {isExpanded ? "▾" : "▸"}
                  </button>
                  <span
                    onClick={() => {
                      onSelectDataset(d.name);
                      setOpen(false);
                    }}
                    style={{ flex: 1, cursor: "pointer" }}
                  >
                    {d.name}
                  </span>
                  <button
                    type="button"
                    onClick={() => setConfirming(d.name)}
                    title={`Delete ${d.name}`}
                    style={{
                      border: "none",
                      background: "transparent",
                      cursor: "pointer",
                      color: "#6b7280",
                      padding: "0 4px",
                    }}
                  >
                    ✕
                  </button>
                </div>
                {isExpanded && (
                  <div style={{ paddingLeft: 28, fontSize: "0.8rem" }}>
                    {isLoading && (
                      <div style={{ padding: "4px 8px", color: "#6b7280" }}>Loading...</div>
                    )}
                    {!isLoading && items && items.length === 0 && (
                      <div style={{ padding: "4px 8px", color: "#6b7280" }}>(empty)</div>
                    )}
                    {!isLoading &&
                      items &&
                      items.map((item) => {
                        const rowKey = `${d.id}:${item.id}`;
                        if (confirmingData === rowKey) {
                          return (
                            <div
                              key={item.id}
                              style={{
                                padding: "4px 8px",
                                display: "flex",
                                gap: 6,
                                alignItems: "center",
                                background: "#fef2f2",
                              }}
                            >
                              <span style={{ flex: 1 }}>Delete "{item.name}"?</span>
                              <button
                                type="button"
                                onClick={() => handleDeleteData(d.id, item.id)}
                                style={{ color: "#b91c1c" }}
                              >
                                Delete
                              </button>
                              <button
                                type="button"
                                onClick={() => setConfirmingData(null)}
                              >
                                Cancel
                              </button>
                            </div>
                          );
                        }
                        return (
                          <div
                            key={item.id}
                            style={{
                              display: "flex",
                              alignItems: "center",
                              gap: 6,
                              padding: "2px 8px",
                              color: "#4b5563",
                            }}
                          >
                            <span style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                              📄 {item.name}
                            </span>
                            <button
                              type="button"
                              onClick={() => setConfirmingData(rowKey)}
                              title={`Delete ${item.name}`}
                              style={{
                                border: "none",
                                background: "transparent",
                                cursor: "pointer",
                                color: "#6b7280",
                                padding: "0 4px",
                              }}
                            >
                              ✕
                            </button>
                          </div>
                        );
                      })}
                  </div>
                )}
              </div>
            );
          })}
          <div
            style={{
              borderTop: "1px solid #e5e7eb",
              padding: "6px 8px",
              display: "flex",
              gap: 6,
            }}
          >
            <input
              type="text"
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") void handleCreate();
              }}
              placeholder="New dataset name"
              style={{ flex: 1, minWidth: 80 }}
            />
            <button type="button" onClick={handleCreate} disabled={!newName.trim()}>
              Create
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

function Toolbar({
  app,
  datasets,
  selectedDataset,
  onSelectDataset,
  onDatasetsChanged,
  onGraphHtml,
  setGraphStatus,
  onSearchResult,
}: {
  app: App;
  datasets: Dataset[];
  selectedDataset: string | null;
  onSelectDataset: (name: string | null) => void;
  onDatasetsChanged: () => Promise<void>;
  onGraphHtml: (html: string | null) => void;
  setGraphStatus: (s: string) => void;
  onSearchResult: (text: string | null) => void;
}) {
  const [status, setStatus] = useState("");
  const [busy, setBusy] = useState(false);
  const [text, setText] = useState("");
  const [searchQuery, setSearchQuery] = useState("");
  const [dataRefreshToken, setDataRefreshToken] = useState(0);

  const handleSearch = async () => {
    const q = searchQuery.trim();
    if (!q) return;
    setBusy(true);
    setStatus("");
    onSearchResult("Searching...");
    try {
      const args: Record<string, string> = {
        search_query: q,
        search_type: "GRAPH_COMPLETION",
      };
      if (selectedDataset) args.datasets = selectedDataset;
      const result = await app.callServerTool({ name: "search", arguments: args });
      const msg = textOf(result) ?? JSON.stringify(result.content);
      onSearchResult(result.isError ? `Error: ${msg}` : msg);
    } catch (err) {
      onSearchResult(`Error: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setBusy(false);
    }
  };

  const handleAddText = async () => {
    const data = text.trim();
    if (!data) return;
    setBusy(true);
    setStatus("Adding text...");
    try {
      const args: Record<string, string> = { data };
      if (selectedDataset) args.dataset_name = selectedDataset;
      const result = await app.callServerTool({ name: "cognify", arguments: args });
      const msg = textOf(result) ?? JSON.stringify(result.content);
      setStatus(result.isError ? `Error: ${msg}` : msg);
      if (!result.isError) setText("");
    } catch (err) {
      setStatus(`Error: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setBusy(false);
    }
  };

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
      const args: Record<string, string> = { filename: file.name, content_base64 };
      if (selectedDataset) args.dataset_name = selectedDataset;
      const result = await app.callServerTool({
        name: "cognify_file",
        arguments: args,
      });
      const text = textOf(result) ?? JSON.stringify(result.content);
      setStatus(result.isError ? `Error: ${text}` : text);
      if (!result.isError) setDataRefreshToken((t) => t + 1);
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

  const rowStyle: React.CSSProperties = {
    display: "flex",
    gap: 12,
    alignItems: "center",
  };

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: 6,
        padding: "8px 12px",
        borderBottom: "1px solid #e5e7eb",
        fontSize: "0.875rem",
      }}
    >
      <div style={rowStyle}>
        <label style={{ fontWeight: 500 }}>Dataset:</label>
        <DatasetPicker
          app={app}
          datasets={datasets}
          selectedDataset={selectedDataset}
          disabled={busy}
          dataRefreshToken={dataRefreshToken}
          onSelectDataset={onSelectDataset}
          onDatasetsChanged={onDatasetsChanged}
          setStatus={setStatus}
          setBusy={setBusy}
        />
        <button type="button" onClick={handleRefresh} disabled={busy}>
          Refresh graph
        </button>
      </div>
      <div style={rowStyle}>
        <label style={{ fontWeight: 500 }}>Upload file:</label>
        <input type="file" onChange={handleFile} disabled={busy} />
        <label style={{ fontWeight: 500 }}>Add text:</label>
        <input
          type="text"
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") void handleAddText();
          }}
          disabled={busy}
          placeholder="Paste or type text..."
          style={{ flex: 1, minWidth: 140 }}
        />
        <button type="button" onClick={handleAddText} disabled={busy || !text.trim()}>
          Add
        </button>
      </div>
      <div style={rowStyle}>
        <label style={{ fontWeight: 500 }}>Search:</label>
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") void handleSearch();
          }}
          disabled={busy}
          placeholder="Ask a question..."
          style={{ flex: 1, minWidth: 140 }}
        />
        <button type="button" onClick={handleSearch} disabled={busy || !searchQuery.trim()}>
          Go
        </button>
      </div>
      {status && <div style={{ color: "#4b5563", fontSize: "0.8rem" }}>{status}</div>}
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
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<string | null>(null);
  const [searchResult, setSearchResult] = useState<string | null>(null);
  const [clientName, setClientName] = useState<string>("");
  const [agentDefaultDataset, setAgentDefaultDataset] = useState<string | null>(null);

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

  const reloadDatasets = async (preferredOverride?: string | null) => {
    if (!app) return;
    try {
      const result = await app.callServerTool({ name: "list_datasets_json", arguments: {} });
      if (result.isError) return;
      const list = datasetsOf(result);
      setDatasets(list);
      setSelectedDataset((prev) => {
        if (prev && list.some((d) => d.name === prev)) return prev;
        if (list.length === 0) return null;
        const preferred =
          preferredOverride !== undefined ? preferredOverride : agentDefaultDataset;
        const match =
          (preferred ? list.find((d) => d.name === preferred) : undefined) ??
          list.find((d) => d.name === "main_dataset") ??
          list[0];
        return match.name;
      });
    } catch {
      /* ignore; selector just stays empty */
    }
  };

  useEffect(() => {
    if (!app) return;
    (async () => {
      let preferred: string | null = null;
      try {
        const info = await app.callServerTool({
          name: "get_client_info_json",
          arguments: {},
        });
        if (!info.isError) {
          const s = info.structuredContent as
            | { client?: { name?: string }; default_dataset?: string }
            | undefined;
          if (s?.client?.name) setClientName(s.client.name);
          if (s?.default_dataset) {
            setAgentDefaultDataset(s.default_dataset);
            preferred = s.default_dataset;
          }
        }
      } catch {
        /* ignore; header just stays empty */
      }
      await reloadDatasets(preferred);
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
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
      {clientName && (
        <div
          style={{
            padding: "4px 12px",
            background: "#f3f4f6",
            borderBottom: "1px solid #e5e7eb",
            fontSize: "0.75rem",
            color: "#4b5563",
          }}
        >
          Agent: <strong>{clientName}</strong>
          {agentDefaultDataset && (
            <>
              {" · Default dataset: "}
              <strong>{agentDefaultDataset}</strong>
            </>
          )}
        </div>
      )}
      <Toolbar
        app={app}
        datasets={datasets}
        selectedDataset={selectedDataset}
        onSelectDataset={setSelectedDataset}
        onDatasetsChanged={reloadDatasets}
        onGraphHtml={setGraphHtml}
        setGraphStatus={setGraphStatus}
        onSearchResult={setSearchResult}
      />
      {searchResult !== null && (
        <div
          style={{
            maxHeight: 180,
            overflowY: "auto",
            padding: "8px 12px",
            background: "#f9fafb",
            borderBottom: "1px solid #e5e7eb",
            fontSize: "0.85rem",
            position: "relative",
          }}
        >
          <button
            type="button"
            onClick={() => setSearchResult(null)}
            title="Close search results"
            style={{
              position: "absolute",
              top: 4,
              right: 8,
              border: "none",
              background: "transparent",
              cursor: "pointer",
              color: "#6b7280",
              fontSize: "1rem",
            }}
          >
            ✕
          </button>
          <div style={{ fontWeight: 600, marginBottom: 4 }}>Search result</div>
          <div style={{ whiteSpace: "pre-wrap", color: "#1f2937" }}>{searchResult}</div>
        </div>
      )}
      <GraphPane html={graphHtml} status={graphStatus} />
    </div>
  );
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <WorkspaceApp />
  </StrictMode>,
);
