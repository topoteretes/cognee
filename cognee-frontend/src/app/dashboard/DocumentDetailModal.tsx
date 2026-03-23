"use client";

import { useEffect, useState } from "react";
import { fetch } from "@/utils";
import { Modal, IconButton, GhostButton } from "@/ui/elements";
import { CloseIcon } from "@/ui/Icons";
import { DataFile } from "@/modules/ingestion/useData";
import { getDocStatus, formatSize, formatTokens } from "@/utils/documentHelpers";

interface DocumentNode {
  id: string;
  type: string;
  label: string;
  text?: string;
  chunk_index?: number;
  chunk_size?: number;
  cut_type?: string;
  attributes?: Record<string, unknown>;
}

interface DocumentNodes {
  data_id: string;
  data_name: string;
  chunks: DocumentNode[];
  entities: DocumentNode[];
  summaries: DocumentNode[];
}

interface DocumentDetailModalProps {
  isOpen: boolean;
  onClose: () => void;
  dataFile: DataFile;
  datasetId: string;
  useCloud?: boolean;
}

type Tab = "overview" | "chunks" | "entities";

function formatDate(iso?: string): string {
  if (!iso) return "—";
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

export default function DocumentDetailModal({
  isOpen,
  onClose,
  dataFile,
  datasetId,
  useCloud = false,
}: DocumentDetailModalProps) {
  const [activeTab, setActiveTab] = useState<Tab>("overview");
  const [nodes, setNodes] = useState<DocumentNodes | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expandedChunks, setExpandedChunks] = useState<Set<number>>(new Set());

  useEffect(() => {
    if (!isOpen) {
      setNodes(null);
      setError(null);
      setActiveTab("overview");
      setExpandedChunks(new Set());
      return;
    }

    setLoading(true);
    setError(null);

    fetch(
      `/v1/datasets/${datasetId}/document-nodes/${dataFile.id}`,
      {},
      useCloud
    )
      .then((res) => {
        if (!res.ok) {
          throw new Error(`HTTP ${res.status}: ${res.statusText}`);
        }
        return res.json();
      })
      .then((data: DocumentNodes) => {
        setNodes(data);
      })
      .catch((err: unknown) => {
        const message =
          err instanceof Error ? err.message : "Failed to load document details";
        setError(message);
      })
      .finally(() => setLoading(false));
  }, [isOpen, datasetId, dataFile.id, useCloud]);

  const toggleChunk = (index: number) => {
    setExpandedChunks((prev) => {
      const next = new Set(prev);
      if (next.has(index)) {
        next.delete(index);
      } else {
        next.add(index);
      }
      return next;
    });
  };

  const status = getDocStatus(dataFile.pipeline_status);
  const statusLabel =
    status === "completed"
      ? "Processed"
      : status === "processing"
      ? "Processing"
      : "Pending";
  const statusColor =
    status === "completed"
      ? "text-green-600"
      : status === "processing"
      ? "text-orange-500"
      : "text-gray-400";

  const tabs: { id: Tab; label: string; count?: number }[] = [
    { id: "overview", label: "Overview" },
    {
      id: "chunks",
      label: "Chunks",
      count: nodes?.chunks?.length,
    },
    {
      id: "entities",
      label: "Entities",
      count: nodes?.entities?.length,
    },
  ];

  return (
    <Modal isOpen={isOpen}>
      <div className="w-full max-w-3xl max-h-[85vh] flex flex-col bg-white rounded-xl shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="flex flex-row items-center justify-between px-6 py-4 border-b border-gray-100">
          <div className="flex flex-col gap-0.5 overflow-hidden">
            <span className="text-lg font-semibold truncate">{dataFile.name}</span>
            <span className={`text-xs font-medium ${statusColor}`}>
              {statusLabel}
            </span>
          </div>
          <IconButton onClick={onClose}>
            <CloseIcon />
          </IconButton>
        </div>

        {/* Tabs */}
        <div className="flex flex-row border-b border-gray-100 px-6" role="tablist">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              role="tab"
              aria-selected={activeTab === tab.id}
              aria-controls={`tabpanel-${tab.id}`}
              id={`tab-${tab.id}`}
              className={`py-3 px-4 text-sm font-medium border-b-2 transition-colors ${
                activeTab === tab.id
                  ? "border-indigo-600 text-indigo-600"
                  : "border-transparent text-gray-500 hover:text-gray-700"
              }`}
            >
              {tab.label}
              {tab.count !== undefined && (
                <span className="ml-1.5 text-xs bg-gray-100 text-gray-500 rounded-full px-1.5 py-0.5">
                  {tab.count}
                </span>
              )}
            </button>
          ))}
        </div>

        {/* Body */}
        <div className="flex-1 overflow-y-auto px-6 py-4" role="tabpanel" id={`tabpanel-${activeTab}`} aria-labelledby={`tab-${activeTab}`}>
          {loading && (
            <div className="flex items-center justify-center py-12 text-gray-400 text-sm">
              Loading...
            </div>
          )}

          {error && (
            <div className="text-red-500 text-sm py-4">{error}</div>
          )}

          {!loading && !error && activeTab === "overview" && (
            <div className="flex flex-col gap-4">
              {/* Metadata grid */}
              <div className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm">
                <div>
                  <span className="text-gray-400 text-xs uppercase tracking-wide">Name</span>
                  <p className="text-gray-800 mt-0.5 break-all">{dataFile.name}</p>
                </div>
                <div>
                  <span className="text-gray-400 text-xs uppercase tracking-wide">Extension</span>
                  <p className="text-gray-800 mt-0.5">{dataFile.extension || "—"}</p>
                </div>
                <div>
                  <span className="text-gray-400 text-xs uppercase tracking-wide">Size</span>
                  <p className="text-gray-800 mt-0.5">{formatSize(dataFile.data_size)}</p>
                </div>
                <div>
                  <span className="text-gray-400 text-xs uppercase tracking-wide">Tokens</span>
                  <p className="text-gray-800 mt-0.5">{formatTokens(dataFile.token_count)}</p>
                </div>
                <div>
                  <span className="text-gray-400 text-xs uppercase tracking-wide">Status</span>
                  <p className={`mt-0.5 font-medium ${statusColor}`}>{statusLabel}</p>
                </div>
                <div>
                  <span className="text-gray-400 text-xs uppercase tracking-wide">Created</span>
                  <p className="text-gray-800 mt-0.5">{formatDate(dataFile.created_at)}</p>
                </div>
              </div>

              {/* Pipeline status detail */}
              {dataFile.pipeline_status && Object.keys(dataFile.pipeline_status).length > 0 && (
                <div>
                  <span className="text-gray-400 text-xs uppercase tracking-wide">Pipeline stages</span>
                  <div className="mt-1 flex flex-col gap-1">
                    {Object.entries(dataFile.pipeline_status).map(([pipeline, stages]) => (
                      <div key={pipeline} className="flex flex-col gap-0.5">
                        {Object.entries(stages).map(([stageId, stageStatus]) => {
                          const isOk = stageStatus === "DATA_ITEM_PROCESSING_COMPLETED";
                          return (
                            <div key={stageId} className="flex flex-row items-center gap-2 text-xs">
                              <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${isOk ? "bg-green-500" : "bg-orange-400"}`} />
                              <span className="text-gray-600 font-mono">{pipeline}</span>
                              <span className={isOk ? "text-green-600" : "text-orange-500"}>
                                {stageStatus.replace("DATA_ITEM_PROCESSING_", "").toLowerCase()}
                              </span>
                            </div>
                          );
                        })}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Summary */}
              {nodes?.summaries && nodes.summaries.length > 0 && nodes.summaries[0]?.text && (
                <div>
                  <span className="text-gray-400 text-xs uppercase tracking-wide">Summary</span>
                  <p className="mt-1 text-sm text-gray-700 leading-relaxed whitespace-pre-wrap border-l-2 border-indigo-200 pl-3">
                    {nodes.summaries[0].text}
                  </p>
                </div>
              )}
            </div>
          )}

          {!loading && !error && activeTab === "chunks" && (
            <div className="flex flex-col gap-2">
              {!nodes?.chunks || nodes.chunks.length === 0 ? (
                <div className="text-gray-400 text-sm py-4">No chunks available.</div>
              ) : (
                nodes.chunks
                  .slice()
                  .sort((a, b) => (a.chunk_index ?? 0) - (b.chunk_index ?? 0))
                  .map((chunk, idx) => {
                    const isExpanded = expandedChunks.has(idx);
                    return (
                      <div
                        key={chunk.id}
                        className="border border-gray-200 rounded-lg overflow-hidden"
                      >
                        <button
                          onClick={() => toggleChunk(idx)}
                          className="w-full flex flex-row items-center justify-between px-4 py-2.5 text-left hover:bg-gray-50 transition-colors"
                        >
                          <div className="flex flex-row items-center gap-3">
                            <span className="text-xs font-mono text-indigo-600 w-8 flex-shrink-0">
                              #{chunk.chunk_index ?? idx}
                            </span>
                            <span className="text-xs text-gray-500">
                              {chunk.chunk_size ?? chunk.text?.length ?? 0} chars
                            </span>
                            {chunk.cut_type && (
                              <span className="text-xs text-gray-400 bg-gray-100 px-1.5 py-0.5 rounded">
                                {chunk.cut_type}
                              </span>
                            )}
                          </div>
                          <span className="text-gray-400 text-xs">{isExpanded ? "▲" : "▼"}</span>
                        </button>
                        {isExpanded && chunk.text && (
                          <div className="px-4 py-3 border-t border-gray-100 bg-gray-50">
                            <pre className="text-xs text-gray-700 whitespace-pre-wrap font-sans leading-relaxed">
                              {chunk.text}
                            </pre>
                          </div>
                        )}
                      </div>
                    );
                  })
              )}
            </div>
          )}

          {!loading && !error && activeTab === "entities" && (
            <div className="flex flex-col gap-2">
              {!nodes?.entities || nodes.entities.length === 0 ? (
                <div className="text-gray-400 text-sm py-4">No entities extracted.</div>
              ) : (
                <div className="flex flex-wrap gap-2">
                  {nodes.entities.map((entity) => (
                    <div
                      key={entity.id}
                      className="flex flex-row items-center gap-1.5 border border-indigo-200 rounded-full px-3 py-1.5 bg-indigo-50"
                    >
                      <span className="text-xs text-indigo-700 font-medium">{entity.label}</span>
                      {entity.type && (
                        <span className="text-xs text-indigo-400">· {entity.type}</span>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-3 border-t border-gray-100 flex justify-end">
          <GhostButton onClick={onClose}>close</GhostButton>
        </div>
      </div>
    </Modal>
  );
}
