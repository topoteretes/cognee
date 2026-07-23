"use client";

import { TrackPageView, trackEvent } from "@/modules/analytics";
import ShareDatasetModal from "@/ui/elements/ShareDatasetModal";
import { decodeFilename } from "@/utils/fileFormat";
import PlusIcon from "@/ui/elements/PlusIcon";
import EmptyDocIcon from "@/ui/elements/EmptyDocIcon";
import PageLoading from "@/ui/elements/PageLoading";
import DeleteConfirmModal from "@/ui/elements/DeleteConfirmModal";
import CreateBrainModal from "./partials/CreateBrainModal";
import PasteTextModal from "./partials/PasteTextModal";
import BrainList from "./partials/BrainList";
import DocumentsPanel from "./partials/DocumentsPanel";
import { useBrainsData } from "./useBrainsData";

export default function DatasetsPage() {
  const {
    isLoading,
    datasets,
    datasetsError,
    selectedId,
    selectedDataset,
    selectedDocs,
    docsLoading,
    docsError,
    retryDocs,
    outdatedDatasets,
    refreshing,
    isUploading,
    uploadStage,
    uploadError,
    canRetryBuild,
    setUploadError,
    showCreateModal,
    setShowCreate,
    newName,
    setNewName,
    creating,
    createError,
    setCreateError,
    deleteTarget,
    setDeleteTarget,
    shareTarget,
    setShareTarget,
    deletingId,
    deleteDocTarget,
    setDeleteDocTarget,
    deletingDocId,
    showPasteModal,
    setShowPasteModal,
    pasteText,
    setPasteText,
    pasting,
    handleRefresh,
    handleSelectDataset,
    handleUploadFiles,
    handleRetryBuild,
    handleDeleteFile,
    handleDelete,
    handleCreate,
    handlePasteText,
  } = useBrainsData();

  if (isLoading) {
    return (
      <><TrackPageView page="Brains" /><PageLoading name="Brain" /></>
    );
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", flex: 1, minHeight: 0, overflow: "hidden" }}>
      <TrackPageView page="Brains" />

      {/* ── Modals ── */}
      {showCreateModal && (
        <CreateBrainModal
          value={newName}
          error={createError}
          creating={creating}
          onChange={setNewName}
          onSubmit={handleCreate}
          onCancel={() => { setShowCreate(false); setNewName(""); setCreateError(""); }}
        />
      )}

      {deleteDocTarget && (
        <DeleteConfirmModal
          title="Delete document"
          message={<>Are you sure you want to delete <strong>{decodeFilename(deleteDocTarget.name)}</strong>? This action cannot be undone.</>}
          onConfirm={() => handleDeleteFile(deleteDocTarget.id)}
          onCancel={() => setDeleteDocTarget(null)}
          busy={deletingDocId === deleteDocTarget.id}
        />
      )}

      {showPasteModal && (
        <PasteTextModal
          value={pasteText}
          pasting={pasting}
          onChange={setPasteText}
          onSubmit={handlePasteText}
          onCancel={() => { setShowPasteModal(false); setPasteText(""); }}
        />
      )}

      {deleteTarget && (
        <DeleteConfirmModal
          title="Delete brain"
          message={<>Are you sure you want to delete <strong>{deleteTarget.name}</strong>? This will permanently remove the dataset and all its files.</>}
          onConfirm={() => handleDelete(deleteTarget)}
          onCancel={() => setDeleteTarget(null)}
          busy={deletingId === deleteTarget.id}
        />
      )}

      {/* ── Share brain modal ── */}
      {shareTarget && (
        <ShareDatasetModal
          datasetId={shareTarget.id}
          datasetName={shareTarget.name}
          pageName="Brains"
          onClose={() => setShareTarget(null)}
        />
      )}

      {/* ── Header ── */}
      <div style={{ padding: "24px 32px 16px", display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexShrink: 0 }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          <h1 style={{ fontSize: 20, fontWeight: 300, color: "#EDECEA", margin: 0, fontFamily: '"TWKLausanne", sans-serif' }}>Brain</h1>
          <p style={{ fontSize: 14, color: "rgba(237,236,234,0.55)", margin: 0 }}>Upload documents to build searchable knowledge graphs.</p>
        </div>
        <button onClick={handleRefresh} disabled={refreshing}
          className="hover:bg-white/10 cursor-pointer"
          style={{ background: "rgba(255,255,255,0.06)", color: "rgba(237,236,234,0.7)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 8, padding: "8px 12px", fontSize: 13, fontWeight: 500, display: "flex", alignItems: "center", gap: 4 }}
          title="Refresh">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="rgba(237,236,234,0.7)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
            style={refreshing ? { animation: "spin 1s linear infinite" } : undefined}>
            <path d="M21 2v6h-6" /><path d="M3 12a9 9 0 0115.36-6.36L21 8" /><path d="M3 22v-6h6" /><path d="M21 12a9 9 0 01-15.36 6.36L3 16" />
          </svg>
        </button>
      </div>

      {/* ── Finder body ── */}
      {datasets.length > 0 ? (
        <div style={{ flex: 1, minHeight: 0, display: "flex", overflow: "hidden", marginInline: 32, marginBottom: 32, border: "1px solid rgba(255,255,255,0.12)", borderRadius: 12, background: "rgba(0,0,0,0.82)", backdropFilter: "blur(20px)" }}>

          {/* Column 1 — Datasets */}
          <BrainList
            brains={datasets}
            selectedId={selectedId}
            outdatedIds={outdatedDatasets}
            onSelect={handleSelectDataset}
            onCreate={() => { trackEvent({ pageName: "Brains", eventName: "dataset_create_modal_opened" }); setShowCreate(true); }}
            onDelete={(ds) => setDeleteTarget(ds)}
          />

          {/* Column 2 — Documents */}
          <DocumentsPanel
            selectedId={selectedId}
            selectedName={selectedDataset?.name ?? null}
            docsLoading={docsLoading}
            docsError={docsError}
            docs={selectedDocs}
            isUploading={isUploading}
            uploadStage={uploadStage}
            uploadError={uploadError}
            canRetryBuild={canRetryBuild}
            onUpload={handleUploadFiles}
            onPaste={() => setShowPasteModal(true)}
            onDeleteDoc={setDeleteDocTarget}
            onClearUploadError={() => setUploadError(null)}
            onRetryBuild={handleRetryBuild}
            onRetryDocs={retryDocs}
          />

        </div>
      ) : datasetsError ? (
        /* ── Load-error state — never render "no brains" for a failed fetch ── */
        <div style={{ flex: 1, display: "flex", flexDirection: "column", paddingInline: 32, paddingBottom: 32 }}>
          <div style={{ flex: 1, background: "rgba(255,255,255,0.06)", backdropFilter: "blur(12px)", border: "1px solid rgba(239,68,68,0.3)", borderRadius: 12, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 10, padding: 48 }}>
            <span style={{ fontSize: 16, fontWeight: 700, color: "#F87171" }}>Couldn&rsquo;t load your brains</span>
            <p style={{ fontSize: 14, color: "rgba(237,236,234,0.35)", margin: 0, maxWidth: 340, textAlign: "center" }}>
              Your brains are safe — we just couldn&rsquo;t reach the server. This can happen while a large upload is still processing.
            </p>
            <button onClick={handleRefresh} disabled={refreshing}
              className="cursor-pointer hover:bg-white/10"
              style={{ background: "rgba(255,255,255,0.06)", color: "#EDECEA", border: "1px solid rgba(255,255,255,0.15)", borderRadius: 8, padding: "8px 20px", fontSize: 14, fontWeight: 500, marginTop: 8 }}>
              {refreshing ? "Retrying…" : "Retry"}
            </button>
          </div>
        </div>
      ) : (
        /* ── Empty state ── */
        <div style={{ flex: 1, display: "flex", flexDirection: "column", paddingInline: 32, paddingBottom: 32 }}>
          <div style={{ flex: 1, background: "rgba(255,255,255,0.06)", backdropFilter: "blur(12px)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 12, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 16, padding: 48 }}>
            <div style={{ width: 56, height: 56, background: "rgba(188,155,255,0.20)", border: "1px solid rgba(188,155,255,0.35)", borderRadius: 12, display: "flex", alignItems: "center", justifyContent: "center" }}>
              <EmptyDocIcon />
            </div>
            <span style={{ fontSize: 16, fontWeight: 700, color: "#EDECEA" }}>No brains yet</span>
            <p style={{ fontSize: 14, color: "rgba(237,236,234,0.35)", margin: 0, maxWidth: 340, textAlign: "center" }}>
              A brain turns the documents you upload into a searchable knowledge graph — Cognee extracts entities and relationships so you can query them later. Create your first one to get started.
            </p>
            <button onClick={() => { trackEvent({ pageName: "Brains", eventName: "dataset_create_modal_opened" }); setShowCreate(true); }}
              className="hover:bg-[#5A0ED6] cursor-pointer"
              style={{ background: "#6510F4", color: "#fff", border: "none", borderRadius: 8, padding: "8px 20px", fontSize: 14, fontWeight: 500, display: "flex", alignItems: "center", gap: 6, marginTop: 12 }}>
              <PlusIcon /> Create brain
            </button>
          </div>
        </div>
      )}

      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}
