"use client";

import { useState } from "react";
import { TrackPageView, trackEvent } from "@/modules/analytics";
import { useTenant } from "@/modules/tenant/TenantProvider";
import ShareAccessModal from "./partials/ShareAccessModal";
import LowBalanceWarningModal from "@/ui/elements/LowBalanceWarningModal";
import { decodeFilename } from "@/utils/fileFormat";
import isCloudEnvironment from "@/utils/isCloudEnvironment";
import PlusIcon from "@/ui/elements/PlusIcon";
import EmptyDocIcon from "@/ui/elements/EmptyDocIcon";
import PageLoading from "@/ui/elements/PageLoading";
import DeleteConfirmModal from "@/ui/elements/DeleteConfirmModal";
import CreateBrainModal from "./partials/CreateBrainModal";
import PasteTextModal from "./partials/PasteTextModal";
import BrainList from "./partials/BrainList";
import DatasetListFilters from "./partials/DatasetListFilters";
import DocumentsPanel from "./partials/DocumentsPanel";
import GroupRail from "./partials/GroupRail";
import { PERSONAL_FACET_ID } from "./partials/datasetFacets";
import useDatasetFinderFilters from "./partials/useDatasetFinderFilters";
import { useBrainsData } from "./useBrainsData";

export default function DatasetsPage() {
  const showGroups = isCloudEnvironment();
  const {
    processingCounts,
    processingCountsError,
    refreshProcessingCounts,
    isLoading,
    datasets,
    datasetsError,
    sessionCreatedIds,
    selectedId,
    selectedDataset,
    selectedDocs,
    docsLoading,
    docsError,
    hasMoreDocs,
    loadMoreDocs,
    retryDocs,
    outdatedDatasets,
    refreshing,
    isUploading,
    uploadStage,
    uploadProgress,
    uploadError,
    canRetryBuild,
    setUploadError,
    pendingLowBalanceWarning,
    cancelLowBalanceUpload,
    showCreateModal,
    setShowCreate,
    newName,
    setNewName,
    creating,
    createError,
    setCreateError,
    deleteTarget,
    setDeleteTarget,
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
  const finder = useDatasetFinderFilters(datasets, outdatedDatasets, sessionCreatedIds);
  const { tenant } = useTenant();
  const [sharing, setSharing] = useState(false);
  // Sharing needs `share` on the dataset; the pod enforces it, the button just does not offer it otherwise.
  const selectedAccess = selectedId ? finder.accessFor(selectedId) : undefined;
  const canShare = showGroups && Boolean(selectedAccess?.permissions.includes("share"));

  if (isLoading) {
    return (
      <><TrackPageView page="Brains" /><PageLoading name="Datasets" /></>
    );
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", flex: 1, minHeight: 0, overflow: "hidden" }}>
      <TrackPageView page="Brains" />

      {/* ── Modals ── */}
      <LowBalanceWarningModal
        warning={pendingLowBalanceWarning}
        onCancel={cancelLowBalanceUpload}
      />

      {showCreateModal && (
        <CreateBrainModal
          value={newName}
          error={createError}
          creating={creating}
          onChange={setNewName}
          onSubmit={async (templateKey) => {
            // A dataset is always created under Personal (handleCreate never
            // shares it), so land the view there too — otherwise, viewed from
            // Shared or a group, the new row (and the selection that follows
            // it) would be invisible in the very list the user is looking at.
            if (await handleCreate(templateKey) && showGroups) finder.selectFacet(PERSONAL_FACET_ID);
          }}
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
          title="Delete dataset"
          message={<>Are you sure you want to delete <strong>{deleteTarget.name}</strong>? This will permanently remove the dataset and all its files.</>}
          onConfirm={() => handleDelete(deleteTarget)}
          onCancel={() => setDeleteTarget(null)}
        />
      )}

      {/* ── Share dataset modal ── */}
      {sharing && selectedDataset && selectedAccess && (
        <ShareAccessModal
          datasetId={selectedDataset.id}
          datasetName={selectedDataset.name}
          access={selectedAccess}
          tenantId={tenant?.tenant_id ?? null}
          onClose={() => setSharing(false)}
          onChanged={() => void finder.refreshAccess()}
        />
      )}

      {/* ── Header ── */}
      <div style={{ padding: "24px 32px 16px", display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexShrink: 0 }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          <h1 style={{ fontSize: 20, fontWeight: 300, color: "#EDECEA", margin: 0, fontFamily: '"TWKLausanne", sans-serif' }}>Datasets</h1>
          <p style={{ fontSize: 14, color: "rgba(237,236,234,0.55)", margin: 0 }}>{showGroups ? "Upload documents to build searchable knowledge graphs and share them with the teams that need them." : "Upload documents to build searchable knowledge graphs."}</p>
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

          {/* Column 1 — Groups: filters over the one dataset list */}
          {showGroups && <GroupRail facets={finder.facets} selectedId={finder.facetId} status={finder.facetsStatus} allCount={datasets.length} onSelect={finder.selectFacet} />}

          {/* Column 2 — Datasets */}
          <BrainList
            brains={finder.visible}
            selectedId={selectedId}
            outdatedIds={outdatedDatasets}
            toolbar={<DatasetListFilters search={finder.search} status={finder.statusFilter} onSearch={finder.setSearch} onStatus={finder.setStatusFilter} />}
            emptyLabel={finder.narrowed ? "No datasets match this filter" : null}
            onSelect={handleSelectDataset}
            onCreate={() => { trackEvent({ pageName: "Brains", eventName: "dataset_create_modal_opened" }); setShowCreate(true); }}
            onDelete={(ds) => setDeleteTarget(ds)}
          />

          {/* Column 3 — Documents */}
          <DocumentsPanel
            processingCounts={processingCounts}
            processingCountsError={processingCountsError}
            processingFailed={selectedDataset?.status === "failed" || selectedDataset?.status === "failed_insufficient_credits"}
            onRefreshProcessing={refreshProcessingCounts}
            selectedId={selectedId}
            selectedName={selectedDataset?.name ?? null}
            ownerLabel={finder.ownerLabel(selectedId)}
            onShare={canShare ? () => setSharing(true) : null}
            docsLoading={docsLoading}
            docsError={docsError}
            docs={selectedDocs}
            hasMore={hasMoreDocs}
            onLoadMore={loadMoreDocs}
            total={selectedDataset?.documents}
            processing={selectedDataset?.status === "pending" || selectedDataset?.status === "running"}
            isUploading={isUploading}
            uploadStage={uploadStage}
            uploadProgress={uploadProgress}
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
        /* ── Load-error state — never render "no datasets" for a failed fetch ── */
        <div style={{ flex: 1, display: "flex", flexDirection: "column", paddingInline: 32, paddingBottom: 32 }}>
          <div style={{ flex: 1, background: "rgba(255,255,255,0.06)", backdropFilter: "blur(12px)", border: "1px solid rgba(239,68,68,0.3)", borderRadius: 12, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 10, padding: 48 }}>
            <span style={{ fontSize: 16, fontWeight: 700, color: "#F87171" }}>Couldn&rsquo;t load your datasets</span>
            <p style={{ fontSize: 14, color: "rgba(237,236,234,0.35)", margin: 0, maxWidth: 340, textAlign: "center" }}>
              Your datasets are safe — we just couldn&rsquo;t reach the server. This can happen while a large upload is still processing.
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
            <span style={{ fontSize: 16, fontWeight: 700, color: "#EDECEA" }}>No datasets yet</span>
            <p style={{ fontSize: 14, color: "rgba(237,236,234,0.35)", margin: 0, maxWidth: 340, textAlign: "center" }}>
              A dataset turns the documents you upload into a searchable knowledge graph — Cognee extracts entities and relationships so you can query them later. Create your first one to get started.
            </p>
            <button onClick={() => { trackEvent({ pageName: "Brains", eventName: "dataset_create_modal_opened" }); setShowCreate(true); }}
              className="hover:bg-[#5A0ED6] cursor-pointer"
              style={{ background: "#6510F4", color: "#fff", border: "none", borderRadius: 8, padding: "8px 20px", fontSize: 14, fontWeight: 500, display: "flex", alignItems: "center", gap: 6, marginTop: 12 }}>
              <PlusIcon /> Create dataset
            </button>
          </div>
        </div>
      )}

      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}
