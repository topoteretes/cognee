import type { DatasetProcessingStatus } from "@/modules/datasets/pollDatasetStatus";
import type { BrainUploadStage } from "@/modules/ingestion/useBrainUpload";
import type { CreateBrainTemplateKey } from "./createBrainTemplates";

export interface DatasetRaw {
  id: string;
  name: string;
  createdAt?: string;
}

export interface FileEntry {
  id: string;
  name: string;
  extension?: string;
  size?: number;
  createdAt?: string;
}

export type DisplayStatus = "pending" | "running" | "completed" | "failed" | "empty" | "loading";

export interface Dataset extends DatasetRaw {
  documents: number;
  status: DisplayStatus;
}

export function mapProcessingStatus(raw: DatasetProcessingStatus | undefined, docCount: number): DisplayStatus {
  if (!raw) return docCount > 0 ? "completed" : "empty";
  if (raw === "DATASET_PROCESSING_COMPLETED") return "completed";
  if (raw === "DATASET_PROCESSING_ERRORED") return "failed";
  if (raw === "DATASET_PROCESSING_STARTED") return "running";
  if (raw === "DATASET_PROCESSING_INITIATED") return "pending";
  return docCount > 0 ? "completed" : "empty";
}

export interface UseBrainsDataResult {
  isLoading: boolean;
  datasets: Dataset[];
  datasetsError: boolean;
  selectedId: string | null;
  selectedDataset: Dataset | null;
  selectedDocs: FileEntry[];
  docsLoading: boolean;
  docsError: boolean;
  retryDocs: () => void;
  outdatedDatasets: Set<string>;
  refreshing: boolean;
  isUploading: boolean;
  uploadStage: BrainUploadStage;
  uploadError: string | null;
  canRetryBuild: boolean;
  setUploadError: (error: string | null) => void;
  showCreateModal: boolean;
  setShowCreate: (show: boolean) => void;
  newName: string;
  setNewName: (name: string) => void;
  creating: boolean;
  createError: string;
  setCreateError: (error: string) => void;
  deleteTarget: Dataset | null;
  setDeleteTarget: (target: Dataset | null) => void;
  shareTarget: Dataset | null;
  setShareTarget: (target: Dataset | null) => void;
  deletingId: string | null;
  deleteDocTarget: FileEntry | null;
  setDeleteDocTarget: (target: FileEntry | null) => void;
  deletingDocId: string | null;
  showPasteModal: boolean;
  setShowPasteModal: (show: boolean) => void;
  pasteText: string;
  setPasteText: (text: string) => void;
  pasting: boolean;
  handleRefresh: () => Promise<void>;
  handleSelectDataset: (id: string) => Promise<void>;
  handleUploadFiles: (files: File[]) => Promise<void>;
  handleRetryBuild: () => Promise<void>;
  handleDeleteFile: (docId: string) => Promise<void>;
  handleDelete: (ds: Dataset) => Promise<void>;
  handleCreate: (templateKey: CreateBrainTemplateKey | null) => Promise<void>;
  handlePasteText: () => Promise<void>;
}
