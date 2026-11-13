import type { DatasetProcessingStatus } from "@/modules/datasets/pollDatasetStatus";
import type { DatasetProcessing } from "@/modules/datasets/useDatasetProcessing";
import { INSUFFICIENT_CREDITS_REASON } from "@/modules/datasets/datasetStatusDetail";
import type { BrainUploadStage } from "@/modules/ingestion/useBrainUpload";
import type { UploadProgress } from "@/modules/ingestion/uploadProgress";
import type { PendingLowBalanceWarning } from "@/modules/billing/useLowBalanceUploadWarning";
import type { CreateBrainTemplateKey } from "./createBrainTemplates";

export interface DatasetRaw {
  id: string;
  name: string;
  createdAt?: string;
}

export interface FileEntry {
  /** Stored processing outcome; undefined while it is not known. */
  completed?: boolean;
  id: string;
  name: string;
  label?: string;
  extension?: string;
  size?: number;
  createdAt?: string;
}

export type DisplayStatus =
  | "pending"
  | "running"
  | "completed"
  | "failed"
  | "failed_insufficient_credits"
  | "empty"
  | "loading";

/** Id prefix of a dataset the pod has been asked to create but has not answered for yet. */
export const PENDING_DATASET_PREFIX = "pending-";

export function isPendingDataset(id: string): boolean {
  return id.startsWith(PENDING_DATASET_PREFIX);
}

export interface Dataset extends DatasetRaw {
  // -1 means the count fetch hasn't resolved yet (BrainList renders a
  // skeleton). null means it resolved but couldn't produce an exact number
  // (getDatasetDataCount got capped — see its docs) — a distinct, terminal
  // state that must not be confused with "still loading".
  documents: number | null;
  status: DisplayStatus;
  /** Only the selected dataset has it: item completion is read per dataset. */
  processingCounts?: DatasetProcessing;
}

/** A pending placeholder and the dataset the pod eventually returns for it
 *  share a name before they ever share an id (the placeholder's id is only
 *  ever a local `pending-*` stand-in) — so once any dataset in the list
 *  carries a placeholder's name for real, the placeholder has served its
 *  purpose and must go, whichever side (a reload, or the create resolving)
 *  noticed first. Leaving both would duplicate the row; always dropping the
 *  placeholder side (never the real one) means a reload that already knows
 *  the real document count and status is never regressed back to the
 *  placeholder's bare guess. */
export function dropResolvedPlaceholders(datasets: readonly Dataset[]): Dataset[] {
  const resolvedNames = new Set(datasets.filter((d) => !isPendingDataset(d.id)).map((d) => d.name));
  return datasets.filter((d) => !isPendingDataset(d.id) || !resolvedNames.has(d.name));
}

/** Keeps the first row for each id, so a dataset that reached the list twice
 *  (a reload and a create's own update learning about the same dataset) never
 *  renders as two rows under the same React key. First, not last: when two
 *  rows share an id, the earlier one already underwent `dropResolvedPlaceholders`
 *  and is at least as informative as a freshly-created stub. */
export function dedupeDatasetsById(datasets: readonly Dataset[]): Dataset[] {
  const seen = new Set<string>();
  const deduped: Dataset[] = [];
  for (const dataset of datasets) {
    if (seen.has(dataset.id)) continue;
    seen.add(dataset.id);
    deduped.push(dataset);
  }
  return deduped;
}

export function mapProcessingStatus(
  raw: DatasetProcessingStatus | undefined,
  docCount: number | null,
  reason: string | null = null,
): DisplayStatus {
  // Neither a still-loading count (-1) nor an unknown one (null) is evidence
  // of an empty dataset (see Dataset.documents) — treat both the same as
  // "has documents" rather than guess wrong.
  const hasDocuments = docCount !== 0;
  if (!raw) return hasDocuments ? "completed" : "empty";
  if (raw === "DATASET_PROCESSING_COMPLETED") return "completed";
  if (raw === "DATASET_PROCESSING_ERRORED") {
    return reason === INSUFFICIENT_CREDITS_REASON ? "failed_insufficient_credits" : "failed";
  }
  if (raw === "DATASET_PROCESSING_STARTED") return "running";
  if (raw === "DATASET_PROCESSING_INITIATED") return "pending";
  return hasDocuments ? "completed" : "empty";
}

export interface UseBrainsDataResult {
  processingCounts?: DatasetProcessing;
  processingCountsError: boolean;
  refreshProcessingCounts: () => void;
  isLoading: boolean;
  datasets: Dataset[];
  datasetsError: boolean;
  /** Ids this session itself created and the pod has confirmed — distinct
   *  from a dataset merely missing from the access list, which can mean it's
   *  shared and the list just hasn't loaded yet (see withUnclaimedAsPersonal). */
  sessionCreatedIds: ReadonlySet<string>;
  selectedId: string | null;
  selectedDataset: Dataset | null;
  selectedDocs: FileEntry[];
  docsLoading: boolean;
  docsError: boolean;
  hasMoreDocs: boolean;
  loadMoreDocs: () => void;
  retryDocs: () => void;
  outdatedDatasets: Set<string>;
  refreshing: boolean;
  isUploading: boolean;
  uploadStage: BrainUploadStage;
  uploadProgress: UploadProgress;
  uploadError: string | null;
  canRetryBuild: boolean;
  setUploadError: (error: string | null) => void;
  pendingLowBalanceWarning: PendingLowBalanceWarning | null;
  cancelLowBalanceUpload: () => void;
  showCreateModal: boolean;
  setShowCreate: (show: boolean) => void;
  newName: string;
  setNewName: (name: string) => void;
  creating: boolean;
  createError: string;
  setCreateError: (error: string) => void;
  deleteTarget: Dataset | null;
  setDeleteTarget: (target: Dataset | null) => void;
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
  handleDelete: (ds: Dataset) => void;
  /** Resolves to whether the dataset was actually created (and selected),
   *  so the page can react to a real creation and not a rejected name. */
  handleCreate: (templateKey: CreateBrainTemplateKey | null) => Promise<boolean>;
  handlePasteText: () => Promise<void>;
}
