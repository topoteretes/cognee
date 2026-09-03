import { trackEvent } from "@/modules/analytics";

// Analytics for the brains-list upload flow (useBrainsData). Page-local, not
// a shared module — the "Brains" pageName and event names are specific to
// this page; DatasetDetailPage and the dashboard track their own versions of
// the same lifecycle under their own pageName.

interface UploadTarget {
  datasetId: string;
  fileCount: number;
  totalBytes: number;
}

export function trackUploadStarted(target: UploadTarget & { fileTypes: string[] }): void {
  trackEvent({
    pageName: "Brains",
    eventName: "dataset_upload_started",
    additionalProperties: {
      dataset_id: target.datasetId,
      file_count: String(target.fileCount),
      total_bytes: String(target.totalBytes),
      file_types: target.fileTypes.join(","),
    },
  });
}

export function trackUploadFailed(
  target: UploadTarget & { fileTypes: string[]; durationMs: number; errorName: string; errorMessage: string },
): void {
  trackEvent({
    pageName: "Brains",
    eventName: "dataset_upload_failed",
    additionalProperties: {
      dataset_id: target.datasetId,
      file_count: String(target.fileCount),
      total_bytes: String(target.totalBytes),
      file_types: target.fileTypes.join(","),
      duration_ms: String(target.durationMs),
      error_name: target.errorName,
      error_message: target.errorMessage,
    },
  });
}

export function trackFilesUploaded(target: UploadTarget & { durationMs: number }): void {
  trackEvent({
    pageName: "Brains",
    eventName: "dataset_files_uploaded",
    additionalProperties: {
      dataset_id: target.datasetId,
      file_count: String(target.fileCount),
      total_bytes: String(target.totalBytes),
      duration_ms: String(target.durationMs),
    },
  });
}

export function trackProcessingFailed(target: UploadTarget & { durationMs: number; errorMessage: string }): void {
  trackEvent({
    pageName: "Brains",
    eventName: "dataset_processing_failed",
    additionalProperties: {
      dataset_id: target.datasetId,
      file_count: String(target.fileCount),
      total_bytes: String(target.totalBytes),
      duration_ms: String(target.durationMs),
      error_message: target.errorMessage,
    },
  });
}
