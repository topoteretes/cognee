// "estimating": the pre-upload cost estimate is analyzing the selection (it
// reads/tokenizes PDF/DOCX text, so it can take real time). "uploading": bytes
// are moving. "processing": every file landed and the knowledge-graph build is
// running (the multi-minute part). "resuming": an upload from a previous page
// load was found and is being picked back up. The three real phases the user
// waits through — estimate → upload → build — all report through this one type
// so they render in one place instead of the estimate living in a toast.
export type UploadStage = "idle" | "estimating" | "resuming" | "uploading" | "processing";

export interface UploadProgress {
  stage: UploadStage;
  filesTotal: number;
  // Files the backend has ACCEPTED. Moves a batch at a time, because a batch is
  // one request and the server acknowledges it whole. This is the durable
  // number — error paths and the persisted session use it, and it must never be
  // inflated with files that are merely in flight.
  //
  // Deliberately NOT shown as an "x of n" counter during upload: at ten files
  // per request it advances in visible jumps, which reads as a stuck UI. Bytes
  // are the continuous signal and are what the bar and the label use.
  filesCompleted: number;
  bytesTotal: number;
  bytesSent: number;
  batchesTotal: number;
  batchesCompleted: number;
  // Present once the upload leg finishes and polling begins.
  processingStatus?: string;
  // Files that can never be sent by this session because the browser refused
  // to persist their bytes (storage quota) and the page has since reloaded.
  // Non-zero means the user has to re-select them — the graph build now
  // running covers only what already landed.
  unrecoverableFiles?: number;
}

export const IDLE_PROGRESS: UploadProgress = {
  stage: "idle",
  filesTotal: 0,
  filesCompleted: 0,
  bytesTotal: 0,
  bytesSent: 0,
  batchesTotal: 0,
  batchesCompleted: 0,
};

/** 0-1 for the upload leg; falls back to file counts when sizes are unknown. */
export function uploadFraction(progress: UploadProgress): number {
  if (progress.bytesTotal > 0) {
    return Math.min(1, progress.bytesSent / progress.bytesTotal);
  }
  if (progress.filesTotal > 0) {
    return Math.min(1, progress.filesCompleted / progress.filesTotal);
  }
  return 0;
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB"];
  let value = bytes / 1024;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  return `${value >= 10 || Number.isInteger(value) ? Math.round(value) : value.toFixed(1)} ${units[unitIndex]}`;
}

/**
 * Fold the pre-upload cost estimate into the same progress stream the bar
 * reads. The estimate runs (and finishes) before upload() is ever called, so
 * the upload hook is still idle while it works; this presents it as the
 * estimating stage so estimate, upload and graph build all report in one place.
 * A no-op once the upload leg has started (upload state always wins).
 */
export function withEstimateStage(progress: UploadProgress, isEstimating: boolean): UploadProgress {
  if (!isEstimating || progress.stage !== "idle") return progress;
  return { ...IDLE_PROGRESS, stage: "estimating" };
}

/** The line shown next to the bar. Never a bare spinner — always counts. */
export function describeProgress(progress: UploadProgress): string {
  switch (progress.stage) {
    case "estimating":
      // No counter by design: "x/y files analyzed" reads as busywork for a
      // step that is usually quick. The bar's motion is the signal.
      return "Estimating cost…";
    case "resuming":
      return `Resuming upload — ${progress.filesCompleted}/${progress.filesTotal} files already sent`;
    case "uploading":
      // No "x of n" here. Files are acknowledged ten at a time, so that counter
      // sat at 0/50 and then jumped to 10/50 — it looked stuck, and made a
      // working upload read as a broken one. The byte figures move continuously
      // and say the same thing without pretending to a precision we don't have.
      return `Uploading ${progress.filesTotal} ${progress.filesTotal === 1 ? "file" : "files"} — ${formatBytes(progress.bytesSent)} of ${formatBytes(progress.bytesTotal)}`;
    case "processing":
      // Never claim "all N" when N didn't get there. A session whose blobs
      // failed to persist reaches this stage with files still unsent, and
      // saying they were uploaded is the lie that made the loss silent.
      if (progress.unrecoverableFiles && progress.unrecoverableFiles > 0) {
        return `Building the knowledge graph from ${progress.filesCompleted} of ${progress.filesTotal} files — ${progress.unrecoverableFiles} couldn't be uploaded and need re-selecting`;
      }
      return `Building the knowledge graph — all ${progress.filesTotal} files uploaded`;
    default:
      return "";
  }
}
