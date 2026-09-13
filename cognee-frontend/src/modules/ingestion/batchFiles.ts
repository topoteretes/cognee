import { FILES_PER_BATCH, MAX_BATCH_BYTES } from "./uploadLimits";

export interface FileBatch {
  index: number;
  files: File[];
  bytes: number;
}

/**
 * Split a selection into upload batches bounded by both file count and total
 * bytes. Files are never split across batches — the backend needs whole files —
 * so a single file larger than `maxBytes` becomes a batch of one rather than
 * being rejected or merged into an oversized request.
 *
 * Order is preserved, which is what makes resume-after-refresh cheap: the
 * pending remainder is always a suffix of the original selection.
 */
export default function batchFiles(
  files: File[],
  { maxFiles = FILES_PER_BATCH, maxBytes = MAX_BATCH_BYTES }: { maxFiles?: number; maxBytes?: number } = {},
): FileBatch[] {
  const batches: FileBatch[] = [];
  let current: File[] = [];
  let currentBytes = 0;

  const flush = (): void => {
    if (current.length === 0) return;
    batches.push({ index: batches.length, files: current, bytes: currentBytes });
    current = [];
    currentBytes = 0;
  };

  for (const file of files) {
    // Close the open batch first when adding this file would breach either
    // bound — but only if it has something in it, so an oversized single file
    // still gets its own batch instead of an empty one.
    const breachesCount = current.length >= maxFiles;
    const breachesBytes = current.length > 0 && currentBytes + file.size > maxBytes;
    if (breachesCount || breachesBytes) {
      flush();
    }
    current.push(file);
    currentBytes += file.size;
  }
  flush();

  return batches;
}

export function totalBytes(files: File[]): number {
  return files.reduce((sum, file) => sum + file.size, 0);
}

// A per-file transferred count was tried here and removed. It is derivable in
// principle — multipart bodies are ordered, so byte progress maps onto files —
// but not in practice at this batch size: `xhr.upload.onprogress` reports bytes
// handed to the OS socket buffer, and a batch of ten ordinary documents is
// small enough to buffer in a single shot, so the only event that arrives is
// already at 100%. There is nothing to interpolate between, and the counter
// still moved a whole batch at a time. Real per-file progress needs either
// one-file batches (200 round trips instead of 20, and 200 pipeline runs) or a
// backend that reports per-file acceptance.
