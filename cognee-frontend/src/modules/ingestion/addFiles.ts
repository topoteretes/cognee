import { captureException } from "@/utils/monitoring";
import { CogneeInstance } from "../instances/types";

export interface AddFilesOptions {
  timeoutMs?: number;
  // Bytes of this request's body that have reached the network. Only fires on
  // the XHR path (instance.upload); the fetch fallback cannot measure it.
  onProgress?: (bytesSent: number, bytesTotal: number) => void;
  signal?: AbortSignal;
}

// The add pipeline's own run — one per request. Its shape is cognee's
// PipelineRunInfo, of which only the id is read here.
export interface AddFilesResponse {
  pipeline_run_id?: string | null;
  dataset_id?: string | null;
  dataset_name?: string | null;
  status?: string;
  error?: string;
}

// Upload-only ingestion: stores the files in the dataset and returns. It does
// NOT build the knowledge graph — that is a separate, single call to
// cognifyDataset once every batch has landed. Splitting the two is deliberate:
//
// The previous flow sent each batch to /v1/remember, which runs add + cognify
// per request. With a 164-file selection that is 17 batches, and so 17 cognify
// runs, every one of which does the full pipeline setup and then queues on the
// dataset's single lock — while the batches still uploading fight those runs'
// LLM work for the same single-process pod. Uploading is I/O; building the
// graph is LLM-bound. Interleaving them made the upload phase pay for both.
//
// /v1/add also carries a far cheaper admission check than /v1/remember: the
// storage/document quota is two aggregate queries, where remember's credit
// guard runs cognee's loader over every file to count tokens before it will
// accept the request. That estimate now happens once, at cognify time, over
// the whole dataset instead of ten files at a time.
//
// Deliberately never sends session_id — that would divert the data into the
// session cache instead of direct ingestion.
export default async function addFiles(
  dataset: { id?: string; name?: string },
  files: File[],
  instance: CogneeInstance,
  options?: AddFilesOptions,
): Promise<AddFilesResponse> {
  // A factory, not a value: a FormData is consumed when sent, so a retried
  // attempt (the upload path retries 429s) needs a freshly built body.
  const buildFormData = (): FormData => {
    const formData = new FormData();
    files.forEach((file) => {
      formData.append("data", file, file.name);
    });
    if (dataset.id) {
      formData.append("datasetId", dataset.id);
    }
    if (dataset.name) {
      formData.append("datasetName", dataset.name);
    }
    return formData;
  };

  // A batch is bounded by MAX_BATCH_BYTES, so this is a transfer-time budget
  // for one request, not a processing budget: /v1/add returns once the files
  // are stored, and the graph build is started and polled separately.
  const timeoutMs = options?.timeoutMs ?? 5 * 60 * 1000;
  const totalBytes = files.reduce((sum, f) => sum + f.size, 0);

  try {
    // Prefer the XHR path: it is the only one that can report bytes leaving the
    // browser. Fall back to fetch when the instance doesn't provide it (test
    // doubles, non-pod instances) — the request is identical, only progress is
    // coarser.
    const response = instance.upload
      ? await instance.upload("/v1/add", buildFormData, {
          timeoutMs,
          signal: options?.signal,
          onProgress: options?.onProgress,
        })
      : await instance.fetch("/v1/add", {
          method: "POST",
          body: buildFormData(),
          timeoutMs,
          signal: options?.signal,
        });
    // instance.fetch already throws HttpError on a non-2xx response, so
    // response.ok is always true here. This catches the soft failure the pod's
    // /add route uses for a failed pipeline: HTTP 200/409 with an app-level
    // error in the body rather than a transport-level status.
    const body = await response.json();
    if (body?.error) {
      throw new Error(body.error);
    }
    return body as AddFilesResponse;
  } catch (err) {
    const context = {
      datasetId: dataset.id,
      fileCount: files.length,
      totalBytes,
      fileTypes: files.map((f) => f.type),
    };

    // normalizeError (@/services/http/errors) converts the client's internal
    // timeout abort into a plain Error with this exact message — there's no
    // dedicated error class or name to check instead.
    if (err instanceof Error && err.message === "Request timed out.") {
      // Names elapsed transfer time, not size (CLO-492): this fires on a clock,
      // and blaming file size sends people to shrink files that were never the
      // problem. The client stopped waiting; the server may still be storing
      // them, which is why callers must not report data loss here.
      const timeoutErr = new Error(
        `Upload stopped waiting after ${Math.round(timeoutMs / 1000)}s (${files.length} file(s), ${Math.round(totalBytes / 1024)}KB). The server may still be processing them.`,
      );
      timeoutErr.name = "UploadTimeoutError";
      captureException(timeoutErr, { ...context, timeoutMs });
      throw timeoutErr;
    }

    captureException(err instanceof Error ? err : new Error(String(err)), context);
    throw err;
  }
}
