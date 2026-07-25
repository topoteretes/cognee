import { captureException } from "@/utils/monitoring";
import { CogneeInstance } from "../instances/types";
import { getPipelineSettingsFromStorage } from "../configuration/pipelineSettings";

export interface RememberOptions {
  graphModel?: object;
  customPrompt?: string;
  ontologyKey?: string[];
  chunkSize?: number;
  chunksPerBatch?: number;
  runInBackground?: boolean;
  timeoutMs?: number;
}

export interface RememberResponse {
  status: string;
  dataset_name: string | null;
  dataset_id: string | null;
  pipeline_run_id: string | null;
  error?: string;
}

// Single-call ingestion: uploads files and builds the knowledge graph
// (add + cognify in one request). Replaces the legacy two-step flow.
// Deliberately never sends session_id — that would divert the data into
// the session cache instead of direct ingestion.
export default async function rememberData(
  dataset: { id?: string; name?: string },
  files: File[],
  instance: CogneeInstance,
  options?: RememberOptions,
): Promise<RememberResponse> {
  const pipelineSettings = getPipelineSettingsFromStorage();
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
  if (options?.graphModel) {
    formData.append("graph_model", JSON.stringify(options.graphModel));
  }
  if (options?.customPrompt) {
    formData.append("custom_prompt", options.customPrompt);
  }
  for (const key of options?.ontologyKey ?? []) {
    formData.append("ontology_key", key);
  }
  formData.append("chunk_size", String(options?.chunkSize ?? pipelineSettings.chunkSize));
  formData.append("chunks_per_batch", String(options?.chunksPerBatch ?? pipelineSettings.chunksPerBatch));
  // Block until the knowledge graph is fully built — no polling needed.
  // Callers can override with runInBackground: true for fire-and-forget use cases.
  formData.append("run_in_background", String(options?.runInBackground ?? false));

  // Large uploads legitimately take longer than the shared http client's
  // default POST timeout (30s) — pass timeoutMs through so the client's own
  // AbortController uses this window instead. A manually-built AbortController
  // here would NOT help: the shared client always races its own 30s default
  // timeout signal against any caller signal via AbortSignal.any and takes
  // whichever fires first, so without timeoutMs every upload over ~30s was
  // silently aborted by the client's default long before this file's intended
  // 5-minute allowance ever kicked in.
  const timeoutMs = options?.timeoutMs ?? 5 * 60 * 1000;
  const totalBytes = files.reduce((sum, f) => sum + f.size, 0);

  try {
    const response = await instance.fetch("/v1/remember", {
      method: "POST",
      body: formData,
      timeoutMs,
    });
    // instance.fetch already throws HttpError on a non-2xx response (see
    // @/services/http/client), so response.ok is always true here — this
    // only catches a soft failure: HTTP 200 with an app-level error in the body.
    const body = await response.json();
    if (body?.error) {
      throw new Error(body.error);
    }
    return body as RememberResponse;
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
      const timeoutErr = new Error(`Upload timed out after ${timeoutMs / 1000}s (${files.length} file(s), ${Math.round(totalBytes / 1024)}KB total)`);
      timeoutErr.name = "UploadTimeoutError";
      captureException(timeoutErr, { ...context, timeoutMs });
      throw timeoutErr;
    }

    captureException(err instanceof Error ? err : new Error(String(err)), context);
    throw err;
  }
}
