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

  // Abort the request if it hangs — default 5 min for large files
  const timeoutMs = options?.timeoutMs ?? 5 * 60 * 1000;
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  const totalBytes = files.reduce((sum, f) => sum + f.size, 0);

  try {
    const response = await instance.fetch("/v1/remember", {
      method: "POST",
      body: formData,
      signal: controller.signal,
    });
    const body = await response.json();
    if (!response.ok || body?.error) {
      const err = new Error(body?.error || `Remember failed (HTTP ${response.status})`);
      captureException(err, {
        datasetId: dataset.id,
        fileCount: files.length,
        totalBytes,
        fileTypes: files.map((f) => f.type),
        httpStatus: response.status,
      });
      throw err;
    }
    return body as RememberResponse;
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      const timeoutErr = new Error(`Upload timed out after ${timeoutMs / 1000}s (${files.length} file(s), ${Math.round(totalBytes / 1024)}KB total)`);
      timeoutErr.name = "UploadTimeoutError";
      captureException(timeoutErr, {
        datasetId: dataset.id,
        fileCount: files.length,
        totalBytes,
        fileTypes: files.map((f) => f.type),
        timeoutMs,
      });
      throw timeoutErr;
    }
    throw err;
  } finally {
    clearTimeout(timeoutId);
  }
}
