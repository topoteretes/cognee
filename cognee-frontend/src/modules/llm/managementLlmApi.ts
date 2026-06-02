import { CogneeInstance } from "@/modules/instances/types";

export interface InferSchemaResult {
  graphSchema: Record<string, unknown>;
}

export async function inferSchema(
  instance: CogneeInstance,
  text?: string,
  files?: File[],
): Promise<InferSchemaResult> {
  const formData = new FormData();
  if (text) formData.append("text", text);
  if (files) {
    for (const file of files) {
      formData.append("data", file, file.name);
    }
  }
  const resp = await instance.fetch("/v1/llm/infer-schema", {
    method: "POST",
    body: formData,
  });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ error: resp.statusText }));
    throw new Error(err.error || err.detail || `${resp.status} ${resp.statusText}`);
  }
  return resp.json();
}

export async function downloadRawData(
  instance: CogneeInstance,
  datasetId: string,
  dataId: string,
): Promise<{ blob: Blob; filename: string }> {
  const resp = await instance.fetch(`/v1/datasets/${datasetId}/data/${dataId}/raw`);
  if (!resp.ok) {
    throw new Error(`Failed to download file: ${resp.status}`);
  }
  const disposition = resp.headers.get("Content-Disposition") ?? "";
  const match = disposition.match(/filename="?([^"]+)"?/);
  const filename = match?.[1] ?? dataId;
  const blob = await resp.blob();
  return { blob, filename };
}

export interface CustomPromptResult {
  customPrompt: string;
}

export async function generateCustomPrompt(instance: CogneeInstance, graphModel: object): Promise<CustomPromptResult> {
  const resp = await instance.fetch("/v1/llm/custom-prompt", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ graph_model: graphModel, parameters: {} }),
  });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ error: resp.statusText }));
    throw new Error(err.error || err.detail || `${resp.status} ${resp.statusText}`);
  }
  return resp.json();
}
