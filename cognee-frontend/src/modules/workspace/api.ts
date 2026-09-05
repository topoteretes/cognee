import localFetch from "@/modules/instances/localFetch";

export type Permission = "read" | "write" | "share" | "delete";
export interface Dataset { id: string; name: string; owner_id: string; permissions: Permission[] }
export interface Team { id: string; name: string; is_owner: boolean }
export interface WorkspaceContext {
  user: { id: string; email: string; tenant_id: string | null; is_agent: boolean };
  teams: Team[];
  datasets: Dataset[];
  providers: { provider: string; configured: boolean; missing_settings: string[] }[];
}
export interface Principal {
  id: string; name: string; kind: string; owner: boolean;
  direct: Permission[]; inherited: Permission[]; effective: Permission[];
}
export interface Access { dataset_id: string; principals: Principal[] }
export interface Promotion {
  status: "planned" | "copied" | "already_promoted";
  source_revision: string; target_data_id: string; target_dataset_id: string;
}

export async function request<T>(path: string, method = "GET", body?: unknown): Promise<T> {
  const response = await localFetch(path, {
    method,
    ...(body === undefined ? {} : { body: JSON.stringify(body), headers: { "Content-Type": "application/json" } }),
  });
  return response.json() as Promise<T>;
}

export function describeError(error: unknown): string {
  return error instanceof Error ? error.message : "The request failed. Refresh and try again.";
}

export async function previewDocument(datasetId: string, documentId: string) {
  const response = await localFetch(`/v1/datasets/${datasetId}/data/${documentId}/raw`);
  const limit = 64 * 1024 * 1024;
  const reader = response.body?.getReader();
  if (!reader) throw new Error("The server returned no document content");
  const chunks: Uint8Array[] = [];
  let size = 0;
  try {
    while (true) {
      const chunk = await reader.read();
      if (chunk.done) break;
      size += chunk.value.byteLength;
      if (size > limit) throw new Error("The document is too large to preview and promote (64 MiB limit)");
      chunks.push(chunk.value);
    }
  } finally { await reader.cancel(); }
  const bytes = new Uint8Array(size);
  let offset = 0;
  for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.byteLength; }
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  const revision = Array.from(new Uint8Array(digest), (v) => v.toString(16).padStart(2, "0")).join("");
  return { revision, text: new TextDecoder().decode(bytes.slice(0, 16000)), size, truncated: size > 16000 };
}
