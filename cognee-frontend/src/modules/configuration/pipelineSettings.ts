import type { CogneeInstance } from "@/modules/instances/types";

export interface PipelineSettings {
  chunkSize: number;
  chunksPerBatch: number;
  topK: number;
  includeReferences: boolean;
}

const STORAGE_KEY = "cognee-pipeline-settings";
const CONFIG_NAME = "pipeline-settings";

export const DEFAULT_PIPELINE_SETTINGS: PipelineSettings = {
  chunkSize: 1024,
  chunksPerBatch: 10,
  topK: 20,
  includeReferences: true,
};

export function getPipelineSettingsFromStorage(): PipelineSettings {
  if (typeof window === "undefined") {
    return { ...DEFAULT_PIPELINE_SETTINGS };
  }
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return { ...DEFAULT_PIPELINE_SETTINGS };
    const parsed = JSON.parse(raw);
    return { ...DEFAULT_PIPELINE_SETTINGS, ...parsed };
  } catch {
    return { ...DEFAULT_PIPELINE_SETTINGS };
  }
}

export function storePipelineSettingsLocally(settings: PipelineSettings): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
  } catch {
    /* ignore */
  }
}

export async function loadPipelineSettings(
  instance: CogneeInstance,
): Promise<PipelineSettings | null> {
  try {
    const response = await instance.fetch("/configuration/get_user_configuration/");
    if (!response.ok) return null;
    const data = await response.json();
    if (!Array.isArray(data)) return null;
    const entry = data.find(
      (item: { name: string; configuration: object }) => item.name === CONFIG_NAME,
    );
    if (!entry?.configuration) return null;
    return { ...DEFAULT_PIPELINE_SETTINGS, ...(entry.configuration as Partial<PipelineSettings>) };
  } catch {
    return null;
  }
}

export async function savePipelineSettings(
  instance: CogneeInstance,
  settings: PipelineSettings,
): Promise<void> {
  const response = await instance.fetch("/configuration/store_user_configuration", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name: CONFIG_NAME, config: settings }),
  });

  if (!response.ok) {
    throw new Error(`Pipeline settings store failed: ${response.status}`);
  }

  storePipelineSettingsLocally(settings);
}
