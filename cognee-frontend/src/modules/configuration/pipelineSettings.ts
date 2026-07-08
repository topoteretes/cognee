import type { CogneeInstance } from "@/modules/instances/types";

export interface PipelineSettings {
  chunkSize: number;
  chunksPerBatch: number;
  topK: number;
  // When true, recall/search responses include source/provenance references
  // attached to completions. Defaults to true so cited answers are the
  // out-of-the-box experience; can be turned off in Extraction Settings.
  includeReferences: boolean;
}

export const DEFAULT_PIPELINE_SETTINGS: PipelineSettings = {
  chunkSize: 4096,
  chunksPerBatch: 100,
  topK: 30,
  includeReferences: true,
};

const PIPELINE_SETTINGS_CONFIG_NAME = "pipeline-settings";
const STORAGE_KEY = "cognee-pipeline-settings";

export function getPipelineSettingsFromStorage(): PipelineSettings {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      return { ...DEFAULT_PIPELINE_SETTINGS, ...(JSON.parse(raw) as Partial<PipelineSettings>) };
    }
  } catch {
    // corrupted or unavailable — return defaults
  }
  return { ...DEFAULT_PIPELINE_SETTINGS };
}

export function storePipelineSettingsLocally(settings: PipelineSettings): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
  } catch {
    // storage unavailable — ignore
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
    const entry = data.find((c: { name: string }) => c.name === PIPELINE_SETTINGS_CONFIG_NAME);
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
  const resp = await instance.fetch("/configuration/store_user_configuration", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name: PIPELINE_SETTINGS_CONFIG_NAME, config: settings }),
  });
  if (!resp.ok) {
    throw new Error(`Failed to save pipeline settings: ${resp.status}`);
  }
  storePipelineSettingsLocally(settings);
}
