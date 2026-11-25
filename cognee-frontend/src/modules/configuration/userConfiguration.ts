// Public copy of the SaaS module of the same path. The directory is excluded
// from the sync (it contains cloud-only billing/config actions), but this file
// itself is portable and its consumers are shared UI — keep it identical to the
// SaaS original below this header; a unit test fails when it drifts.
import type { CogneeInstance } from "@/modules/instances/types";

export interface UserConfiguration {
  name: string;
  config: object;
}

interface UserConfigurationResponse {
  name: string;
  configuration: object;
  id?: string;
  ownerId?: string;
  createdAt?: string;
  updatedAt?: string;
}

const MEMORY_SETTINGS_CONFIG_NAME = "memory-settings";
const GRAPH_MODELS_CONFIG_NAME = "graph-models";

// The HTTP client's own timeout ends when the response headers arrive, so a
// body that stalls after that would hang a read for good. This signal is merged
// into the request, and aborting it rejects the body read as well. Every write
// waits behind the one before it (see inWriteQueue), so one hung request must
// not be able to hold all of them.
const CONFIG_REQUEST_TIMEOUT_MS = 30_000;

/** Rejects when the configuration cannot be read, so a caller can tell "the
 *  read failed" apart from "nothing is stored". */
async function fetchUserConfigurations(
  instance: CogneeInstance,
): Promise<UserConfiguration[]> {
  const response = await instance.fetch(
    "/configuration/get_user_configuration/",
    { signal: AbortSignal.timeout(CONFIG_REQUEST_TIMEOUT_MS) },
  );
  if (!response.ok) throw new Error(`Config read failed: ${response.status}`);
  const data = await response.json();
  if (!Array.isArray(data)) throw new Error("Config read returned a non-array");
  return data.map((item: UserConfigurationResponse) => ({
    name: item.name,
    config: item.configuration,
  }));
}

/** Lenient read for callers where "no configuration" is a safe fallback. */
async function getUserConfigurations(
  instance: CogneeInstance,
): Promise<UserConfiguration[]> {
  try {
    return await fetchUserConfigurations(instance);
  } catch {
    return [];
  }
}

async function storeUserConfiguration(
  instance: CogneeInstance,
  name: string,
  config: object,
): Promise<void> {
  const body = { name, config };
  const resp = await instance.fetch("/configuration/store_user_configuration", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(CONFIG_REQUEST_TIMEOUT_MS),
  });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({}));
    console.error("[Config] Store failed:", resp.status, err);
    throw new Error(`Config store failed: ${resp.status}`);
  }
}

/**
 * Store a single config by name. The backend looks the row up by (owner_id,
 * name) and updates it, or inserts one when there is none. Nothing enforces that
 * pair as unique, so two first-time stores racing can create two rows; sending
 * only the one config is still right.
 */
async function safeUpdateConfig(
  instance: CogneeInstance,
  name: string,
  config: object,
): Promise<void> {
  await storeUserConfiguration(instance, name, config);
}

export async function syncMemorySettings(
  instance: CogneeInstance,
  settings: object,
): Promise<void> {
  await safeUpdateConfig(instance, MEMORY_SETTINGS_CONFIG_NAME, settings);
}

export async function loadMemorySettingsFromBackend(
  instance: CogneeInstance,
): Promise<object | null> {
  const all = await getUserConfigurations(instance);
  const entry = all.find((c) => c.name === MEMORY_SETTINGS_CONFIG_NAME);
  return entry?.config ?? null;
}

// ── Graph model ↔ dataset assignments ────────────────────────────────────────

import type { GraphModel } from "@/modules/graphModels/types";

/**
 * customPrompts is a simple dict: { "Prompt Name": "prompt text" }
 * This matches the backend config format exactly.
 */
export type CustomPromptsMap = Record<string, string>;

/** Maps datasetId → promptName. One prompt can appear for multiple datasets. */
export type PromptAssignmentsMap = Record<string, string>;

/** Maps datasetId → ontologyKey. One ontology can serve multiple datasets. */
export type OntologyAssignmentsMap = Record<string, string>;

export interface GraphModelsConfig {
  models: GraphModel[];
  customPrompts?: CustomPromptsMap;
  promptAssignments?: PromptAssignmentsMap;
  ontologyAssignments?: OntologyAssignmentsMap;
  outdatedDatasets?: string[];
}

function toGraphModelsConfig(all: UserConfiguration[]): GraphModelsConfig {
  const entry = all.find((c) => c.name === GRAPH_MODELS_CONFIG_NAME);
  if (!entry?.config) return { models: [], customPrompts: {}, promptAssignments: {}, ontologyAssignments: {}, outdatedDatasets: [] };
  const cfg = entry.config as GraphModelsConfig;
  return { models: cfg.models ?? [], customPrompts: cfg.customPrompts ?? {}, promptAssignments: cfg.promptAssignments ?? {}, ontologyAssignments: cfg.ontologyAssignments ?? {}, outdatedDatasets: cfg.outdatedDatasets ?? [] };
}

/** Empty config when the read fails. Only for readers that can do without it:
 *  a writer that merges into this and stores the result would overwrite every
 *  prompt and assignment with an empty set. Writers use fetchGraphModelsConfig. */
export async function loadGraphModelsConfig(
  instance: CogneeInstance,
): Promise<GraphModelsConfig> {
  return toGraphModelsConfig(await getUserConfigurations(instance));
}

/** Rejects when the read fails. For callers that must not mistake an outage
 *  for "nothing is stored": a missing model, or the base of a read-modify-write. */
export async function fetchGraphModelsConfig(
  instance: CogneeInstance,
): Promise<GraphModelsConfig> {
  return toGraphModelsConfig(await fetchUserConfigurations(instance));
}

export async function loadGraphModelsFromBackend(
  instance: CogneeInstance,
): Promise<GraphModel[]> {
  const cfg = await loadGraphModelsConfig(instance);
  return cfg.models;
}

function buildGraphModelsPayload(cfg: GraphModelsConfig): object {
  return {
    models: cfg.models,
    customPrompts: cfg.customPrompts ?? {},
    promptAssignments: cfg.promptAssignments ?? {},
    ontologyAssignments: cfg.ontologyAssignments ?? {},
    outdatedDatasets: cfg.outdatedDatasets ?? [],
  };
}

// Every writer below reads the whole stored configuration, changes one part of
// it and stores it back. Two of them running at once would both start from the
// same read, and whichever stored last would silently drop the other's change
// (picking a prompt right after an ontology lost the ontology). So within this
// tab they run one after another per workspace, each reading what the previous
// one stored. Another tab can still interleave; only the backend can order that.
// A write that timed out (see CONFIG_REQUEST_TIMEOUT_MS) has also left the queue
// but may still be applied by the pod after the next one: the ordering holds for
// writes that got an answer.
const writeQueues = new WeakMap<CogneeInstance, Promise<void>>();

function inWriteQueue(instance: CogneeInstance, write: () => Promise<void>): Promise<void> {
  const run = (writeQueues.get(instance) ?? Promise.resolve()).then(write);
  // The queue goes on after a failed write; the caller still sees the failure.
  writeQueues.set(instance, run.catch(() => {}));
  return run;
}

/**
 * Reads the stored configuration, lets `change` edit it in place and stores the
 * result, as one queued step. A failed read stores nothing (see
 * fetchGraphModelsConfig). `change` returns false when there is nothing to store.
 */
function updateConfig(instance: CogneeInstance, change: (cfg: GraphModelsConfig) => boolean | void): Promise<void> {
  return inWriteQueue(instance, async () => {
    const cfg = await fetchGraphModelsConfig(instance);
    if (change(cfg) === false) return;
    await safeUpdateConfig(instance, GRAPH_MODELS_CONFIG_NAME, buildGraphModelsPayload(cfg));
  });
}

/** The datasets whose assignment in `assignments` is `value`. */
export function datasetsAssignedTo(assignments: Record<string, string> | undefined, value: string): string[] {
  return Object.entries(assignments ?? {}).filter(([, assigned]) => assigned === value).map(([datasetId]) => datasetId);
}

function flagOutdated(cfg: GraphModelsConfig, datasetIds: string[]): void {
  cfg.outdatedDatasets = [...new Set([...(cfg.outdatedDatasets ?? []), ...datasetIds])];
}

function withAssignment(assignments: Record<string, string> | undefined, datasetId: string, value: string | null): Record<string, string> {
  const next = { ...(assignments ?? {}) };
  if (value) next[datasetId] = value;
  else delete next[datasetId];
  return next;
}

/** `assignments` without the datasets that point at `value`. */
export function withoutValue(assignments: Record<string, string> | undefined, value: string): Record<string, string> {
  return Object.fromEntries(Object.entries(assignments ?? {}).filter(([, assigned]) => assigned !== value));
}

/**
 * Stores a model as edited, keeping what is stored about it (when it was
 * created, which datasets use it). A changed schema makes the graphs of those
 * datasets outdated, as new wording does for a prompt. The schemas are compared
 * as JSON: a difference in key order alone flags them too, which errs toward a
 * needless rebuild rather than a graph that is silently out of date.
 */
export function saveGraphModel(
  instance: CogneeInstance,
  edited: Pick<GraphModel, "id" | "name" | "schema">,
  savedAt: string,
): Promise<void> {
  return updateConfig(instance, (cfg) => {
    const stored = cfg.models.find((m) => m.id === edited.id);
    const model: GraphModel = {
      ...edited,
      createdAt: stored?.createdAt ?? savedAt,
      updatedAt: savedAt,
      status: "draft",
      assignedDatasets: stored?.assignedDatasets,
    };
    cfg.models = stored ? cfg.models.map((m) => (m.id === edited.id ? model : m)) : [...cfg.models, model];
    if (stored && JSON.stringify(stored.schema) !== JSON.stringify(edited.schema)) {
      flagOutdated(cfg, stored.assignedDatasets ?? []);
    }
  });
}

/** A dataset uses one graph model: take it off whichever one had it. */
function unassignDataset(models: GraphModel[], datasetId: string): void {
  for (const model of models) {
    if (model.assignedDatasets) {
      model.assignedDatasets = model.assignedDatasets.filter((id) => id !== datasetId);
    }
  }
}

/** Adds a model that `datasetId` then uses, and takes the dataset off the model it used before. */
export function addGraphModelForDataset(
  instance: CogneeInstance,
  model: GraphModel,
  datasetId: string,
): Promise<void> {
  return updateConfig(instance, (cfg) => {
    unassignDataset(cfg.models, datasetId);
    cfg.models = [...cfg.models, { ...model, assignedDatasets: [datasetId] }];
    flagOutdated(cfg, [datasetId]);
  });
}

/** Deletes a model. Its datasets fall back to the default model, so their graphs are outdated. */
export function deleteGraphModel(
  instance: CogneeInstance,
  modelId: string,
): Promise<void> {
  return updateConfig(instance, (cfg) => {
    flagOutdated(cfg, cfg.models.find((m) => m.id === modelId)?.assignedDatasets ?? []);
    cfg.models = cfg.models.filter((m) => m.id !== modelId);
  });
}

export function assignGraphModelToDataset(
  instance: CogneeInstance,
  datasetId: string,
  modelId: string | null,
): Promise<void> {
  return updateConfig(instance, (cfg) => {
    unassignDataset(cfg.models, datasetId);
    if (modelId !== null) {
      const target = cfg.models.find((m) => m.id === modelId);
      if (target) {
        target.assignedDatasets = [...(target.assignedDatasets ?? []), datasetId];
      }
    }
    flagOutdated(cfg, [datasetId]);
  });
}

export function clearDatasetOutdated(
  instance: CogneeInstance,
  datasetId: string,
): Promise<void> {
  return updateConfig(instance, (cfg) => {
    cfg.outdatedDatasets = (cfg.outdatedDatasets ?? []).filter((id) => id !== datasetId);
  });
}

export function findModelForDataset(
  models: GraphModel[],
  datasetId: string,
): GraphModel | null {
  return models.find((m) => m.assignedDatasets?.includes(datasetId)) ?? null;
}

export function findPromptForDataset(
  assignments: PromptAssignmentsMap,
  datasetId: string,
): string | null {
  return assignments[datasetId] ?? null;
}

export function assignPromptToDataset(
  instance: CogneeInstance,
  datasetId: string,
  promptName: string | null,
): Promise<void> {
  return updateConfig(instance, (cfg) => {
    cfg.promptAssignments = withAssignment(cfg.promptAssignments, datasetId, promptName);
    flagOutdated(cfg, [datasetId]);
  });
}

export function findOntologyForDataset(
  assignments: OntologyAssignmentsMap,
  datasetId: string,
): string | null {
  return assignments[datasetId] ?? null;
}

export function assignOntologyToDataset(
  instance: CogneeInstance,
  datasetId: string,
  ontologyKey: string | null,
): Promise<void> {
  return updateConfig(instance, (cfg) => {
    cfg.ontologyAssignments = withAssignment(cfg.ontologyAssignments, datasetId, ontologyKey);
    flagOutdated(cfg, [datasetId]);
  });
}

export function saveCustomPrompt(
  instance: CogneeInstance,
  name: string,
  promptText: string,
): Promise<void> {
  return updateConfig(instance, (cfg) => {
    const previousText = cfg.customPrompts?.[name];
    cfg.customPrompts = { ...(cfg.customPrompts ?? {}), [name]: promptText };
    // New wording makes every graph built from the old one outdated, in each
    // dataset that uses this prompt, not only the one being edited.
    if (previousText !== undefined && previousText !== promptText) {
      flagOutdated(cfg, datasetsAssignedTo(cfg.promptAssignments, name));
    }
  });
}

export function deleteCustomPrompt(
  instance: CogneeInstance,
  name: string,
): Promise<void> {
  return updateConfig(instance, (cfg) => {
    const prompts = { ...(cfg.customPrompts ?? {}) };
    delete prompts[name];
    cfg.customPrompts = prompts;
    // A dataset must not keep pointing at a prompt that no longer exists; its
    // next build falls back to the default, so its graph is outdated.
    flagOutdated(cfg, datasetsAssignedTo(cfg.promptAssignments, name));
    cfg.promptAssignments = withoutValue(cfg.promptAssignments, name);
  });
}

/**
 * Call after an ontology is uploaded. A dataset can still point at its key (a
 * delete whose unassign failed, or another tab), and it builds from the new file
 * now, so its graph is outdated.
 */
export function flagDatasetsUsingOntology(
  instance: CogneeInstance,
  key: string,
): Promise<void> {
  return updateConfig(instance, (cfg) => {
    const affected = datasetsAssignedTo(cfg.ontologyAssignments, key);
    if (affected.length === 0) return false;
    flagOutdated(cfg, affected);
  });
}

/** Call after an ontology is deleted: drops every dataset's assignment of it, as deleteCustomPrompt does for a prompt. */
export function removeOntologyAssignments(
  instance: CogneeInstance,
  key: string,
): Promise<void> {
  return updateConfig(instance, (cfg) => {
    const affected = datasetsAssignedTo(cfg.ontologyAssignments, key);
    if (affected.length === 0) return false;
    flagOutdated(cfg, affected);
    cfg.ontologyAssignments = withoutValue(cfg.ontologyAssignments, key);
  });
}
